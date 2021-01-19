use std::io::{self, Read, Write};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use rayon::prelude::*;

use super::msm;
use crate::bls::Engine;
use crate::groth16::multiscalar::{precompute_fixed_window, MultiscalarPrecompOwned, WINDOW_SIZE};

#[derive(Clone, Debug)]
pub struct SRS<E: Engine> {
    pub g_alpha_powers: Vec<E::G1Affine>,
    pub g_alpha_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    pub h_beta_powers: Vec<E::G2Affine>,
    pub h_beta_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    pub g_beta: E::G1,
    pub h_alpha: E::G2,
}

#[derive(Clone, Debug)]
pub struct VerifierSRS<E: Engine> {
    pub g: E::G1,
    pub h: E::G2,
    pub g_beta: E::G1,
    pub h_alpha: E::G2,
}

impl<E: Engine> PartialEq for SRS<E> {
    fn eq(&self, other: &Self) -> bool {
        self.g_alpha_powers.len() == other.g_alpha_powers.len()
            && self.h_beta_powers.len() == other.h_beta_powers.len()
            && self.g_alpha_powers_table == other.g_alpha_powers_table
            && self.h_beta_powers_table == other.h_beta_powers_table
            && self.g_beta == other.g_beta
            && self.h_alpha == other.h_alpha
    }
}

impl<E: Engine> PartialEq for VerifierSRS<E> {
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g
            && self.h == other.h
            && self.g_beta == other.g_beta
            && self.h_alpha == other.h_alpha
    }
}

impl<E: Engine> SRS<E> {
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<BigEndian>(self.g_alpha_powers.len() as u32)?;

        for g_alpha_power in &self.g_alpha_powers {
            writer.write_all(g_alpha_power.into_uncompressed().as_ref())?;
        }

        self.g_alpha_powers_table.write::<W>(writer)?;

        writer.write_u32::<BigEndian>(self.h_beta_powers.len() as u32)?;

        for h_beta_power in &self.h_beta_powers {
            writer.write_all(h_beta_power.into_uncompressed().as_ref())?;
        }

        self.h_beta_powers_table.write::<W>(writer)?;

        writer.write_all(self.g_beta.into_affine().into_uncompressed().as_ref())?;
        writer.write_all(self.h_alpha.into_affine().into_uncompressed().as_ref())?;

        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let g1_len = std::mem::size_of::<<E::G1Affine as CurveAffine>::Uncompressed>();
        let g2_len = std::mem::size_of::<<E::G2Affine as CurveAffine>::Uncompressed>();

        let g_alpha_power_len = reader.read_u32::<BigEndian>()? as usize;

        let mut data = vec![0; g_alpha_power_len * g1_len];
        reader.read_exact(&mut data)?;

        let g_alpha_powers = (0..g_alpha_power_len)
            .into_par_iter()
            .map(|i| {
                let data_start = i * g1_len;
                let data_end = data_start + g1_len;
                let ptr = &data[data_start..data_end];

                // Safety: this operation is safe because it's a read on
                // a buffer that's already allocated and being iterated on.
                let g1_repr: <E::G1Affine as CurveAffine>::Uncompressed = unsafe {
                    *(ptr as *const [u8] as *const <E::G1Affine as CurveAffine>::Uncompressed)
                };

                g1_repr.into_affine().unwrap()
            })
            .collect();

        let g_alpha_powers_table = MultiscalarPrecompOwned::<E::G1Affine>::read::<R>(reader)?;

        let h_beta_power_len = reader.read_u32::<BigEndian>()? as usize;

        let mut data = vec![0; h_beta_power_len * g2_len];
        reader.read_exact(&mut data)?;

        let h_beta_powers = (0..h_beta_power_len)
            .into_par_iter()
            .map(|i| {
                let data_start = i * g2_len;
                let data_end = data_start + g2_len;
                let ptr = &data[data_start..data_end];

                // Safety: this operation is safe because it's a read on
                // a buffer that's already allocated and being iterated on.
                let g2_repr: <E::G2Affine as CurveAffine>::Uncompressed = unsafe {
                    *(ptr as *const [u8] as *const <E::G2Affine as CurveAffine>::Uncompressed)
                };

                g2_repr.into_affine().unwrap()
            })
            .collect();

        let h_beta_powers_table = MultiscalarPrecompOwned::<E::G2Affine>::read::<R>(reader)?;

        let mut g1_repr = <E::G1Affine as CurveAffine>::Uncompressed::empty();
        reader.read_exact(g1_repr.as_mut())?;
        let g_beta = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut g2_repr = <E::G2Affine as CurveAffine>::Uncompressed::empty();
        reader.read_exact(g2_repr.as_mut())?;
        let h_alpha = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        Ok(SRS {
            g_alpha_powers,
            g_alpha_powers_table,
            h_beta_powers,
            h_beta_powers_table,
            g_beta: g_beta.into_projective(),
            h_alpha: h_alpha.into_projective(),
        })
    }

    pub fn get_commitment_keys(&self) -> (Vec<E::G2Affine>, Vec<E::G1Affine>) {
        let ck_1 = self.get_ck_1().cloned().collect();
        let ck_2 = self.get_ck_2().cloned().collect();
        (ck_1, ck_2)
    }

    pub fn get_ck_1(&self) -> impl Iterator<Item = &E::G2Affine> {
        self.h_beta_powers.iter().step_by(2)
    }

    pub fn get_ck_2(&self) -> impl Iterator<Item = &E::G1Affine> {
        self.g_alpha_powers.iter().step_by(2)
    }

    pub fn get_verifier_key(&self) -> VerifierSRS<E> {
        VerifierSRS {
            g: self.g_alpha_powers[0].into_projective(),
            h: self.h_beta_powers[0].into_projective(),
            g_beta: self.g_beta.clone(),
            h_alpha: self.h_alpha.clone(),
        }
    }
}

pub fn setup_inner_product<E: Engine, R: rand::RngCore>(rng: &mut R, size: usize) -> SRS<E> {
    let alpha = E::Fr::random(rng);
    let beta = E::Fr::random(rng);
    let g = E::G1::one();
    let mut g_beta = g;
    g_beta.mul_assign(beta);

    let h = E::G2::one();
    let mut h_alpha = h;
    h_alpha.mul_assign(alpha);

    let mut g_alpha_powers = Vec::new();
    let mut h_beta_powers = Vec::new();
    rayon::scope(|s| {
        let g = &g;
        let alpha = &alpha;
        let h = &h;
        let beta = &beta;

        let g_alpha_powers = &mut g_alpha_powers;
        s.spawn(move |_| {
            *g_alpha_powers = structured_generators_scalar_power(2 * size - 1, g, alpha);
        });

        let h_beta_powers = &mut h_beta_powers;
        s.spawn(move |_| {
            *h_beta_powers = structured_generators_scalar_power(2 * size - 1, h, beta);
        });
    });

    let g_alpha_powers_table = precompute_fixed_window(&g_alpha_powers, WINDOW_SIZE);
    let h_beta_powers_table = precompute_fixed_window(&h_beta_powers, WINDOW_SIZE);

    SRS {
        g_alpha_powers,
        g_alpha_powers_table,
        h_beta_powers,
        h_beta_powers_table,
        g_beta,
        h_alpha,
    }
}

fn structured_generators_scalar_power<G: CurveProjective>(
    num: usize,
    g: &G,
    s: &G::Scalar,
) -> Vec<G::Affine> {
    assert!(num > 0);
    let mut powers_of_scalar = Vec::with_capacity(num);
    let mut pow_s = G::Scalar::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s.mul_assign(s);
    }

    let window_size = msm::fixed_base::get_mul_window_size(num);
    let scalar_bits = G::Scalar::NUM_BITS as usize;
    let g_table = msm::fixed_base::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = msm::fixed_base::multi_scalar_mul::<G>(
        scalar_bits,
        window_size,
        &g_table,
        &powers_of_scalar[..],
    );
    powers_of_g.into_iter().map(|v| v.into_affine()).collect()
}
