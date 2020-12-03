use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};

use super::msm;
use crate::bls::Engine;

#[derive(Clone, Debug)]
pub struct SRS<E: Engine> {
    pub g_alpha_powers: Vec<E::G1Affine>,
    pub h_beta_powers: Vec<E::G2Affine>,
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

impl<E: Engine> SRS<E> {
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

    SRS {
        g_alpha_powers: structured_generators_scalar_power(2 * size - 1, &g, &alpha),
        h_beta_powers: structured_generators_scalar_power(2 * size - 1, &h, &beta),
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
