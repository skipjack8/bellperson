use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};

use super::msm;
use crate::bls::Engine;
use crate::groth16::aggregate::commit::*;
use crate::groth16::multiscalar::{precompute_fixed_window, MultiscalarPrecompOwned, WINDOW_SIZE};

/// SRS is the partial combination of two Groth16 CRS of size 2n.
#[derive(Clone, Debug)]
pub struct SRS<E: Engine> {
    /// number of proofs to aggregate
    pub n: usize,
    /// $\{g^a^i\}_{i=0}^{2n}$ where n is the number of proofs to be aggregated
    pub g_alpha_powers: Vec<E::G1Affine>,
    pub g_alpha_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^a^i\}_{i=0}^{2n}$ where n is the number of proofs to be aggregated
    pub h_alpha_powers: Vec<E::G2Affine>,
    pub h_alpha_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    /// $\{g^b^i\}_{i=0}^{2n}$ where n is the number of proofs to be aggregated
    pub g_beta_powers: Vec<E::G1Affine>,
    pub g_beta_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    /// $\{h^b^i\}_{i=0}^{2n}$ where n is the number of proofs to be aggregated
    pub h_beta_powers: Vec<E::G2Affine>,
    pub h_beta_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
}

#[derive(Clone, Debug)]
pub struct VerifierSRS<E: Engine> {
    pub g: E::G1,
    pub h: E::G2,
    // TODO look if g_alpha and g_beta still needed
    pub g_alpha: E::G1,
    pub g_beta: E::G1,
    pub h_alpha: E::G2,
    pub h_beta: E::G2,
    /// equals to $g^{alpha^{n+1}}$
    pub g_alpha_n: E::G1,
    /// equals to $g^{beta^{n+1}}$
    pub g_beta_n: E::G1,
}

impl<E: Engine> SRS<E> {
    pub fn get_commitment_keys(&self) -> (VKey<E>, WKey<E>) {
        (self.get_vkey(), self.get_wkey())
    }

    pub fn get_vkey(&self) -> VKey<E> {
        let v1 = self
            .h_alpha_powers
            .iter()
            .skip(1) // skip the h
            .take(self.n)
            .map(|p| p.into_projective())
            .collect::<Vec<E::G2>>();
        let v2 = self
            .h_beta_powers
            .par_iter()
            .skip(1) // skip the h
            .take(self.n)
            .map(|p| p.into_projective())
            .collect::<Vec<E::G2>>();
        VKey { a: v1, b: v2 }
    }

    pub fn get_wkey(&self) -> WKey<E> {
        // +1 because we skip first g then g^1...g^n
        let w1 = self
            .g_alpha_powers
            .par_iter()
            .skip(self.n + 1)
            .map(|p| p.into_projective())
            .collect::<Vec<_>>();
        let w2 = self
            .g_beta_powers
            .par_iter()
            .skip(self.n + 1)
            .map(|p| p.into_projective())
            .collect::<Vec<_>>();
        WKey { a: w1, b: w2 }
    }

    pub fn get_verifier_key(&self) -> VerifierSRS<E> {
        VerifierSRS {
            g: self.g_alpha_powers[0].into_projective(),
            h: self.h_beta_powers[0].into_projective(),
            g_alpha: self.g_alpha_powers[1].clone(),
            h_alpha: self.h_alpha_powers[1].clone(),
            g_beta: self.g_beta_powers[1].clone(),
            h_beta: self.h_beta_powers[1].clone(),
            g_alpha_n: self.g_alpha[1 + n].clone(),
            g_beta_n: self.g_beta[1 + n].clone(),
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
    let mut g_beta_powers = Vec::new();
    let mut h_alpha_powers = Vec::new();
    let mut h_beta_powers = Vec::new();
    rayon::scope(|s| {
        let g = &g;
        let alpha = &alpha;
        let h = &h;
        let beta = &beta;

        let g_alpha_powers = &mut g_alpha_powers;
        s.spawn(move |_| {
            *g_alpha_powers = structured_generators_scalar_power(2 * size, g, alpha);
        });
        let g_beta_powers = &mut g_beta_powers;
        s.spawn(move |_| {
            *g_beta_powers = structured_generators_scalar_power(2 * size, g, beta);
        });

        let h_alpha_powers = &mut h_alpha_powers;
        s.spawn(move |_| {
            *h_alpha_powers = structured_generators_scalar_power(2 * size, h, alpha);
        });

        let h_beta_powers = &mut h_beta_powers;
        s.spawn(move |_| {
            *h_beta_powers = structured_generators_scalar_power(2 * size, h, beta);
        });
    });

    let g_alpha_powers_table = precompute_fixed_window(&g_alpha_powers, WINDOW_SIZE);
    let h_beta_powers_table = precompute_fixed_window(&h_beta_powers, WINDOW_SIZE);
    let g_beta_powers_table = precompute_fixed_window(&g_beta_powers, WINDOW_SIZE);
    let h_alpha_powers_table = precompute_fixed_window(&h_alpha_powers, WINDOW_SIZE);

    SRS {
        g_alpha_powers,
        g_alpha_powers_table,
        h_beta_powers,
        h_beta_powers_table,
        g_beta_powers,
        g_beta_powers_table,
        h_beta_powers,
        h_beta_powers_table,
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
