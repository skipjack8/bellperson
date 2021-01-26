use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};

use super::msm;
use crate::bls::Engine;
use crate::groth16::aggregate::commit::*;
use crate::groth16::multiscalar::{precompute_fixed_window, MultiscalarPrecompOwned, WINDOW_SIZE};
use rayon::prelude::*;

/// SRS is the partial combination of two Groth16 CRS of size 2n.
#[derive(Clone, Debug)]
pub struct SRS<E: Engine> {
    /// number of proofs to aggregate
    pub n: usize,
    /// $\{g^a^i\}_{i=n}^{2n}$ where n is the number of proofs to be aggregated
    /// NOTE in practice we only need the first half of that - it may be worth
    /// doing the logic for that
    pub g_alpha_powers: Vec<E::G1Affine>,
    /// $\{h^a^i\}_{i=0}^{n}$ where n is the number of proofs to be aggregated
    pub h_alpha_powers: Vec<E::G2Affine>,
    /// $\{g^b^i\}_{i=n}^{2n}$ where n is the number of proofs to be aggregated
    pub g_beta_powers: Vec<E::G1Affine>,
    /// $\{h^b^i\}_{i=0}^{n}$ where n is the number of proofs to be aggregated
    pub h_beta_powers: Vec<E::G2Affine>,
}

/// PrecompSRS contains the precomputed tables from the SRS - call
/// `srs.precompute()` to get it. It does starts at h^a and h^b and g^{a^{n+1}}
/// and g^{b^{n+1}} unlike the SRS because that's what is only required for the
/// prover.
#[derive(Clone, Debug)]
pub struct PrecompSRS<E: Engine> {
    pub n: usize,
    /// $\{g^a^i\}_{i=n+1}^{2n}$ where n is the number of proofs to be aggregated
    /// table starts at i=1 since base is offset with commitment keys - it's not g and h
    /// but g^a and h^a
    pub g_alpha_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
    pub h_alpha_powers_table: MultiscalarPrecompOwned<E::G2Affine>,
    pub g_beta_powers_table: MultiscalarPrecompOwned<E::G1Affine>,
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
    pub g_alpha_n1: E::G1,
    /// equals to $g^{beta^{n+1}}$
    pub g_beta_n1: E::G1,
}

impl<E: Engine> SRS<E> {
    pub fn precompute(&self) -> PrecompSRS<E> {
        // we skip the first one since g^a^0 = g which is not part of the commitment
        // key (i.e. we don't use it in the prover's code).
        PrecompSRS {
            n: self.n,
            g_alpha_powers_table: precompute_fixed_window(&self.g_alpha_powers[1..], WINDOW_SIZE),
            h_beta_powers_table: precompute_fixed_window(&self.h_beta_powers[1..], WINDOW_SIZE),
            g_beta_powers_table: precompute_fixed_window(&self.g_beta_powers[1..], WINDOW_SIZE),
            h_alpha_powers_table: precompute_fixed_window(&self.h_alpha_powers[1..], WINDOW_SIZE),
        }
    }
    pub fn get_commitment_keys(&self) -> (VKey<E>, WKey<E>) {
        (self.get_vkey(), self.get_wkey())
    }

    pub fn get_vkey(&self) -> VKey<E> {
        let v1 = self
            .h_alpha_powers
            .iter()
            .skip(1) // skip the h
            .cloned()
            .collect::<Vec<_>>();
        let v2 = self
            .h_beta_powers
            .par_iter()
            .skip(1) // skip the h
            .cloned()
            .collect::<Vec<_>>();
        assert!(v1.len() == self.n);
        assert!(v2.len() == self.n);
        VKey::<E> { a: v1, b: v2 }
    }

    pub fn get_wkey(&self) -> WKey<E> {
        // +1 because we skip first g then g^1...g^n
        let w1 = self
            .g_alpha_powers
            .par_iter()
            .skip(1)
            .cloned()
            .collect::<Vec<_>>();
        let w2 = self
            .g_beta_powers
            .par_iter()
            .skip(1)
            .cloned()
            .collect::<Vec<_>>();
        assert!(w1.len() == self.n);
        assert!(w2.len() == self.n);
        WKey::<E> { a: w1, b: w2 }
    }
}

pub fn setup_fake_srs<E: Engine, R: rand::RngCore>(
    rng: &mut R,
    size: usize,
) -> (SRS<E>, VerifierSRS<E>) {
    let alpha = E::Fr::random(rng);
    let beta = E::Fr::random(rng);
    let g = E::G1::one();
    let h = E::G2::one();
    let mut g_alpha = E::G1::one();
    g_alpha.mul_assign(alpha.into_repr());
    let mut h_alpha = E::G2::one();
    h_alpha.mul_assign(alpha.into_repr());
    let mut g_beta = E::G1::one();
    g_beta.mul_assign(beta.into_repr());
    let mut h_beta = E::G2::one();
    h_beta.mul_assign(beta.into_repr());

    let pow = |s: &E::Fr| -> E::Fr {
        let mut t = s.clone();
        for i in 0..size {
            t.mul_assign(&s)
        }
        t
    };
    // alpha^n
    // TODO replace via a call to pow that works..
    //let alpha_n = alpha.pow(size as u64);
    let alpha_n = pow(&alpha);
    // beta^n
    let beta_n = pow(&beta);
    //let beta_n = beta.pow(size as u64);
    // g^alpha^n
    let mut g_alpha_n = E::G1::one();
    g_alpha_n.mul_assign(alpha_n.into_repr());
    // g^beta^n
    let mut g_beta_n = E::G1::one();
    g_beta_n.mul_assign(beta_n.into_repr());

    let mut g_alpha_powers = Vec::new();
    let mut g_beta_powers = Vec::new();
    let mut h_alpha_powers = Vec::new();
    let mut h_beta_powers = Vec::new();
    rayon::scope(|s| {
        let g = &g;
        let alpha = &alpha;
        let h = &h;
        let beta = &beta;
        let g_alpha_n = &g_alpha_n;
        let g_beta_n = &g_beta_n;
        let g_alpha_powers = &mut g_alpha_powers;
        s.spawn(move |_| {
            // +1 because we go to power 2n included and we start at power g^a^n
            *g_alpha_powers = structured_generators_scalar_power(size + 1, g_alpha_n, alpha);
        });
        let g_beta_powers = &mut g_beta_powers;
        s.spawn(move |_| {
            *g_beta_powers = structured_generators_scalar_power(size + 1, g_beta_n, beta);
        });

        let h_alpha_powers = &mut h_alpha_powers;
        s.spawn(move |_| {
            *h_alpha_powers = structured_generators_scalar_power(size + 1, h, alpha);
        });

        let h_beta_powers = &mut h_beta_powers;
        s.spawn(move |_| {
            *h_beta_powers = structured_generators_scalar_power(size + 1, h, beta);
        });
    });

    assert!(h_alpha_powers[0] == E::G2::one().into_affine());
    assert!(h_beta_powers[0] == E::G2::one().into_affine());
    assert!(g_alpha_powers[0] == g_alpha_n.into_affine());
    assert!(g_beta_powers[0] == g_beta_n.into_affine());
    // g^alpha^{n+1}
    let mut g_alpha_n1 = g_alpha_powers[1].clone();
    // g^beta^{n+1}
    let mut g_beta_n1 = g_beta_powers[1].clone();
    let vk = VerifierSRS {
        g,
        h,
        g_alpha,
        g_beta,
        h_alpha,
        h_beta,
        g_alpha_n1: g_alpha_n1.into_projective(),
        g_beta_n1: g_beta_n1.into_projective(),
    };

    (
        SRS {
            n: size,
            g_alpha_powers,
            h_beta_powers,
            g_beta_powers,
            h_alpha_powers,
        },
        vk,
    )
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
