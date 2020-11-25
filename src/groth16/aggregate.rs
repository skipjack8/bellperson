use std::marker::PhantomData;

use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};

use super::{fixed_base_msm, Proof, VerifyingKey};
use crate::bls::{Engine, PairingCurveAffine};

#[derive(Clone, Debug)]
pub struct SRS<E: Engine> {
    pub g_alpha_powers: Vec<E::G1>,
    pub h_beta_powers: Vec<E::G2>,
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
    pub fn get_commitment_keys(&self) -> (Vec<E::G2>, Vec<E::G1>) {
        let ck_1 = self.h_beta_powers.iter().step_by(2).cloned().collect();
        let ck_2 = self.g_alpha_powers.iter().step_by(2).cloned().collect();
        (ck_1, ck_2)
    }

    pub fn get_verifier_key(&self) -> VerifierSRS<E> {
        VerifierSRS {
            g: self.g_alpha_powers[0].clone(),
            h: self.h_beta_powers[0].clone(),
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
) -> Vec<G> {
    assert!(num > 0);
    let mut powers_of_scalar = Vec::with_capacity(num);
    let mut pow_s = G::Scalar::one();
    for _ in 0..num {
        powers_of_scalar.push(pow_s);
        pow_s.mul_assign(s);
    }

    let window_size = fixed_base_msm::get_mul_window_size(num);
    let scalar_bits = G::Scalar::NUM_BITS as usize;
    let g_table = fixed_base_msm::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = fixed_base_msm::multi_scalar_mul::<G>(
        scalar_bits,
        window_size,
        &g_table,
        &powers_of_scalar[..],
    );
    powers_of_g
}

pub struct AggregateProof<E: Engine, D: Digest> {
    com_a: E::Fqk,
    com_b: E::Fqk,
    com_c: E::Fqk,
    ip_ab: E::Fqk,
    agg_c: E::G1,
    tipa_proof_ab: PairingInnerProductABProof<E, D>,
    tipa_proof_c: MultiExpInnerProductCProof<E, D>,
}

pub struct PairingInnerProductABProof<E: Engine, D: Digest> {
    gipa_proof: GIPAProof<E, D>,
    final_ck: (E::G2, E::G1), // Key
    final_ck_proof: (E::G2, E::G1),
    _marker: PhantomData<D>,
}

pub struct GIPAProof<E: Engine, D: Digest> {
    r_commitment_steps: Vec<((E::Fqk, E::Fqk, E::Fqk), (E::Fqk, E::Fqk, E::Fqk))>, // Output
    r_base: (E::G1, E::G2),                                                        // Message
    _marker: PhantomData<D>,
}

pub struct MultiExpInnerProductCProof<E: Engine, D: Digest> {
    gipa_proof: GIPAProofWithSSM<E, D>,
    final_ck: E::G1,
    final_ck_proof: E::G2,
    _marker: PhantomData<D>,
}

pub struct GIPAProofWithSSM<E: Engine, D: Digest> {
    r_commitment_steps: Vec<((E::Fqk, E::Fr, E::Fqk), (E::Fqk, E::Fr, E::Fqk))>, // Output
    r_base: (E::G1, E::Fr),                                                      // Message
    _marker: PhantomData<D>,
}

pub fn aggregate_proofs<E: Engine, D: Digest>(
    ip_srs: &SRS<E>,
    proofs: &[Proof<E>],
) -> AggregateProof<E, D> {
    let a = proofs
        .iter()
        .map(|proof| proof.a.into_projective())
        .collect::<Vec<E::G1>>();
    let b = proofs
        .iter()
        .map(|proof| proof.b.into_projective())
        .collect::<Vec<E::G2>>();
    let c = proofs
        .iter()
        .map(|proof| proof.c.into_projective())
        .collect::<Vec<E::G1>>();

    let (ck_1, ck_2) = ip_srs.get_commitment_keys();

    let com_a = pairing_inner_product::<E>(&a, &ck_1);
    let com_b = pairing_inner_product::<E>(&ck_2, &b);
    let com_c = pairing_inner_product::<E>(&c, &ck_1);

    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        // TODO: add serialize and deserialize to Fqk
        // hash_input.extend_from_slice(&to_bytes![com_a, com_b, com_c]?);
        // if let Some(r) = E::Fr::deserialize(&D::digest(&hash_input)) {
        // break r;
        // };
        // fake
        if counter_nonce == 10 {
            break E::Fr::zero();
        }
        counter_nonce += 1;
    };

    let r_vec = structured_scalar_power(proofs.len(), &r);
    let a_r = a
        .iter()
        .zip(&r_vec)
        .map(|(a, r)| {
            let mut a = *a;
            a.mul_assign(*r);
            a
        })
        .collect::<Vec<E::G1>>();
    let ip_ab = pairing_inner_product::<E>(&a_r, &b);
    let agg_c = multiexponentiation_inner_product::<E::G1>(&c, &r_vec);

    let ck_1_r = ck_1
        .iter()
        .zip(&r_vec)
        .map(|(ck, r)| {
            let mut ck = *ck;
            ck.mul_assign(r.inverse().unwrap());
            ck
        })
        .collect::<Vec<E::G2>>();

    assert_eq!(com_a, pairing_inner_product::<E>(&a_r, &ck_1_r));

    let tipa_proof_ab = prove_with_srs_shift::<E, D>(&ip_srs, (&a_r, &b), (&ck_1_r, &ck_2), &r);

    let tipa_proof_c = prove_with_structured_scalar_message::<E, D>(&ip_srs, (&c, &r_vec), &ck_1);

    AggregateProof {
        com_a,
        com_b,
        com_c,
        ip_ab,
        agg_c,
        tipa_proof_ab,
        tipa_proof_c,
    }
}

pub fn verify_aggregate_proof<E: Engine, D: Digest>(
    ip_verifier_srs: &VerifierSRS<E>,
    vk: &VerifyingKey<E>,
    public_inputs: &Vec<Vec<E::Fr>>,
    proof: &AggregateProof<E, D>,
) -> bool {
    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        // TODO:
        // hash_input.extend_from_slice(&to_bytes![proof.com_a, proof.com_b, proof.com_c]?);
        // if let Some(r) = <P::Fr>::from_random_bytes(&D::digest(&hash_input)) {
        // break r;
        // };
        if counter_nonce == 10 {
            break E::Fr::zero();
        }
        counter_nonce += 1;
    };

    // Check TIPA proofs
    let tipa_proof_ab_valid = verify_with_srs_shift::<E, D>(
        ip_verifier_srs,
        (&proof.com_a, &proof.com_b, &proof.ip_ab),
        &proof.tipa_proof_ab,
        &r,
    );
    let tipa_proof_c_valid = verify_with_structured_scalar_message::<E, D>(
        ip_verifier_srs,
        (&proof.com_c, &proof.agg_c),
        &r,
        &proof.tipa_proof_c,
    );

    // Check aggregate pairing product equation

    let mut r_sum = r.pow(&[public_inputs.len() as u64]);
    r_sum.sub_assign(&E::Fr::one());
    let mut r_one = r;
    r_one.sub_assign(&E::Fr::one());
    r_sum.mul_assign(&r_one.inverse().unwrap()); // TODO: check that div r_one is this

    let p1 = {
        let mut alpha_g1_r_sum = vk.alpha_g1.into_projective();
        alpha_g1_r_sum.mul_assign(r_sum);
        E::pairing(alpha_g1_r_sum, vk.beta_g2)
    };

    assert_eq!(vk.ic.len(), public_inputs[0].len() + 1);
    let r_vec = structured_scalar_power(public_inputs.len(), &r);
    let mut g_ic = vk.ic[0].into_projective();
    g_ic.mul_assign(r_sum);
    for (i, b) in vk.ic.iter().skip(1).enumerate() {
        let mut x = b.into_projective();
        x.mul_assign(scalar_inner_product(
            &public_inputs
                .iter()
                .map(|inputs| inputs[i].clone())
                .collect::<Vec<E::Fr>>(),
            &r_vec,
        ));
        g_ic.add_assign(&x);
    }
    let p2 = E::pairing(g_ic, vk.gamma_g2);
    let p3 = E::pairing(proof.agg_c, vk.delta_g2);

    let mut p1_p2_p3 = p1;
    p1_p2_p3.mul_assign(&p2);
    p1_p2_p3.mul_assign(&p3);
    let ppe_valid = proof.ip_ab == p1_p2_p3;

    tipa_proof_ab_valid && tipa_proof_c_valid && ppe_valid
}

fn structured_scalar_power<F: Field>(num: usize, s: &F) -> Vec<F> {
    let mut powers = vec![F::one()];
    for i in 1..num {
        let mut x = powers[i - 1];
        x.mul_assign(s);
        powers.push(x);
    }
    powers
}

fn pairing_inner_product<E: Engine>(left: &[E::G1], right: &[E::G2]) -> E::Fqk {
    let pairs = left
        .iter()
        .map(|e| e.into_affine())
        .zip(right.iter().map(|e| e.into_affine()))
        .map(|(a, b)| (a.prepare(), b.prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    let ml: E::Fqk = E::miller_loop(pairs_ref.iter());
    E::final_exponentiation(&ml).expect("invalid pairing")
}

fn multiexponentiation_inner_product<G: CurveProjective>(left: &[G], right: &[G::Scalar]) -> G {
    todo!()
}

fn scalar_inner_product<F: Field>(left: &[F], right: &[F]) -> F {
    left.iter()
        .zip(right)
        .map(|(x, y)| {
            let mut x = *x;
            x.mul_assign(y);
            y
        })
        .fold(F::zero(), |mut acc, curr| {
            acc.add_assign(curr);
            acc
        })
}

// Shifts KZG proof for left message by scalar r (used for efficient composition with aggregation protocols)
// LMC commitment key should already be shifted before being passed as input
fn prove_with_srs_shift<E: Engine, D: Digest>(
    srs: &SRS<E>,
    values: (&[E::G1], &[E::G2]),
    ck: (&[E::G2], &[E::G1]),
    r_shift: &E::Fr,
) -> PairingInnerProductABProof<E, D> {
    todo!()
}

fn prove_with_structured_scalar_message<E: Engine, D: Digest>(
    srs: &SRS<E>,
    values: (&[E::G1], &[E::Fr]),
    ck: &[E::G2],
) -> MultiExpInnerProductCProof<E, D> {
    todo!()
}

fn verify_with_srs_shift<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::Fqk, &E::Fqk),
    proof: &PairingInnerProductABProof<E, D>,
    r_shift: &E::Fr,
) -> bool {
    todo!()
}

fn verify_with_structured_scalar_message<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::G1),
    scalar_b: &E::Fr,
    proof: &MultiExpInnerProductCProof<E, D>,
) -> bool {
    todo!()
}
