use std::marker::PhantomData;

use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use itertools::Itertools;

use super::{msm, poly::DensePolynomial, Proof, VerifyingKey};
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

    let window_size = msm::fixed_base::get_mul_window_size(num);
    let scalar_bits = G::Scalar::NUM_BITS as usize;
    let g_table = msm::fixed_base::get_window_table(scalar_bits, window_size, g.clone());
    let powers_of_g = msm::fixed_base::multi_scalar_mul::<G>(
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
    r_commitment_steps: Vec<((E::Fqk, E::Fqk, Vec<E::Fqk>), (E::Fqk, E::Fqk, Vec<E::Fqk>))>, // Output
    r_base: (E::G1, E::G2), // Message
    _marker: PhantomData<D>,
}
pub struct GIPAAux<E: Engine, D: Digest> {
    r_transcript: Vec<E::Fr>,
    ck_base: (E::G2, E::G1),
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
        hash_input.extend_from_slice(&com_a.as_bytes());
        hash_input.extend_from_slice(&com_b.as_bytes());
        hash_input.extend_from_slice(&com_c.as_bytes());
        if let Some(r) = E::Fr::from_bytes(&D::digest(&hash_input)) {
            break r;
        };

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
        hash_input.extend_from_slice(&proof.com_a.as_bytes());
        hash_input.extend_from_slice(&proof.com_b.as_bytes());
        hash_input.extend_from_slice(&proof.com_c.as_bytes());
        if let Some(r) = E::Fr::from_bytes(&D::digest(&hash_input)) {
            break r;
        };
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
    r_one.negate();
    r_sum.mul_assign(&r_one); // TODO: check that div r_one is this

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
    assert_eq!(left.len(), right.len());
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
    assert_eq!(left.len(), right.len());
    msm::variable_base::multi_scalar_mul(
        &left.iter().map(|b| b.into_affine()).collect::<Vec<_>>(),
        &right.iter().map(|b| b.into_repr()).collect::<Vec<_>>(),
    )
}

fn scalar_inner_product<F: Field>(left: &[F], right: &[F]) -> F {
    assert_eq!(left.len(), right.len());
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
    // Run GIPA
    let (proof, aux) = GIPAProof::<E, D>::prove_with_aux(values, (ck.0, ck.1));

    // Prove final commitment keys are wellformed
    let (ck_a_final, ck_b_final) = aux.ck_base;
    let transcript = aux.r_transcript;
    let transcript_inverse = transcript.iter().map(|x| x.inverse().unwrap()).collect();
    let r_inverse = r_shift.inverse().unwrap();

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        hash_input.extend_from_slice(&transcript.first().unwrap().as_bytes());
        hash_input.extend_from_slice(ck_a_final.into_affine().into_uncompressed().as_ref());
        hash_input.extend_from_slice(ck_b_final.into_affine().into_uncompressed().as_ref());
        if let Some(c) = E::Fr::from_bytes(&D::digest(&hash_input)) {
            break c;
        };
        counter_nonce += 1;
    };

    // Complete KZG proofs
    let ck_a_kzg_opening =
        prove_commitment_key_kzg_opening(&srs.h_beta_powers, &transcript_inverse, &r_inverse, &c);
    let ck_b_kzg_opening =
        prove_commitment_key_kzg_opening(&srs.g_alpha_powers, &transcript, &<E::Fr>::one(), &c);

    PairingInnerProductABProof {
        gipa_proof: proof,
        final_ck: (ck_a_final, ck_b_final),
        final_ck_proof: (ck_a_kzg_opening, ck_b_kzg_opening),
        _marker: PhantomData,
    }
}

macro_rules! mul {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.mul_assign($b);
        a
    }};
}

macro_rules! add {
    ($a:expr, $b:expr) => {{
        let mut a = $a;
        a.add_assign($b);
        a
    }};
}

// IP: PairingInnerProduct<E>
// LMC: AFGHOCommitmentG1<E>
// RMC: AFGHOCommitmentG2<E>
// IPC: IdentityCommitment<E::Fqk, E::Fr>
impl<E: Engine, D: Digest> GIPAProof<E, D> {
    fn prove(
        values: (&[E::G1], &[E::G2], &E::Fqk),
        ck: (&[E::G2], &[E::G1]),
        com: (&E::Fqk, &E::Fqk, &Vec<E::Fqk>),
    ) -> Self {
        assert_eq!(
            pairing_inner_product::<E>(values.0, values.1),
            values.2.clone()
        );
        assert_eq!(
            values.0.len().count_ones(),
            1,
            "message length must be a power of two"
        );
        assert!(pairing_inner_product::<E>(values.0, ck.0) == *com.0);
        assert!(pairing_inner_product::<E>(ck.1, values.1) == *com.1);
        assert!(
            // &vec![ck.2.clone()],
            &vec![values.2.clone()] == com.2
        );

        let (proof, _) = Self::prove_with_aux((values.0, values.1), (ck.0, ck.1));
        proof
    }

    pub fn prove_with_aux(
        values: (&[E::G1], &[E::G2]),
        ck: (&[E::G2], &[E::G1]),
    ) -> (Self, GIPAAux<E, D>) {
        let (m_a, m_b) = values;
        let (ck_a, ck_b) = ck;
        Self::_prove((m_a.to_vec(), m_b.to_vec()), (ck_a.to_vec(), ck_b.to_vec()))
    }

    /// Returns vector of recursive commitments and transcripts in reverse order.
    fn _prove(
        values: (Vec<E::G1>, Vec<E::G2>),
        ck: (Vec<E::G2>, Vec<E::G1>),
    ) -> (Self, GIPAAux<E, D>) {
        let (mut m_a, mut m_b) = values;
        let (mut ck_a, mut ck_b) = ck;
        let mut r_commitment_steps = Vec::new();
        let mut r_transcript = Vec::new();
        assert!(m_a.len().is_power_of_two());
        let (m_base, ck_base) = 'recurse: loop {
            if m_a.len() == 1 {
                // base case
                break 'recurse (
                    (m_a[0].clone(), m_b[0].clone()),
                    (ck_a[0].clone(), ck_b[0].clone()),
                );
            } else {
                // recursive step
                // Recurse with problem of half size
                let split = m_a.len() / 2;

                let m_a_1 = &m_a[split..];
                let m_a_2 = &m_a[..split];
                let ck_a_1 = &ck_a[..split];
                let ck_a_2 = &ck_a[split..];

                let m_b_1 = &m_b[..split];
                let m_b_2 = &m_b[split..];
                let ck_b_1 = &ck_b[split..];
                let ck_b_2 = &ck_b[..split];

                let com_1 = (
                    pairing_inner_product::<E>(m_a_1, ck_a_1),
                    pairing_inner_product::<E>(ck_b_1, m_b_1),
                    vec![pairing_inner_product::<E>(m_a_1, m_b_1)],
                );
                let com_2 = (
                    pairing_inner_product::<E>(m_a_2, ck_a_2),
                    pairing_inner_product::<E>(ck_b_2, m_b_2),
                    vec![pairing_inner_product::<E>(m_a_2, m_b_2)],
                );

                // Fiat-Shamir challenge
                let mut counter_nonce: usize = 0;
                let default_transcript = E::Fr::zero();
                let transcript = r_transcript.last().unwrap_or(&default_transcript);
                let (c, c_inv) = 'challenge: loop {
                    let mut hash_input = Vec::new();
                    hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
                    hash_input.extend_from_slice(&transcript.as_bytes());
                    hash_input.extend_from_slice(&com_1.0.as_bytes());
                    hash_input.extend_from_slice(&com_1.1.as_bytes());
                    for c in &com_1.2 {
                        hash_input.extend_from_slice(&c.as_bytes());
                    }
                    hash_input.extend_from_slice(&com_2.0.as_bytes());
                    hash_input.extend_from_slice(&com_2.1.as_bytes());
                    for c in &com_2.2 {
                        hash_input.extend_from_slice(&c.as_bytes());
                    }
                    let c = E::Fr::from_bytes(
                        &D::digest(&hash_input).as_slice()[0..E::Fr::SERIALIZED_BYTES],
                    );
                    if let Some(c) = c {
                        if let Some(c_inv) = c.inverse() {
                            // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                            // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                            break 'challenge (c_inv, c);
                        }
                    }

                    counter_nonce += 1;
                };

                // Set up values for next step of recursion
                m_a = m_a_1
                    .iter()
                    .map(|a| mul!(*a, c))
                    .zip(m_a_2)
                    .map(|(a_1, a_2)| add!(a_1, a_2))
                    .collect::<Vec<_>>();

                m_b = m_b_2
                    .iter()
                    .map(|b| mul!(*b, c_inv))
                    .zip(m_b_1)
                    .map(|(b_1, b_2)| add!(b_1, b_2))
                    .collect::<Vec<_>>();

                ck_a = ck_a_2
                    .iter()
                    .map(|a| mul!(*a, c_inv))
                    .zip(ck_a_1)
                    .map(|(a_1, a_2)| add!(a_1, a_2))
                    .collect::<Vec<_>>();

                ck_b = ck_b_1
                    .iter()
                    .map(|b| mul!(*b, c))
                    .zip(ck_b_2)
                    .map(|(b_1, b_2)| add!(b_1, b_2))
                    .collect::<Vec<_>>();

                r_commitment_steps.push((com_1, com_2));
                r_transcript.push(c);
            }
        };
        r_transcript.reverse();
        r_commitment_steps.reverse();
        (
            GIPAProof {
                r_commitment_steps,
                r_base: m_base,
                _marker: PhantomData,
            },
            GIPAAux {
                r_transcript,
                ck_base,
                _marker: PhantomData,
            },
        )
    }
}

pub fn prove_commitment_key_kzg_opening<G: CurveProjective>(
    srs_powers: &Vec<G>,
    transcript: &Vec<G::Scalar>,
    r_shift: &G::Scalar,
    kzg_challenge: &G::Scalar,
) -> G {
    let ck_polynomial =
        DensePolynomial::from_coeffs(polynomial_coefficients_from_transcript(transcript, r_shift));
    assert_eq!(srs_powers.len(), ck_polynomial.coeffs().len());

    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(&transcript, kzg_challenge, &r_shift);

    let mut neg_kzg_challenge = *kzg_challenge;
    neg_kzg_challenge.negate();

    let quotient_polynomial = &(&ck_polynomial
        - &DensePolynomial::from_coeffs(vec![ck_polynomial_c_eval]))
        / &(DensePolynomial::from_coeffs(vec![neg_kzg_challenge, G::Scalar::one()]));

    let mut quotient_polynomial_coeffs = quotient_polynomial.into_coeffs();
    quotient_polynomial_coeffs.resize(srs_powers.len(), G::Scalar::zero());

    let opening = multiexponentiation_inner_product(srs_powers, &quotient_polynomial_coeffs);
    opening
}

fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &Vec<F>,
    z: &F,
    r_shift: &F,
) -> F {
    let mut power_2_zr = *z;
    power_2_zr.mul_assign(z);
    power_2_zr.mul_assign(r_shift);
    let mut product_form = Vec::new();
    for x in transcript.iter() {
        product_form.push(add!(F::one(), &mul!(*x, &power_2_zr)));
        power_2_zr.mul_assign(&power_2_zr.clone());
    }
    product_form[1..]
        .iter()
        .fold(product_form[0], |mut acc, curr| {
            acc.mul_assign(curr);
            acc
        })
}

fn polynomial_coefficients_from_transcript<F: Field>(transcript: &Vec<F>, r_shift: &F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = r_shift.clone();
    for (i, x) in transcript.iter().enumerate() {
        for j in 0..(2_usize).pow(i as u32) {
            coefficients.push(mul!(coefficients[j], &mul!(*x, &power_2_r)));
        }
        power_2_r.mul_assign(&power_2_r.clone());
    }
    // Interleave with 0 coefficients
    coefficients
        .iter()
        .interleave(vec![F::zero()].iter().cycle().take(coefficients.len() - 1))
        .cloned()
        .collect()
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
