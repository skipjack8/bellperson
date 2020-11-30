use std::marker::PhantomData;

use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::{inner_product, poly::DensePolynomial, structured_scalar_power, SRS};
use crate::bls::Engine;
use crate::groth16::Proof;

#[derive(Serialize, Deserialize)]
pub struct AggregateProof<E: Engine, D: Digest> {
    pub com_a: E::Fqk,
    pub com_b: E::Fqk,
    pub com_c: E::Fqk,
    pub ip_ab: E::Fqk,
    pub agg_c: E::G1,
    #[serde(bound(
        serialize = "PairingInnerProductABProof<E, D>: Serialize",
        deserialize = "PairingInnerProductABProof<E, D>: Deserialize<'de>",
    ))]
    pub tipa_proof_ab: PairingInnerProductABProof<E, D>,
    #[serde(bound(
        serialize = "MultiExpInnerProductCProof<E, D>: Serialize",
        deserialize = "MultiExpInnerProductCProof<E, D>: Deserialize<'de>",
    ))]
    pub tipa_proof_c: MultiExpInnerProductCProof<E, D>,
}

#[derive(Serialize, Deserialize)]
pub struct PairingInnerProductABProof<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "GIPAProof<E, D>: Serialize",
        deserialize = "GIPAProof<E, D>: Deserialize<'de>",
    ))]
    pub gipa_proof: GIPAProof<E, D>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub final_ck: (E::G2, E::G1), // Key
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub final_ck_proof: (E::G2, E::G1),
    pub _marker: PhantomData<D>,
}

#[derive(Serialize, Deserialize)]
pub struct GIPAProof<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fr: Serialize,E::G1: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fr: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub r_commitment_steps: Vec<((E::Fqk, E::Fqk, Vec<E::Fqk>), (E::Fqk, E::Fqk, Vec<E::Fqk>))>, // Output
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub r_base: (E::G1, E::G2), // Message
    pub _marker: PhantomData<D>,
}

#[derive(Serialize, Deserialize)]
pub struct GIPAAux<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "E::Fr: Serialize",
        deserialize = "E::Fr: Deserialize<'de>",
    ))]
    pub r_transcript: Vec<E::Fr>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub ck_base: (E::G2, E::G1),
    pub _marker: PhantomData<D>,
}

#[derive(Serialize, Deserialize)]
pub struct MultiExpInnerProductCProof<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "GIPAProofWithSSM<E, D>: Serialize",
        deserialize = "GIPAProofWithSSM<E, D>: Deserialize<'de>",
    ))]
    pub gipa_proof: GIPAProofWithSSM<E, D>,
    pub final_ck: E::G2,
    pub final_ck_proof: E::G2,
    pub _marker: PhantomData<D>,
}

#[derive(Serialize, Deserialize)]
pub struct GIPAProofWithSSM<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fr: Serialize,E::G1: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fr: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub r_commitment_steps: Vec<((E::Fqk, E::Fr, Vec<E::G1>), (E::Fqk, E::Fr, Vec<E::G1>))>, // Output
    pub r_base: (E::G1, E::Fr), // Message
    pub _marker: PhantomData<D>,
}

#[derive(Serialize, Deserialize)]
pub struct GIPAAuxWithSSM<E: Engine, D: Digest> {
    #[serde(bound(
        serialize = "E::Fr: Serialize",
        deserialize = "E::Fr: Deserialize<'de>",
    ))]
    pub r_transcript: Vec<E::Fr>,
    #[serde(bound(
        serialize = "E::G2: Serialize",
        deserialize = "E::G2: Deserialize<'de>",
    ))]
    pub ck_base: E::G2,
    pub _marker: PhantomData<D>,
}

pub fn aggregate_proofs<E: Engine + std::fmt::Debug, D: Digest>(
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

    let com_a = inner_product::pairing::<E>(&a, &ck_1);
    let com_b = inner_product::pairing::<E>(&ck_2, &b);
    let com_c = inner_product::pairing::<E>(&c, &ck_1);

    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &com_a).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_b).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_c).expect("vec");

        if let Some(r) = E::Fr::from_random_bytes(&D::digest(&hash_input).as_slice()[..]) {
            break r;
        };

        counter_nonce += 1;
    };
    let r_vec = structured_scalar_power(proofs.len(), &r);
    let a_r = a
        .iter()
        .zip(&r_vec)
        .map(|(a, r)| mul!(*a, *r))
        .collect::<Vec<E::G1>>();
    let ip_ab = inner_product::pairing::<E>(&a_r, &b);
    let agg_c = inner_product::multiexponentiation::<E::G1>(&c, &r_vec);

    let ck_1_r = ck_1
        .iter()
        .zip(&r_vec)
        .map(|(ck, r)| mul!(*ck, r.inverse().unwrap()))
        .collect::<Vec<E::G2>>();

    assert_eq!(com_a, inner_product::pairing::<E>(&a_r, &ck_1_r));
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
        bincode::serialize_into(&mut hash_input, &transcript.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &ck_a_final).expect("vec");
        bincode::serialize_into(&mut hash_input, &ck_b_final).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &D::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
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

// IP: PairingInnerProduct<E>
// LMC: AFGHOCommitmentG1<E>
// RMC: AFGHOCommitmentG2<E>
// IPC: IdentityCommitment<E::Fqk, E::Fr>
impl<E: Engine, D: Digest> GIPAProof<E, D> {
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
                    inner_product::pairing::<E>(m_a_1, ck_a_1),      // LMC
                    inner_product::pairing::<E>(ck_b_1, m_b_1),      // RMC
                    vec![inner_product::pairing::<E>(m_a_1, m_b_1)], // IPC
                );
                let com_2 = (
                    inner_product::pairing::<E>(m_a_2, ck_a_2),      // LLMC
                    inner_product::pairing::<E>(ck_b_2, m_b_2),      // RMC
                    vec![inner_product::pairing::<E>(m_a_2, m_b_2)], // IPC
                );

                // Fiat-Shamir challenge
                let mut counter_nonce: usize = 0;
                let default_transcript = E::Fr::zero();
                let transcript = r_transcript.last().unwrap_or(&default_transcript);
                let (c, c_inv) = 'challenge: loop {
                    let mut hash_input = Vec::new();
                    hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
                    bincode::serialize_into(&mut hash_input, &transcript).expect("vec");

                    bincode::serialize_into(&mut hash_input, &com_1.0).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_1.1).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_1.2).expect("vec");

                    bincode::serialize_into(&mut hash_input, &com_2.0).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_2.1).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_2.2).expect("vec");

                    let d = D::digest(&hash_input);
                    let c = fr_from_u128::<E::Fr>(d.as_slice());
                    if let Some(c_inv) = c.inverse() {
                        // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                        // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                        break 'challenge (c_inv, c);
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

// IPAWithSSMProof<
//    MultiexponentiationInnerProduct<<P as PairingEngine>::G1Projective>,
//    AFGHOCommitmentG1<P>,
//    IdentityCommitment<<P as PairingEngine>::G1Projective, <P as PairingEngine>::Fr>,
//    P,
//    D,
// >;
//
// GIPAProof<
//   IP = MultiexponentiationInnerProduct<<P as PairingEngine>::G1Projective>,
//   LMC = AFGHOCommitmentG1<P>,
//   RMC = SSMPlaceholderCommitment<LMC::Scalar>
//   IPC = IdentityCommitment<<P as PairingEngine>::G1Projective, <P as PairingEngine>::Fr>,,
// D>,
//
// IP: MultiexponentiationInnerProduct<<P as PairingEngine>::G1Projective>,
// LMC: AFGHOCommitmentG1<P>,
// RMC: SSMPlaceholderCommitment<LMC::Scalar>,
// IPC: IdentityCommitment<<P as PairingEngine>::G1Projective, <P as PairingEngine>::Fr>,
impl<E: Engine, D: Digest> GIPAProofWithSSM<E, D> {
    pub fn prove_with_aux(
        values: (&[E::G1], &[E::Fr]),
        ck: &[E::G2],
    ) -> (Self, GIPAAuxWithSSM<E, D>) {
        let (m_a, m_b) = values;
        Self::_prove((m_a.to_vec(), m_b.to_vec()), ck.to_vec())
    }

    /// Returns vector of recursive commitments and transcripts in reverse order.
    fn _prove(values: (Vec<E::G1>, Vec<E::Fr>), ck: Vec<E::G2>) -> (Self, GIPAAuxWithSSM<E, D>) {
        let (mut m_a, mut m_b) = values;
        let mut ck_a = ck;
        let mut r_commitment_steps = Vec::new();
        let mut r_transcript = Vec::new();
        assert!(m_a.len().is_power_of_two());
        let (m_base, ck_base) = 'recurse: loop {
            if m_a.len() == 1 {
                // base case
                break 'recurse ((m_a[0].clone(), m_b[0].clone()), ck_a[0].clone());
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

                let com_1 = (
                    inner_product::pairing::<E>(m_a_1, ck_a_1), // LMC::commit
                    E::Fr::zero(),                              // RMC::commit
                    vec![inner_product::multiexponentiation::<E::G1>(m_a_1, m_b_1)], // IPC::commit
                );
                let com_2 = (
                    inner_product::pairing::<E>(m_a_2, ck_a_2),
                    E::Fr::zero(),
                    vec![inner_product::multiexponentiation::<E::G1>(m_a_2, m_b_2)],
                );
                // Fiat-Shamir challenge
                let mut counter_nonce: usize = 0;
                let default_transcript = E::Fr::zero();
                let transcript = r_transcript.last().unwrap_or(&default_transcript);

                let (c, c_inv) = 'challenge: loop {
                    let mut hash_input = Vec::new();
                    hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
                    bincode::serialize_into(&mut hash_input, &transcript).expect("vec");

                    bincode::serialize_into(&mut hash_input, &com_1.0).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_1.1).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_1.2).expect("vec");

                    bincode::serialize_into(&mut hash_input, &com_2.0).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_2.1).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_2.2).expect("vec");

                    let d = D::digest(&hash_input);
                    let c = fr_from_u128::<E::Fr>(d.as_slice());
                    if let Some(c_inv) = c.inverse() {
                        // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                        // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                        break 'challenge (c_inv, c);
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
                    .map(|b| mul!(*b, &c_inv))
                    .zip(m_b_1)
                    .map(|(b_1, b_2)| add!(b_1, b_2))
                    .collect::<Vec<_>>();

                ck_a = ck_a_2
                    .iter()
                    .map(|a| mul!(*a, c_inv))
                    .zip(ck_a_1)
                    .map(|(a_1, a_2)| add!(a_1, a_2))
                    .collect::<Vec<_>>();

                r_commitment_steps.push((com_1, com_2));
                r_transcript.push(c);
            }
        };
        r_transcript.reverse();
        r_commitment_steps.reverse();

        (
            GIPAProofWithSSM {
                r_commitment_steps,
                r_base: m_base,
                _marker: PhantomData,
            },
            GIPAAuxWithSSM {
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

    let opening = inner_product::multiexponentiation(srs_powers, &quotient_polynomial_coeffs);
    opening
}

pub(super) fn polynomial_evaluation_product_form_from_transcript<F: Field>(
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
    // Run GIPA
    let (proof, aux) = GIPAProofWithSSM::<E, D>::prove_with_aux(values, ck); // TODO: add plaeholder value

    // Prove final commitment key is wellformed
    let ck_a_final = aux.ck_base;
    let transcript = aux.r_transcript;
    let transcript_inverse = transcript.iter().map(|x| x.inverse().unwrap()).collect();

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &transcript.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &ck_a_final).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &D::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    // Complete KZG proof
    let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
        &srs.h_beta_powers,
        &transcript_inverse,
        &E::Fr::one(),
        &c,
    );

    MultiExpInnerProductCProof {
        gipa_proof: proof,
        final_ck: ck_a_final,
        final_ck_proof: ck_a_kzg_opening,
        _marker: PhantomData,
    }
}

pub(super) fn fr_from_u128<F: PrimeField>(bytes: &[u8]) -> F {
    use std::convert::TryInto;

    let other = u128::from_be_bytes(bytes[..16].try_into().unwrap());
    let upper = (other >> 64) as u64;
    let lower = ((other << 64) >> 64) as u64;

    let mut repr = F::Repr::default();
    repr.as_mut()[0] = lower;
    repr.as_mut()[1] = upper;

    F::from_repr(repr).unwrap()
}
