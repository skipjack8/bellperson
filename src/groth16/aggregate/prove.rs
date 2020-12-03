use std::marker::PhantomData;

use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use itertools::Itertools;
use log::*;
use rayon::prelude::*;
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
    pub r_commitment_steps: Vec<((E::Fqk, E::Fqk, E::Fqk), (E::Fqk, E::Fqk, E::Fqk))>, // Output
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
    pub r_commitment_steps: Vec<((E::Fqk, E::G1), (E::Fqk, E::G1))>, // Output
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

pub fn aggregate_proofs<E: Engine + std::fmt::Debug, D: Digest + Sync + Send>(
    ip_srs: &SRS<E>,
    proofs: &[Proof<E>],
) -> AggregateProof<E, D> {
    info!("aggregate_proofs");
    let (ck_1, ck_2) = ip_srs.get_commitment_keys();

    let mut a = Vec::new();
    let mut com_a = E::Fqk::zero();
    let mut b = Vec::new();
    let mut com_b = E::Fqk::zero();
    let mut c = Vec::new();
    let mut com_c = E::Fqk::zero();
    rayon::scope(|s| {
        let ck_1 = &ck_1;
        let ck_2 = &ck_2;

        let a = &mut a;
        let com_a = &mut com_a;
        s.spawn(move |_| {
            *a = proofs.par_iter().map(|proof| proof.a).collect::<Vec<_>>();
            *com_a = inner_product::pairing_affine::<E>(&a, ck_1);
        });

        let b = &mut b;
        let com_b = &mut com_b;
        s.spawn(move |_| {
            *b = proofs.par_iter().map(|proof| proof.b).collect::<Vec<_>>();
            *com_b = inner_product::pairing_affine::<E>(ck_2, &b);
        });

        let c = &mut c;
        let com_c = &mut com_c;
        s.spawn(move |_| {
            *c = proofs.par_iter().map(|proof| proof.c).collect::<Vec<_>>();
            *com_c = inner_product::pairing_affine::<E>(&c, ck_1);
        });
    });

    info!("compute r");
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
    info!("compute r_vec");
    let r_vec = structured_scalar_power(proofs.len(), &r);

    let mut a_r = Vec::new();
    let mut ck_1_r = Vec::new();

    rayon::scope(|s| {
        let r_vec = &r_vec;
        let ck_1 = &ck_1;
        let a = &a;

        let a_r = &mut a_r;
        s.spawn(move |_| {
            *a_r = a
                .par_iter()
                .zip(r_vec.par_iter())
                .map(|(a, r)| mul!(a.into_projective(), *r).into_affine())
                .collect::<Vec<E::G1Affine>>();
        });

        let ck_1_r = &mut ck_1_r;
        s.spawn(move |_| {
            *ck_1_r = ck_1
                .par_iter()
                .zip(r_vec.par_iter())
                .map(|(ck, r)| mul!(ck.into_projective(), r.inverse().unwrap()).into_affine())
                .collect::<Vec<E::G2Affine>>();
        });
    });

    let mut computed_com_a = E::Fqk::zero();
    let mut ip_ab = E::Fqk::zero();
    let mut agg_c = E::G1::zero();
    let mut tipa_proof_ab = None;
    let mut tipa_proof_c = None;

    info!("scope:start");
    rayon::scope(|s| {
        let a_r = &a_r;
        let ck_1_r = &ck_1_r;
        let b = &b;
        let c = &c;
        let r_vec = &r_vec;

        let computed_com_a = &mut computed_com_a;
        s.spawn(move |_| {
            *computed_com_a = inner_product::pairing_affine::<E>(a_r, ck_1_r);
        });

        let tipa_proof_ab = &mut tipa_proof_ab;
        s.spawn(move |_| {
            *tipa_proof_ab = Some(prove_with_srs_shift::<E, D>(
                &ip_srs,
                (a_r, b),
                (ck_1_r, &ck_2),
                &r,
            ));
        });

        let tipa_proof_c = &mut tipa_proof_c;
        s.spawn(move |_| {
            *tipa_proof_c = Some(prove_with_structured_scalar_message::<E, D>(
                &ip_srs,
                (c, r_vec),
                &ck_1,
            ));
        });

        let ip_ab = &mut ip_ab;
        s.spawn(move |_| {
            *ip_ab = inner_product::pairing_affine::<E>(&a_r, b);
        });

        let agg_c = &mut agg_c;
        s.spawn(move |_| {
            *agg_c = inner_product::multiexponentiation_affine::<E::G1Affine>(&c, r_vec);
        });
    });
    info!("scope:end");

    assert_eq!(com_a, computed_com_a);

    AggregateProof {
        com_a,
        com_b,
        com_c,
        ip_ab,
        agg_c,
        tipa_proof_ab: tipa_proof_ab.unwrap(),
        tipa_proof_c: tipa_proof_c.unwrap(),
    }
}

// Shifts KZG proof for left message by scalar r (used for efficient composition with aggregation protocols)
// LMC commitment key should already be shifted before being passed as input
fn prove_with_srs_shift<E: Engine, D: Digest>(
    srs: &SRS<E>,
    values: (&[E::G1Affine], &[E::G2Affine]),
    ck: (&[E::G2Affine], &[E::G1Affine]),
    r_shift: &E::Fr,
) -> PairingInnerProductABProof<E, D> {
    info!("prove_with_srs_shift:start");
    // Run GIPA
    let (proof, aux) = GIPAProof::<E, D>::prove_with_aux(values, (ck.0, ck.1));

    // Prove final commitment keys are wellformed
    let (ck_a_final, ck_b_final) = aux.ck_base;
    let transcript = aux.r_transcript;
    let transcript_inverse = transcript
        .par_iter()
        .map(|x| x.inverse().unwrap())
        .collect::<Vec<_>>();
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
    let mut ck_a_kzg_opening = E::G2::zero();
    let mut ck_b_kzg_opening = E::G1::zero();

    rayon::scope(|s| {
        let ck_a_kzg_opening = &mut ck_a_kzg_opening;
        s.spawn(move |_| {
            *ck_a_kzg_opening = prove_commitment_key_kzg_opening(
                &srs.h_beta_powers,
                &transcript_inverse,
                &r_inverse,
                &c,
            );
        });

        let ck_b_kzg_opening = &mut ck_b_kzg_opening;
        s.spawn(move |_| {
            *ck_b_kzg_opening = prove_commitment_key_kzg_opening(
                &srs.g_alpha_powers,
                &transcript,
                &<E::Fr>::one(),
                &c,
            );
        });
    });
    info!("prove_with_srs_shift:end");

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
    /// Returns vector of recursive commitments and transcripts in reverse order.
    pub fn prove_with_aux(
        values: (&[E::G1Affine], &[E::G2Affine]),
        ck: (&[E::G2Affine], &[E::G1Affine]),
    ) -> (Self, GIPAAux<E, D>) {
        info!("prove_with_aux:start");
        let (mut m_a, mut m_b) = (values.0.to_vec(), values.1.to_vec());
        let (mut ck_a, mut ck_b) = (ck.0.to_vec(), ck.1.to_vec());
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
                    inner_product::pairing_affine::<E>(m_a_1, ck_a_1), // LMC
                    inner_product::pairing_affine::<E>(ck_b_1, m_b_1), // RMC
                    inner_product::pairing_affine::<E>(m_a_1, m_b_1),  // IPC
                );
                let com_2 = (
                    inner_product::pairing_affine::<E>(m_a_2, ck_a_2), // LLMC
                    inner_product::pairing_affine::<E>(ck_b_2, m_b_2), // RMC
                    inner_product::pairing_affine::<E>(m_a_2, m_b_2),  // IPC
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
                    .par_iter()
                    .map(|a| mul!(a.into_projective(), c))
                    .zip(m_a_2.par_iter())
                    .map(|(a_1, a_2)| add!(a_1, &a_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                m_b = m_b_2
                    .par_iter()
                    .map(|b| mul!(b.into_projective(), c_inv))
                    .zip(m_b_1.par_iter())
                    .map(|(b_1, b_2)| add!(b_1, &b_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                ck_a = ck_a_2
                    .par_iter()
                    .map(|a| mul!(a.into_projective(), c_inv))
                    .zip(ck_a_1.par_iter())
                    .map(|(a_1, a_2)| add!(a_1, &a_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                ck_b = ck_b_1
                    .par_iter()
                    .map(|b| mul!(b.into_projective(), c))
                    .zip(ck_b_2.par_iter())
                    .map(|(b_1, b_2)| add!(b_1, &b_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                r_commitment_steps.push((com_1, com_2));
                r_transcript.push(c);
            }
        };
        r_transcript.reverse();
        r_commitment_steps.reverse();
        info!("prove_with_aux:end");

        (
            GIPAProof {
                r_commitment_steps,
                r_base: (m_base.0.into_projective(), m_base.1.into_projective()),
                _marker: PhantomData,
            },
            GIPAAux {
                r_transcript,
                ck_base: (ck_base.0.into_projective(), ck_base.1.into_projective()),
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
    /// Returns vector of recursive commitments and transcripts in reverse order.
    pub fn prove_with_aux(
        values: (&[E::G1Affine], &[E::Fr]),
        ck: &[E::G2Affine],
    ) -> (Self, GIPAAuxWithSSM<E, D>) {
        info!("prove_with_aux (ssm):start");
        let (mut m_a, mut m_b) = (values.0.to_vec(), values.1.to_vec());
        let mut ck_a = ck.to_vec();

        let mut r_commitment_steps = Vec::new();
        let mut r_transcript = Vec::new();
        assert!(m_a.len().is_power_of_two());
        let (m_base, ck_base) = 'recurse: loop {
            if m_a.len() == 1 {
                // base case
                break 'recurse ((m_a[0], m_b[0]), ck_a[0]);
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
                    inner_product::pairing_affine::<E>(m_a_1, ck_a_1), // LMC::commit
                    inner_product::multiexponentiation_affine::<E::G1Affine>(m_a_1, m_b_1), // IPC::commit
                );
                let com_2 = (
                    inner_product::pairing_affine::<E>(m_a_2, ck_a_2),
                    inner_product::multiexponentiation_affine::<E::G1Affine>(m_a_2, m_b_2),
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

                    bincode::serialize_into(&mut hash_input, &com_2.0).expect("vec");
                    bincode::serialize_into(&mut hash_input, &com_2.1).expect("vec");

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
                    .map(|a| mul!(a.into_projective(), c))
                    .zip(m_a_2)
                    .map(|(a_1, a_2)| add!(a_1, &a_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                m_b = m_b_2
                    .iter()
                    .map(|b| mul!(*b, &c_inv))
                    .zip(m_b_1)
                    .map(|(b_1, b_2)| add!(b_1, b_2))
                    .collect::<Vec<_>>();

                ck_a = ck_a_2
                    .iter()
                    .map(|a| mul!(a.into_projective(), c_inv))
                    .zip(ck_a_1)
                    .map(|(a_1, a_2)| add!(a_1, &a_2.into_projective()).into_affine())
                    .collect::<Vec<_>>();

                r_commitment_steps.push((com_1, com_2));
                r_transcript.push(c);
            }
        };
        r_transcript.reverse();
        r_commitment_steps.reverse();

        info!("prove_with_aux (ssm):end");
        (
            GIPAProofWithSSM {
                r_commitment_steps,
                r_base: (m_base.0.into_projective(), m_base.1),
                _marker: PhantomData,
            },
            GIPAAuxWithSSM {
                r_transcript,
                ck_base: ck_base.into_projective(),
                _marker: PhantomData,
            },
        )
    }
}
pub fn prove_commitment_key_kzg_opening<G: CurveProjective>(
    srs_powers: &[G::Affine],
    transcript: &[G::Scalar],
    r_shift: &G::Scalar,
    kzg_challenge: &G::Scalar,
) -> G {
    info!("prove_commitment_key_kzg_opening:start");
    let ck_polynomial =
        DensePolynomial::from_coeffs(polynomial_coefficients_from_transcript(transcript, r_shift));
    assert_eq!(srs_powers.len(), ck_polynomial.coeffs().len());

    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(&transcript, kzg_challenge, &r_shift);

    let mut neg_kzg_challenge = *kzg_challenge;
    neg_kzg_challenge.negate();

    info!("kzg:poly");
    let quotient_polynomial = &(&ck_polynomial
        - &DensePolynomial::from_coeffs(vec![ck_polynomial_c_eval]))
        / &(DensePolynomial::from_coeffs(vec![neg_kzg_challenge, G::Scalar::one()]));

    let mut quotient_polynomial_coeffs = quotient_polynomial.into_coeffs();
    quotient_polynomial_coeffs.resize(srs_powers.len(), G::Scalar::zero());

    info!("kzg:multiexp");
    let opening =
        inner_product::multiexponentiation_affine(srs_powers, &quotient_polynomial_coeffs);
    info!("prove_commitment_key_kzg_opening:end");
    opening
}

pub(super) fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: &F,
    r_shift: &F,
) -> F {
    info!("poly_eval_from_transcript:start");
    let mut power_2_zr = *z;
    power_2_zr.mul_assign(z);
    power_2_zr.mul_assign(r_shift);
    let mut product_form: Vec<F> = Vec::with_capacity(transcript.len());
    for x in transcript.iter() {
        product_form.push(add!(F::one(), &mul!(*x, &power_2_zr)));
        power_2_zr.mul_assign(&power_2_zr.clone());
    }
    let res = product_form[1..]
        .iter()
        .fold(product_form[0], |mut acc, curr| {
            acc.mul_assign(curr);
            acc
        });
    info!("poly_eval_from_transcript:end");
    res
}

fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: &F) -> Vec<F> {
    info!("poly_coeffs:start");
    let mut coefficients = vec![F::one()];
    let mut power_2_r = r_shift.clone();
    for (i, x) in transcript.iter().enumerate() {
        for j in 0..(2_usize).pow(i as u32) {
            coefficients.push(mul!(coefficients[j], &mul!(*x, &power_2_r)));
        }
        power_2_r.mul_assign(&power_2_r.clone());
    }
    // Interleave with 0 coefficients
    let res = coefficients
        .iter()
        .interleave(vec![F::zero()].iter().cycle().take(coefficients.len() - 1))
        .cloned()
        .collect();
    info!("poly_coeffs:end");
    res
}

fn prove_with_structured_scalar_message<E: Engine, D: Digest>(
    srs: &SRS<E>,
    values: (&[E::G1Affine], &[E::Fr]),
    ck: &[E::G2Affine],
) -> MultiExpInnerProductCProof<E, D> {
    info!("prove_ssm:start");
    // Run GIPA
    let (proof, aux) = GIPAProofWithSSM::<E, D>::prove_with_aux(values, ck); // TODO: add plaeholder value

    // Prove final commitment key is wellformed
    let ck_a_final = aux.ck_base;
    let transcript = aux.r_transcript;
    let transcript_inverse = transcript
        .iter()
        .map(|x| x.inverse().unwrap())
        .collect::<Vec<_>>();

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

    info!("prove_ssm:end");
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
