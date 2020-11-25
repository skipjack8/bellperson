use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use itertools::Itertools;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    inner_product, poly::DensePolynomial, structured_scalar_power, AggregateProof, GIPAProof,
    GIPAProofWithSSM, MultiExpInnerProductCProof, PairingInnerProductABProof, SRS,
};
use crate::bls::Engine;
use crate::groth16::{multiscalar::*, Proof};
use crate::SynthesisError;

/// Aggregate `n` zkSnark proofs, where `n` must be a power of two.
pub fn aggregate_proofs<E: Engine + std::fmt::Debug>(
    ip_srs: &SRS<E>,
    proofs: &[Proof<E>],
) -> Result<AggregateProof<E>, SynthesisError> {
    let (ck_1, ck_2) = ip_srs.get_commitment_keys();

    if ck_1.len() != proofs.len() || ck_2.len() != proofs.len() {
        return Err(SynthesisError::MalformedSrs);
    }

    let ck_1 = &ck_1;
    let ck_2 = &ck_2;

    par! {
        let (a, com_a) = {
            let a = proofs.iter().map(|proof| proof.a).collect::<Vec<_>>();
            let com_a = inner_product::pairing::<E>(&a, ck_1);
            (a, com_a)
        },
        let (b, com_b) = {
            let b = proofs.iter().map(|proof| proof.b).collect::<Vec<_>>();
            let com_b = inner_product::pairing::<E>(ck_2, &b);
            (b, com_b)
        },
        let (c, com_c) = {
            let c = proofs.iter().map(|proof| proof.c).collect::<Vec<_>>();
            let com_c = inner_product::pairing::<E>(&c, ck_1);
            (c, com_c)
        }
    }

    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &com_a).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_b).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_c).expect("vec");

        if let Some(r) = E::Fr::from_random_bytes(&Sha256::digest(&hash_input).as_slice()[..]) {
            break r;
        };

        counter_nonce += 1;
    };

    let r_vec = structured_scalar_power(proofs.len(), &r);

    let r_vec = &r_vec;
    let ck_1 = &ck_1;
    let a = &a;

    par! {
        let a_r = a.par_iter()
                   .zip(r_vec.par_iter())
                   .map(|(a, r)| mul!(a.into_projective(), *r).into_affine())
                   .collect::<Vec<E::G1Affine>>(),
        let ck_1_r = ck_1.par_iter()
                         .zip(r_vec.par_iter())
                         .map(|(ck, r)| mul!(ck.into_projective(), r.inverse().unwrap()).into_affine())
                         .collect::<Vec<E::G2Affine>>()
    };

    let a_r = &a_r;
    let ck_1_r = &ck_1_r;
    let b = &b;
    let c = &c;

    par! {
        let computed_com_a = inner_product::pairing::<E>(a_r, ck_1_r),
        let tipa_proof_ab = prove_with_srs_shift::<E>(
                &ip_srs,
                (a_r, b),
                (ck_1_r, &ck_2),
                &r,
        ),
        let tipa_proof_c = prove_with_structured_scalar_message::<E>(
            &ip_srs,
            (c, r_vec),
            &ck_1,
        ),
        let ip_ab = inner_product::pairing::<E>(&a_r, b),
        let agg_c = inner_product::multiexponentiation::<E::G1Affine>(&c, r_vec)
    };

    assert_eq!(com_a, computed_com_a);

    Ok(AggregateProof {
        com_a,
        com_b,
        com_c,
        ip_ab,
        agg_c,
        tipa_proof_ab: tipa_proof_ab?,
        tipa_proof_c: tipa_proof_c?,
    })
}

// Shifts KZG proof for left message by scalar r (used for efficient composition with aggregation protocols)
// LMC commitment key should already be shifted before being passed as input
fn prove_with_srs_shift<E: Engine>(
    srs: &SRS<E>,
    values: (&[E::G1Affine], &[E::G2Affine]),
    ck: (&[E::G2Affine], &[E::G1Affine]),
    r_shift: &E::Fr,
) -> Result<PairingInnerProductABProof<E>, SynthesisError> {
    // Run GIPA
    let (proof, aux) = GIPAProof::<E>::prove_with_aux(values, (ck.0, ck.1))?;

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
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    // Complete KZG proofs

    par! {
        let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
            &srs.h_beta_powers_table,
            srs.h_beta_powers.len(),
            &transcript_inverse,
            &r_inverse,
            &c,
        ),
        let ck_b_kzg_opening = prove_commitment_key_kzg_opening(
            &srs.g_alpha_powers_table,
            srs.g_alpha_powers.len(),
            &transcript,
            &<E::Fr>::one(),
            &c,
        )
    };

    Ok(PairingInnerProductABProof {
        gipa_proof: proof,
        final_ck: (ck_a_final, ck_b_final),
        final_ck_proof: (ck_a_kzg_opening?, ck_b_kzg_opening?),
    })
}

// IP: PairingInnerProduct<E>
// LMC: AFGHOCommitmentG1<E>
// RMC: AFGHOCommitmentG2<E>
// IPC: IdentityCommitment<E::Fqk, E::Fr>
impl<E: Engine> GIPAProof<E> {
    /// Returns vector of recursive commitments and transcripts in reverse order.
    fn prove_with_aux(
        values: (&[E::G1Affine], &[E::G2Affine]),
        ck: (&[E::G2Affine], &[E::G1Affine]),
    ) -> Result<(Self, GIPAAux<E>), SynthesisError> {
        let (mut m_a, mut m_b) = (values.0.to_vec(), values.1.to_vec());
        let (mut ck_a, mut ck_b) = (ck.0.to_vec(), ck.1.to_vec());
        let mut r_commitment_steps = Vec::new();
        let mut r_transcript = Vec::new();

        if !m_a.len().is_power_of_two() {
            return Err(SynthesisError::MalformedProofs);
        }

        while m_a.len() > 1 {
            // recursive step
            // Recurse with problem of half size
            let split = m_a.len() / 2;

            let (m_a_2, m_a_1) = m_a.split_at_mut(split);
            let (ck_a_1, ck_a_2) = ck_a.split_at_mut(split);

            let (m_b_1, m_b_2) = m_b.split_at_mut(split);
            let (ck_b_2, ck_b_1) = ck_b.split_at_mut(split);

            let (((com_1_0, com_1_1), com_1_2), ((com_2_0, com_2_1), com_2_2)) = rayon::join(
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || inner_product::pairing::<E>(m_a_1, ck_a_1),
                                || inner_product::pairing::<E>(ck_b_1, m_b_1),
                            )
                        },
                        || inner_product::pairing::<E>(m_a_1, m_b_1),
                    )
                },
                || {
                    rayon::join(
                        || {
                            rayon::join(
                                || inner_product::pairing::<E>(m_a_2, ck_a_2),
                                || inner_product::pairing::<E>(ck_b_2, m_b_2),
                            )
                        },
                        || inner_product::pairing::<E>(m_a_2, m_b_2),
                    )
                },
            );

            let com_1 = (com_1_0, com_1_1, com_1_2);
            let com_2 = (com_2_0, com_2_1, com_2_2);

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

                let d = Sha256::digest(&hash_input);
                let c = fr_from_u128::<E::Fr>(d.as_slice());
                if let Some(c_inv) = c.inverse() {
                    // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                    // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                    break 'challenge (c_inv, c);
                }

                counter_nonce += 1;
            };

            // Set up values for next step of recursion
            m_a_1
                .par_iter()
                .zip(m_a_2.par_iter_mut())
                .for_each(|(a_1, a_2)| {
                    let mut x: E::G1 = mul!(a_1.into_projective(), c);
                    x.add_assign_mixed(&a_2);
                    *a_2 = x.into_affine();
                });

            let len = m_a_2.len();
            m_a.resize(len, E::G1Affine::zero()); // shrink to new size

            m_b_1
                .par_iter_mut()
                .zip(m_b_2.par_iter())
                .for_each(|(b_1, b_2)| {
                    let mut x = b_2.into_projective();
                    x.mul_assign(c_inv);
                    x.add_assign_mixed(&b_1);
                    *b_1 = x.into_affine();
                });

            let len = m_b_2.len();
            m_b.resize(len, E::G2Affine::zero()); // shrink to new size

            ck_a_1
                .par_iter_mut()
                .zip(ck_a_2.par_iter())
                .for_each(|(ck_1, ck_2)| {
                    let mut x = ck_2.into_projective();
                    x.mul_assign(c_inv);
                    x.add_assign_mixed(ck_1);
                    *ck_1 = x.into_affine();
                });

            let len = ck_a_1.len();
            ck_a.resize(len, E::G2Affine::zero()); // shrink to new size

            ck_b_1
                .par_iter()
                .zip(ck_b_2.par_iter_mut())
                .for_each(|(ck_1, ck_2)| {
                    let mut x = ck_1.into_projective();
                    x.mul_assign(c);
                    x.add_assign_mixed(ck_2);
                    *ck_2 = x.into_affine();
                });

            let len = ck_b_1.len();
            ck_b.resize(len, E::G1Affine::zero()); // shrink to new size

            r_commitment_steps.push((com_1, com_2));
            r_transcript.push(c);
        }

        // base case
        let m_base = (m_a[0], m_b[0]);
        let ck_base = (ck_a[0], ck_b[0]);

        r_transcript.reverse();
        r_commitment_steps.reverse();

        Ok((
            GIPAProof {
                r_commitment_steps,
                r_base: (m_base.0.into_projective(), m_base.1.into_projective()),
            },
            GIPAAux {
                r_transcript,
                ck_base: (ck_base.0.into_projective(), ck_base.1.into_projective()),
            },
        ))
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
impl<E: Engine> GIPAProofWithSSM<E> {
    /// Returns vector of recursive commitments and transcripts in reverse order.
    fn prove_with_aux(
        values: (&[E::G1Affine], &[E::Fr]),
        ck: &[E::G2Affine],
    ) -> Result<(Self, GIPAAuxWithSSM<E>), SynthesisError> {
        let (mut m_a, mut m_b) = (values.0.to_vec(), values.1.to_vec());
        let mut ck_a = ck.to_vec();

        let mut r_commitment_steps = Vec::new();
        let mut r_transcript = Vec::new();
        if !m_a.len().is_power_of_two() {
            return Err(SynthesisError::MalformedProofs);
        }

        while m_a.len() > 1 {
            // recursive step
            // Recurse with problem of half size
            let split = m_a.len() / 2;

            let (m_a_2, m_a_1) = m_a.split_at_mut(split);
            let (ck_a_1, ck_a_2) = ck_a.split_at_mut(split);
            let (m_b_1, m_b_2) = m_b.split_at_mut(split);

            let (com_1, com_2) = rayon::join(
                || {
                    rayon::join(
                        || inner_product::pairing::<E>(m_a_1, ck_a_1), // LMC::commit
                        || inner_product::multiexponentiation::<E::G1Affine>(m_a_1, m_b_1), // IPC::commit
                    )
                },
                || {
                    rayon::join(
                        || inner_product::pairing::<E>(m_a_2, ck_a_2),
                        || inner_product::multiexponentiation::<E::G1Affine>(m_a_2, m_b_2),
                    )
                },
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

                let d = Sha256::digest(&hash_input);
                let c = fr_from_u128::<E::Fr>(d.as_slice());
                if let Some(c_inv) = c.inverse() {
                    // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                    // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                    break 'challenge (c_inv, c);
                }

                counter_nonce += 1;
            };

            // Set up values for next step of recursion
            m_a_1
                .par_iter()
                .zip(m_a_2.par_iter_mut())
                .for_each(|(a_1, a_2)| {
                    let mut x: E::G1 = mul!(a_1.into_projective(), c);
                    x.add_assign_mixed(&a_2);
                    *a_2 = x.into_affine();
                });

            let len = m_a_2.len();
            m_a.resize(len, E::G1Affine::zero()); // shrink to new size

            m_b_1
                .par_iter_mut()
                .zip(m_b_2.par_iter_mut())
                .for_each(|(b_1, b_2)| {
                    b_2.mul_assign(&c_inv);
                    b_1.add_assign(b_2);
                });

            let len = m_b_1.len();
            m_b.resize(len, E::Fr::zero()); // shrink to new size

            ck_a_1
                .par_iter_mut()
                .zip(ck_a_2.par_iter())
                .for_each(|(ck_1, ck_2)| {
                    let mut x = ck_2.into_projective();
                    x.mul_assign(c_inv);
                    x.add_assign_mixed(ck_1);
                    *ck_1 = x.into_affine();
                });

            let len = ck_a_1.len();
            ck_a.resize(len, E::G2Affine::zero()); // shrink to new size

            r_commitment_steps.push((com_1, com_2));
            r_transcript.push(c);
        }
        // base case
        let m_base = (m_a[0], m_b[0]);
        let ck_base = ck_a[0];

        r_transcript.reverse();
        r_commitment_steps.reverse();

        Ok((
            GIPAProofWithSSM {
                r_commitment_steps,
                r_base: (m_base.0.into_projective(), m_base.1),
            },
            GIPAAuxWithSSM {
                r_transcript,
                ck_base: ck_base.into_projective(),
            },
        ))
    }
}

fn prove_commitment_key_kzg_opening<G: CurveProjective>(
    srs_powers_table: &dyn MultiscalarPrecomp<G::Affine>,
    srs_powers_len: usize,
    transcript: &[G::Scalar],
    r_shift: &G::Scalar,
    kzg_challenge: &G::Scalar,
) -> Result<G, SynthesisError> {
    let ck_polynomial =
        DensePolynomial::from_coeffs(polynomial_coefficients_from_transcript(transcript, r_shift));

    if srs_powers_len != ck_polynomial.coeffs().len() {
        return Err(SynthesisError::MalformedSrs);
    }

    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(&transcript, kzg_challenge, &r_shift);

    let mut neg_kzg_challenge = *kzg_challenge;
    neg_kzg_challenge.negate();

    let quotient_polynomial = &(&ck_polynomial
        - &DensePolynomial::from_coeffs(vec![ck_polynomial_c_eval]))
        / &(DensePolynomial::from_coeffs(vec![neg_kzg_challenge, G::Scalar::one()]));

    let quotient_polynomial_coeffs = quotient_polynomial.into_coeffs();

    // multiexponentiation inner_product, inlined to optimize
    let zero = G::Scalar::zero().into_repr();
    let quotient_polynomial_coeffs_len = quotient_polynomial_coeffs.len();
    let getter = |i: usize| -> <G::Scalar as PrimeField>::Repr {
        if i >= quotient_polynomial_coeffs_len {
            return zero;
        }
        quotient_polynomial_coeffs[i].into_repr()
    };

    Ok(par_multiscalar::<_, G::Affine>(
        &ScalarList::Getter(getter, srs_powers_len),
        srs_powers_table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    ))
}

pub(super) fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: &F,
    r_shift: &F,
) -> F {
    let mut power_2_zr = *z;
    power_2_zr.mul_assign(z);
    power_2_zr.mul_assign(r_shift);

    // 0 iteration
    let mut res = add!(F::one(), &mul!(transcript[0], &power_2_zr));
    power_2_zr.mul_assign(&power_2_zr.clone());

    // the rest
    for x in transcript[1..].iter() {
        res.mul_assign(&add!(F::one(), &mul!(*x, &power_2_zr)));
        power_2_zr.mul_assign(&power_2_zr.clone());
    }

    res
}

fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: &F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = *r_shift;

    for (i, x) in transcript.iter().enumerate() {
        for j in 0..(2_usize).pow(i as u32) {
            let coeff = mul!(coefficients[j], &mul!(*x, &power_2_r));
            coefficients.push(coeff);
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

fn prove_with_structured_scalar_message<E: Engine>(
    srs: &SRS<E>,
    values: (&[E::G1Affine], &[E::Fr]),
    ck: &[E::G2Affine],
) -> Result<MultiExpInnerProductCProof<E>, SynthesisError> {
    // Run GIPA
    let (proof, aux) = GIPAProofWithSSM::<E>::prove_with_aux(values, ck)?;

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
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    // Complete KZG proof
    let ck_a_kzg_opening = prove_commitment_key_kzg_opening(
        &srs.h_beta_powers_table,
        srs.h_beta_powers.len(),
        &transcript_inverse,
        &E::Fr::one(),
        &c,
    )?;

    Ok(MultiExpInnerProductCProof {
        gipa_proof: proof,
        final_ck: ck_a_final,
        final_ck_proof: ck_a_kzg_opening,
    })
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

struct GIPAAux<E: Engine> {
    r_transcript: Vec<E::Fr>,
    ck_base: (E::G2, E::G1),
}

struct GIPAAuxWithSSM<E: Engine> {
    r_transcript: Vec<E::Fr>,
    ck_base: E::G2,
}
