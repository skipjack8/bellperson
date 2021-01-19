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
    let (vkey, wkey) = ip_srs.get_commitment_keys();

    if vkey.correct_ley(proofs.len()) || wkey.correct_len(proofs.len()) {
        return Err(SynthesisError::MalformedSrs);
    }

    par! {
        let (A,B, com_ab) = {
            let A = proofs.iter().map(|proof| proof.a).collect::<Vec<_>>();
            let B = proofs.iter().map(|proof| proof.b).collect::<Vec<_>>();
            let com_ab = commit::pair::<E>(&vkey, &wkey,&A,&B);
            (A,B, com_ab)
        },
        let (C, com_c) = {
            let C = proofs.iter().map(|proof| proof.c).collect::<Vec<_>>();
            let com_c = commit::single_g1::<E>(&vkey, &C);
            (C, com_c)
        }
    }

    // Random linear combination of proofs
    // TODO: extract logic in separate function (might require a macro for
    // handling varargs)
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
    let r_inv = r_vec
        .par_iter()
        .map(|r| r.inverse().unwrap())
        .collect::<Vec<_>>();

    par! {
        // A^{r}
        let A_r = A.par_iter()
                   .zip(r_vec.par_iter())
                   .map(|(a, r)| mul!(a.into_projective(), *r).into_affine())
                   .collect::<Vec<E::G1Affine>>(),
        // V^{r^{-1}}
        let vkey_r_inv = vkey.scale(r_inv),
    };

    par! {
        let tipa_proof_ab = prove_with_srs_shift::<E>(
                &ip_srs,
                (&A_r, &B),
                (&vkey_r_inv, &wkey),
                &r,
        ),
        let tipa_proof_c = prove_with_structured_scalar_message::<E>(
            &ip_srs,
            // c^r
            (&C, &r_vec),
            // v - note we dont use the rescaled here since we dont need the
            // trick as in AB - we just need to commit to C normally.
            &vkey,
        ),
        let ip_ab = inner_product::pairing::<E>(&A_r, &B),
        let agg_c = inner_product::multiexponentiation::<E::G1Affine>(&C, r_vec)
    };

    // TODO - move assertion to a test - this is a property of the scheme
    let computed_com_ab = commit::pair::<E>(&vkey_r_inv, &wkey, &A, &B);
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

/// Proves a TIPP relation between A and B. Commitment keys must be of size of A
/// and B. In the context of Groth16 aggregation, we have that A = A^r and vkey
/// is scaled by r^{-1}.
fn prove_tipp<E: Engine>(
    srs: &SRS<E>,
    A: &[E::G1Affine],
    B: &[E::G2Affine], // values: (&[E::G1Affine], &[E::G2Affine]),
    vkey: &VKey<E>,
    wkey: &WKey<E>  // vkey: (&[E::G2Affine], &[E::G1Affine]),
    r_shift: &E::Fr,
) -> Result<PairingInnerProductABProof<E>, SynthesisError> {
    if !m_a.len().is_power_of_two() {
            return Err(SynthesisError::MalformedProofs);
    }
    // Run GIPA
    let (proof, challenges) = gipa_tipp(A,B, vkey,wkey)?;

    // Prove final commitment keys are wellformed
    let transcript = challenges;
    let transcript_inverse = challenges
        .par_iter()
        .map(|x| x.inverse().unwrap())
        .collect::<Vec<_>>();
    let r_inverse = r_shift.inverse().unwrap();

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let z = loop {
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
        let vkey_opening = prove_commitment_key_kzg_opening(
            srs.h_alpha_powers_table,
            srs.h_beta_powers_table,
            srs.n,
            &transcript_inverse,
            &r_inverse,
            &z,
        ),
        let wkey_opening = prove_commitment_key_kzg_opening(
            srs.g_alpha_powers_table,
            srs.g_beta_powers_table,
            srs.g_beta_powers,
            &transcript,
            &<E::Fr>::one(),
            &z,
        )
    };

    TIPPProof {
        gipa: proof,
        vkey_opening: vkey_opening,
        wkey_opening: wkey_opening,
    }
}

/// gipa_tipp peforms the recursion of the GIPA protocol for TIPP. It returns a
/// proof containing all intermdiate committed values, as well as the
/// challenges generated necessary to do the polynomial commitment proof later
/// in TIPP.
fn gipa_tipp<E: Engine>(
        A: &[E::G1Affine],
        B: &[E::G2Affine],
        vkey: &VKey<E>, 
        wkey: &WKey<E>,
    ) -> GipaTIPP<E> {
        let (mut m_a, mut m_b) = (A.to_vec(), B.to_vec());
        let (mut vkey, mut wkey) = (vkey.clone(), wkey.clone());
        let mut comms = Vec::new();
        let mut z_vec = Vec::new();
        let mut challenges = Vec::new();

            while m_a.len() > 1 {
            // recursive step
            // Recurse with problem of half size
            let split = m_a.len() / 2;
            
            let (A_left, A_right) = m_a.split_at_mut(split);
            let (B_left, B_right) = m_b.split_at_mut(split);
            // TODO: make that mutable split to avoid copying - may require to
            // not use struct...  for the moment i prefer readability 
            let (vk_left, vk_right) = vkey.split(split);
            let (wk_left, wk_right) = wkey.split(split);

            // See section 3.3 for paper version with equivalent names
            let ((C_r,C_l),(Z_l,Z_r))= rayon::join(
                || {
                    rayon::join(
                        || commit::pair(vk_left,wk_right,A_right,B_left),
                        || commit::pair(vk_right,wk_left,A_left,B_right)
                    )
                },
                || {
                    rayon::join(
                        || inner_product::pairing::<E>(A_right, B_left),
                        || inner_product::pairing::<E>(A_left, B_right)
                    )
                },
            );

            // Fiat-Shamir challenge
            // TODO extract logic in separate function
            let mut counter_nonce: usize = 0;
            let default_transcript = E::Fr::zero();
            let transcript = r_transcript.last().unwrap_or(&default_transcript);

            let (c, c_inv) = 'challenge: loop {
                let mut hash_input = Vec::new();
                hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
                bincode::serialize_into(&mut hash_input, &transcript).expect("vec");

                bincode::serialize_into(&mut hash_input, &C_r.0).expect("vec");
                bincode::serialize_into(&mut hash_input, &C_r.1).expect("vec");
                bincode::serialize_into(&mut hash_input, &Z_r).expect("vec");

                bincode::serialize_into(&mut hash_input, &C_l.0).expect("vec");
                bincode::serialize_into(&mut hash_input, &C_r.1).expect("vec");
                bincode::serialize_into(&mut hash_input, &Z_l).expect("vec");

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
            // A[:n'] + A[n':] ^ x
            A_right
                .par_iter()
                .zip(A_left.par_iter_mut())
                .for_each(|(a_r, a_l)| {
                    let mut x: E::G1 = mul!(a_r.into_projective(), c);
                    x.add_assign_mixed(&a_l);
                    *a_l = x.into_affine();
                });

            let len = A_left.len();
            m_a.resize(len, E::G1Affine::zero()); // shrink to new size

            // B[:n'] + B[n':] ^ x^-1
            B_left
                .par_iter_mut()
                .zip(B_right.par_iter())
                .for_each(|(b_l, b_r)| {
                    let mut x = b_r.into_projective();
                    x.mul_assign(c_inv);
                    x.add_assign_mixed(&b_l);
                    *b_l = x.into_affine();
                });

            let len = B_right.len();
            m_b.resize(len, E::G2Affine::zero()); // shrink to new size

            // v_left + v_right^x^-1
            vkey = VKey::<E>::compress(vk_left,vk_right,c_inv);
            // w_left + w_right^x
            wkey = WKey::<E>::compress(wk_left,wk_right,c);

            comms.push((C_l,C_r));
            z_vec.push((Z_l,Z_r))
            challenges.push(c);
        }

        let (final_A, final_B) = (m_a[0], m_b[0]);
        let (final_vkey, final_wkey) = (vkey.first(), wkey.first());

        // TODO should we reverse?
        //r_transcript.reverse();
        //r_commitment_steps.reverse();
        (GipaTIPP{
                comms: comms,
                z_vec: z_vec,
                final_A: final_A,
                final_B: final_B,
                final_vkey: final_vkey,
                final_wkey: final_wkey,
        }, challenges)
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
        // c  and r
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

            // c[:n']   c[n':]
            let (m_a_2, m_a_1) = m_a.split_at_mut(split);
            // v[:n']   v[n':]
            let (ck_a_1, ck_a_2) = ck_a.split_at_mut(split);
            // r[:n']   r[:n']
            let (m_b_1, m_b_2) = m_b.split_at_mut(split);

            let (com_1, com_2) = rayon::join(
                || {
                    rayon::join(
                        // U_r = c[n':] * v[:n']
                        || inner_product::pairing::<E>(m_a_1, ck_a_1), // LMC::commit
                        // C_r = c[n':] ^ r[:n']
                        || inner_product::multiexponentiation::<E::G1Affine>(m_a_1, m_b_1), // IPC::commit
                    )
                },
                || {
                    rayon::join(
                        // U_l = c[:n'] * v[n':]
                        || inner_product::pairing::<E>(m_a_2, ck_a_2),
                        // Z_l = c[:n'] ^ r[n':]
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
                    // c[:n'] + c[n':]^x
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
                    // r[:n'] + r[n':]^x^-1
                    b_2.mul_assign(&c_inv);
                    b_1.add_assign(b_2);
                });

            let len = m_b_1.len();
            m_b.resize(len, E::Fr::zero()); // shrink to new size

            ck_a_1
                .par_iter_mut()
                .zip(ck_a_2.par_iter())
                .for_each(|(ck_1, ck_2)| {
                    // v[:n'] + v[n':]^x^-1
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
        // final c and r
        let m_base = (m_a[0], m_b[0]);
        // final v
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

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
type KZGOpening<G: CurveProjective> = (G,G);

/// Returns the KZG opening proof for the given commitment key. In math, it
/// returns $g^{f(alpha) - f(z) / (alpha - z)}$ for $a$ and $b$.
fn prove_commitment_key_kzg_opening<G: CurveProjective>(
    srs_powers_alpha_table: &dyn MultiscalarPrecomp<G::Affine>,
    srs_powers_beta_table: &dyn MultiscalarPrecomp<G::Affine>,
    srs_powers_len: usize,
    transcript: &[G::Scalar],
    r_shift: &G::Scalar,
    kzg_challenge: &G::Scalar,
) -> KZGOpening {
    // f_v
    let vkey_poly =
        DensePolynomial::from_coeffs(polynomial_coefficients_from_transcript(transcript, r_shift));

    if srs_powers_len != vkey_poly.coeffs().len() {
        return Err(SynthesisError::MalformedSrs);
    }

    // f_v(z)
    let vkey_poly_z =
        polynomial_evaluation_product_form_from_transcript(&transcript, kzg_challenge, &r_shift);

    let mut neg_kzg_challenge = *kzg_challenge;
    neg_kzg_challenge.negate();

    // f_v(X) - f_v(z) / (X - z)
    let quotient_polynomial = &(&vkey_poly
        - &DensePolynomial::from_coeffs(vec![vkey_poly_z]))
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

    KZGOpening(rayon::join( || par_multiscalar::<_, G::Affine>(
        &ScalarList::Getter(getter, srs_powers_len),
        srs_powers_alpha_table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    ),
    || par_multiscalar::<_, G::Affine>(
        &ScalarList::Getter(getter, srs_powers_len),
        srs_powers_alpha_table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    )))
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

/// Compute the coefficients of the polynomial $\prod_{j=0}^{l-1} (1 + x_{l-j}(rX)^{2j})$
/// It does this in logarithmic time directly; here is an example with 2
/// challenges:
///
///     We wish to compute $(1+x_1ra)(1+x_0(ra)^2) = 1 +  x_1ra + x_0(ra)^2 + x_0x_1(ra)^3$ 
///     Algorithm: $c_{-1} = [1]$; $c_j = c_{i-1} \| (x_{l-j} * c_{i-1})$; $r = r*r$
///     $c_0 = c_{-1} \| (x_1 * r * c_{-1}) = [1] \| [rx_1] = [1, rx_1]$, $r = r^2$
///     $c_1 = c_0 \| (x_0 * r^2c_0) = [1, rx_1] \| [x_0r^2, x_0x_1r^3] = [1, x_1r, x_0r^2, x_0x_1r^3]$ 
///     which is equivalent to $f(a) = 1 + x_1ra + x_0(ra)^2 + x_0x_1r^2a^3$
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
    coefficients
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

struct GIPAAuxWithSSM<E: Engine> {
    r_transcript: Vec<E::Fr>,
    ck_base: E::G2,
}
