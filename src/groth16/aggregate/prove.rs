use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use itertools::Itertools;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    commit,
    commit::{VKey, WKey},
    inner_product,
    poly::DensePolynomial,
    structured_scalar_power, AggregateProof, GipaMIPP, GipaTIPP, KZGOpening, MIPPProof, TIPPProof,
    SRS,
};
use crate::bls::Engine;
use crate::groth16::{multiscalar::*, Proof};
use crate::SynthesisError;

/// Aggregate `n` zkSnark proofs, where `n` must be a power of two.
/// It implements the algorithm section 5 of the paper.
pub fn aggregate_proofs<E: Engine + std::fmt::Debug>(
    ip_srs: &SRS<E>,
    proofs: &[Proof<E>],
) -> Result<AggregateProof<E>, SynthesisError> {
    let (vkey, wkey) = ip_srs.get_commitment_keys();
    if !vkey.correct_len(proofs.len()) || !wkey.correct_len(proofs.len()) {
        return Err(SynthesisError::MalformedSrs);
    }

    // We first commit to A B and C - these commitments are what the verifier
    // will use later to verify the TIPP and MIPP proofs - even though the TIPP
    // and MIPP proofs doesn't use them directly.
    // TODO parallelize that
    let A = proofs.iter().map(|proof| proof.a).collect::<Vec<_>>();
    let B = proofs.iter().map(|proof| proof.b).collect::<Vec<_>>();
    // A and B are committed together in this scheme
    let com_ab = commit::pair::<E>(&vkey, &wkey, &A, &B);
    let C = proofs.iter().map(|proof| proof.c).collect::<Vec<_>>();
    let com_c = commit::single_g1::<E>(&vkey, &C);

    // Random linear combination of proofs
    // TODO: extract logic in separate function (might require a macro for
    // handling varargs)
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        // TODO use serde to avoid specifying fields by hand
        bincode::serialize_into(&mut hash_input, &com_ab.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_ab.1).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_c.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &com_c.1).expect("vec");

        if let Some(r) = E::Fr::from_random_bytes(&Sha256::digest(&hash_input).as_slice()[..]) {
            break r;
        };

        counter_nonce += 1;
    };
    println!(" ---------------------------- Prove Challenge r: {}", r);

    // r, r^2, r^3, r^4 ...
    let r_vec = structured_scalar_power(proofs.len(), &r);
    // r^-1, r^-2, r^-3
    let r_inv = r_vec
        .par_iter()
        .map(|ri| ri.inverse().unwrap())
        .collect::<Vec<_>>();

    // A^{r}
    let A_r = A
        .par_iter()
        .zip(r_vec.par_iter())
        .map(|(ai, ri)| mul!(ai.into_projective(), ri.into_repr()).into_affine())
        .collect::<Vec<_>>();
    // V^{r^{-1}}
    let vkey_r_inv = vkey.scale(&r_inv);

    let tipa_proof_ab = prove_tipp::<E>(&ip_srs, &A_r, &B, &vkey_r_inv, &wkey, &r);
    let tipa_proof_c = prove_mipp::<E>(
        &ip_srs, &C, &r_vec,
        // v - note we dont use the rescaled here since we dont need the
        // trick as in AB - we just need to commit to C normally.
        &vkey,
    );
    let ip_ab = inner_product::pairing::<E>(&A_r, &B);
    let agg_c = inner_product::multiexponentiation::<E::G1Affine>(&C, &r_vec);

    // TODO - move assertion to a test - this is a property of the scheme
    let computed_com_ab = commit::pair::<E>(&vkey_r_inv, &wkey, &A_r, &B);
    assert_eq!(com_ab, computed_com_ab);

    Ok(AggregateProof {
        com_ab,
        com_c,
        ip_ab,
        agg_c,
        proof_ab: tipa_proof_ab?,
        proof_c: tipa_proof_c?,
    })
}

/// Proves a TIPP relation between A and B. Commitment keys must be of size of A
/// and B. In the context of Groth16 aggregation, we have that A = A^r and vkey
/// is scaled by r^{-1}.
fn prove_tipp<E: Engine>(
    srs: &SRS<E>,
    A: &[E::G1Affine],
    B: &[E::G2Affine],
    vkey: &VKey<E>,
    wkey: &WKey<E>,
    r_shift: &E::Fr,
) -> Result<TIPPProof<E>, SynthesisError> {
    if !A.len().is_power_of_two() || A.len() != B.len() {
        return Err(SynthesisError::MalformedProofs);
    }
    // Run GIPA
    let (proof, mut challenges, mut challenges_inv) = gipa_tipp::<E>(A, B, vkey, wkey);
    println!("\tTIPP.prove challenges: {:?}", challenges.clone());
    // Prove final commitment keys are wellformed
    // we reverse the transcript so the polynomial in kzg opening is constructed
    // correctly - the formula indicates x_{l-j}. Also for deriving KZG
    // challenge point, input must be the last challenge.
    challenges.reverse();
    challenges_inv.reverse();
    let r_inverse = r_shift.inverse().unwrap();

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let z = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &challenges.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_vkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_vkey.1).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_wkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_wkey.1).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    println!("\t PROVE TIPP KZG -> {:?}", &z);
    // Complete KZG proofs
    par! {
        let vkey_opening = prove_commitment_key_kzg_opening(
            &srs.h_alpha_powers_table,
            &srs.h_beta_powers_table,
            srs.n,
            &challenges_inv,
            &r_inverse,
            &z,
        ),
        let wkey_opening = prove_commitment_key_kzg_opening(
            &srs.g_alpha_powers_table,
            &srs.g_beta_powers_table,
            srs.n,
            &challenges,
            &<E::Fr>::one(),
            &z,
        )
    };

    Ok(TIPPProof {
        gipa: proof,
        vkey_opening: vkey_opening?,
        wkey_opening: wkey_opening?,
    })
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
) -> (GipaTIPP<E>, Vec<E::Fr>, Vec<E::Fr>) {
    let (mut m_a, mut m_b) = (A.to_vec(), B.to_vec());
    let (mut vkey, mut wkey) = (vkey.clone(), wkey.clone());
    let mut comms = Vec::new();
    let mut z_vec = Vec::new();
    let mut challenges: Vec<E::Fr> = Vec::new();
    let mut challenges_inv: Vec<E::Fr> = Vec::new();

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
        let ((C_l, C_r), (Z_l, Z_r)) = rayon::join(
            || {
                rayon::join(
                    || commit::pair::<E>(&vk_left, &wk_right, &A_right, &B_left),
                    || commit::pair::<E>(&vk_right, &wk_left, &A_left, &B_right),
                )
            },
            || {
                rayon::join(
                    || inner_product::pairing::<E>(&A_right, &B_left),
                    || inner_product::pairing::<E>(&A_left, &B_right),
                )
            },
        );
        let (T_l, U_l) = C_l;
        let (T_r, U_r) = C_r;

        // Fiat-Shamir challenge
        // TODO extract logic in separate function and use the same as in
        // verification
        let mut counter_nonce: usize = 0;
        let default_transcript = E::Fr::zero();
        let transcript = challenges.last().unwrap_or(&default_transcript);

        let (c, c_inv) = 'challenge: loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            bincode::serialize_into(&mut hash_input, &transcript).expect("vec");
            bincode::serialize_into(&mut hash_input, &T_l).expect("vec");
            bincode::serialize_into(&mut hash_input, &U_l).expect("vec");
            bincode::serialize_into(&mut hash_input, &T_r).expect("vec");
            bincode::serialize_into(&mut hash_input, &U_r).expect("vec");
            bincode::serialize_into(&mut hash_input, &Z_r).expect("vec");
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
        A_left
            .par_iter_mut()
            .zip(A_right.par_iter())
            .for_each(|(a_l, a_r)| {
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
                let mut x: E::G2 = mul!(b_r.into_projective(), c_inv);
                x.add_assign_mixed(&b_l);
                *b_l = x.into_affine();
            });

        let len = B_right.len();
        m_b.resize(len, E::G2Affine::zero()); // shrink to new size

        // v_left + v_right^x^-1
        vkey = VKey::<E>::compress(&vk_left, &vk_right, &c_inv);
        // w_left + w_right^x
        wkey = WKey::<E>::compress(&wk_left, &wk_right, &c);

        comms.push((C_l, C_r));
        z_vec.push((Z_l, Z_r));
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    assert!(m_a.len() == 1 && m_b.len() == 1);
    assert!(vkey.a.len() == 1 && vkey.b.len() == 1);
    assert!(wkey.a.len() == 1 && wkey.b.len() == 1);

    let (final_A, final_B) = (m_a[0], m_b[0]);
    let (final_vkey, final_wkey) = (vkey.first(), wkey.first());
    (
        GipaTIPP {
            comms: comms,
            z_vec: z_vec,
            final_A: final_A,
            final_B: final_B,
            final_vkey: final_vkey,
            final_wkey: final_wkey,
        },
        challenges,
        challenges_inv,
    )
}

/// gipa_mipp proves the relation Z = C^r and V = C * v
/// Returns vector of recursive commitments and transcripts in reverse order.
fn gipa_mipp<E: Engine>(
    C: &[E::G1Affine],
    r: &[E::Fr],
    vkey: &VKey<E>,
) -> (GipaMIPP<E>, Vec<E::Fr>) {
    let (mut m_c, mut m_r) = (C.to_vec(), r.to_vec());
    let mut comms = Vec::new();
    let mut z_vec = Vec::new();
    let mut challenges = Vec::new();
    let mut vkey = vkey.clone();

    while m_c.len() > 1 {
        // recursive step
        // Recurse with problem of half size
        let split = m_c.len() / 2;

        // c[:n']   c[n':]
        let (C_left, C_right) = m_c.split_at_mut(split);
        // r[:n']   r[:n']
        let (r_left, r_right) = m_r.split_at_mut(split);
        // v[:n']   v[n':]
        let (vk_left, vk_right) = vkey.split(split);

        let ((Z_r, Z_l), (TU_r, TU_l)) = rayon::join(
            || {
                rayon::join(
                    // Z_r = c[:n'] ^ r[n':]
                    || inner_product::multiexponentiation::<E::G1Affine>(C_left, r_right),
                    // Z_l = c[n':] ^ r[:n']
                    || inner_product::multiexponentiation::<E::G1Affine>(C_right, r_left),
                )
            },
            || {
                rayon::join(
                    // U_r = c[:n'] * v[n':]
                    || commit::single_g1::<E>(&vk_right, C_left),
                    // U_l = c[n':] * v[:n']
                    || commit::single_g1::<E>(&vk_left, C_right),
                )
            },
        );

        // Fiat-Shamir challenge
        // TODO move that to separate function
        let mut counter_nonce: usize = 0;
        let default_transcript = E::Fr::zero();
        let transcript = challenges.last().unwrap_or(&default_transcript);

        let (c, c_inv) = 'challenge: loop {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
            bincode::serialize_into(&mut hash_input, &transcript).expect("vec");
            bincode::serialize_into(&mut hash_input, &TU_r.0).expect("vec");
            bincode::serialize_into(&mut hash_input, &TU_r.1).expect("vec");
            bincode::serialize_into(&mut hash_input, &TU_l.0).expect("vec");
            bincode::serialize_into(&mut hash_input, &TU_l.1).expect("vec");
            bincode::serialize_into(&mut hash_input, &Z_r).expect("vec");
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
        C_right
            .par_iter()
            .zip(C_left.par_iter_mut())
            .for_each(|(c_r, c_l)| {
                // c[:n'] + c[n':]^x
                let mut x: E::G1 = mul!(c_r.into_projective(), c);
                x.add_assign_mixed(&c_l);
                *c_l = x.into_affine();
            });

        let len = C_left.len();
        m_c.resize(len, E::G1Affine::zero()); // shrink to new size

        r_left
            .par_iter_mut()
            .zip(r_right.par_iter_mut())
            .for_each(|(r_l, r_r)| {
                // r[:n'] + r[n':]^x^-1
                r_r.mul_assign(&c_inv);
                r_l.add_assign(r_r);
            });

        let len = r_left.len();
        m_r.resize(len, E::Fr::zero()); // shrink to new size

        // v[:n'] + v[n':]^{x^{-1}}
        vkey = VKey::<E>::compress(&vk_left, &vk_right, &c_inv);

        comms.push((TU_l, TU_r));
        z_vec.push((Z_l, Z_r));
        challenges.push(c);
    }

    // final c and r
    let (final_C, final_r) = (m_c[0], m_r[0]);
    // final v
    let final_vkey = vkey.first();
    (
        GipaMIPP {
            comms: comms,
            z_vec: z_vec,
            final_C: final_C,
            final_r: final_r,
            final_vkey: final_vkey,
        },
        challenges,
    )
}

/// Returns the KZG opening proof for the given commitment key. In math, it
/// returns $g^{f(alpha) - f(z) / (alpha - z)}$ for $a$ and $b$.
fn prove_commitment_key_kzg_opening<G: CurveAffine>(
    srs_powers_alpha_table: &dyn MultiscalarPrecomp<G>,
    srs_powers_beta_table: &dyn MultiscalarPrecomp<G>,
    srs_powers_len: usize,
    transcript: &[G::Scalar],
    r_shift: &G::Scalar,
    kzg_challenge: &G::Scalar,
) -> Result<KZGOpening<G>, SynthesisError> {
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
    let quotient_polynomial = &(&vkey_poly - &DensePolynomial::from_coeffs(vec![vkey_poly_z]))
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

    // we do one proof over h^a and one proof over h^b (or g^a and g^b depending
    // on the curve we are on). that's the extra cost of the commitment scheme
    // used which is compatible with Groth16 CRS.
    Ok(rayon::join(
        || {
            par_multiscalar::<_, G>(
                &ScalarList::Getter(getter, srs_powers_len),
                srs_powers_alpha_table,
                std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
            )
            .into_affine()
        },
        || {
            par_multiscalar::<_, G>(
                &ScalarList::Getter(getter, srs_powers_len),
                srs_powers_beta_table,
                std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
            )
            .into_affine()
        },
    ))
}

/// It returns the evaluation of the polynomial $\prod (1 + x_{l-j}(rX)^{2j}$ at
/// the point z, where transcript contains the reversed order of all challenges (the x).
pub(super) fn polynomial_evaluation_product_form_from_transcript<F: Field>(
    transcript: &[F],
    z: &F,
    r_shift: &F,
) -> F {
    // this is the term (rz) that will get squared at each step to produce the
    // $(rz)^{2j}$ of the formula
    let mut power_zr = *z;
    power_zr.mul_assign(r_shift);

    // 0 iteration
    let mut res = add!(F::one(), &mul!(transcript[0], &power_zr));
    power_zr.mul_assign(&power_zr.clone());

    // the rest
    for x in transcript[1..].iter() {
        res.mul_assign(&add!(F::one(), &mul!(*x, &power_zr)));
        power_zr.mul_assign(&power_zr.clone());
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
///
/// This method expects the coefficients in reverse order so transcript[i] =
/// x_{l-j}.
fn polynomial_coefficients_from_transcript<F: Field>(transcript: &[F], r_shift: &F) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    let mut power_2_r = *r_shift;

    for (i, x) in transcript.iter().enumerate() {
        //for j in 0..(2_usize).pow(i as u32) {
        let n = coefficients.len();
        for j in (0..n) {
            let coeff = mul!(coefficients[j], &mul!(*x, &power_2_r));
            coefficients.push(coeff);
        }
        power_2_r.mul_assign(&power_2_r.clone());
    }
    coefficients
}

/// prove_mipp returns a GIPA and MIPP proof for proving statement Z = C^r
/// and T = C * v. Section 4 in the paper.
fn prove_mipp<E: Engine>(
    srs: &SRS<E>,
    C: &[E::G1Affine],
    r: &[E::Fr],
    vkey: &VKey<E>,
) -> Result<MIPPProof<E>, SynthesisError> {
    if !C.len().is_power_of_two() || C.len() != r.len() {
        return Err(SynthesisError::MalformedProofs);
    }
    // Run GIPA
    let (proof, mut challenges) = gipa_mipp::<E>(C, r, vkey);

    // Prove final commitment key is wellformed
    // we reverse the transcript so challenges are in the right order (inverse
    // from creation) for the KZG opening proof
    challenges.reverse();
    let challenges_inv = challenges
        .iter()
        .map(|x| x.inverse().unwrap())
        .collect::<Vec<_>>();

    // KZG challenge point
    // TODO move to separate function (or macro)
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        // we take the last challenge generated
        bincode::serialize_into(&mut hash_input, &challenges.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_vkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.final_vkey.1).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    println!(" +++ MIPP - prove: challenges {:?}", challenges);
    println!(" +++ MIPP - prove: c {:?}", c);

    // Complete KZG proof
    let vkey_opening = prove_commitment_key_kzg_opening(
        &srs.h_alpha_powers_table,
        &srs.h_beta_powers_table,
        srs.n,
        &challenges_inv,
        &E::Fr::one(),
        &c,
    );

    /*    {*/
    //// f_v
    //let vkey_poly = DensePolynomial::from_coeffs(polynomial_coefficients_from_transcript(
    //transcript, r_shift,
    //));
    //}

    Ok(MIPPProof {
        gipa: proof,
        vkey_opening: vkey_opening?,
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
