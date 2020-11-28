use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use log::*;
use rayon::prelude::*;
use std::sync::mpsc::channel;

use super::HomomorphicPlaceholderValue;
use super::{
    inner_product,
    prove::{fr_from_u128, polynomial_evaluation_product_form_from_transcript},
    structured_scalar_power, AggregateProof, GIPAProof, GIPAProofWithSSM,
    MultiExpInnerProductCProof, PairingInnerProductABProof, VerifierSRS,
};
use crate::bls::Engine;
use crate::groth16::VerifyingKey;
use paired::PairingCurveAffine;

pub fn verify_aggregate_proof<E: Engine + std::fmt::Debug, D: Digest + Sync>(
    ip_verifier_srs: &VerifierSRS<E>,
    vk: &VerifyingKey<E>,
    public_inputs: &Vec<Vec<E::Fr>>,
    proof: &AggregateProof<E, D>,
) -> bool {
    info!("verify_aggregate_proof");
    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        hash_input.extend_from_slice(&proof.com_a.as_bytes());
        hash_input.extend_from_slice(&proof.com_b.as_bytes());
        hash_input.extend_from_slice(&proof.com_c.as_bytes());
        if let Some(r) =
            E::Fr::from_random_bytes(&D::digest(&hash_input).as_slice()[..E::Fr::SERIALIZED_BYTES])
        {
            break r;
        };
        counter_nonce += 1;
    };

    let mut tipa_proof_ab_valid = E::Fqk::zero();
    let mut tipa_proof_c_valid = E::Fqk::zero();
    let mut p1 = E::Fqk::zero();
    let mut p2 = E::Fqk::zero();
    let mut p3 = E::Fqk::zero();

    assert_eq!(vk.ic.len(), public_inputs[0].len() + 1);

    rayon::scope(|s| {
        // Check TIPA proofs
        let valid = &mut tipa_proof_ab_valid;
        s.spawn(move |_| {
            *valid = verify_with_srs_shift::<E, D>(
                ip_verifier_srs,
                &HomomorphicPlaceholderValue,
                (&proof.com_a, &proof.com_b, &vec![proof.ip_ab]),
                &proof.tipa_proof_ab,
                &r,
            );
        });

        let valid = &mut tipa_proof_c_valid;
        s.spawn(move |_| {
            *valid = verify_with_structured_scalar_message::<E, D>(
                ip_verifier_srs,
                &HomomorphicPlaceholderValue,
                (&proof.com_c, &vec![proof.agg_c]),
                &r,
                &proof.tipa_proof_c,
            );
        });

        // Check aggregate pairing product equation
        info!("checking aggregate pairing");
        let r_sum = {
            let a = sub!(r.pow(&[public_inputs.len() as u64]), &E::Fr::one());
            let b = sub!(r, &E::Fr::one());
            let b = b.inverse().unwrap();
            mul!(a, &b)
        };

        let p1 = &mut p1;
        let p2 = &mut p2;
        let p3 = &mut p3;

        s.spawn(move |_| {
            *p1 = {
                let mut alpha_g1_r_sum = vk.alpha_g1.into_projective();
                alpha_g1_r_sum.mul_assign(r_sum);
                E::miller_loop(&[(
                    &alpha_g1_r_sum.into_affine().prepare(),
                    &vk.beta_g2.prepare(),
                )])
            };
            *p3 = E::miller_loop(&[(&proof.agg_c.into_affine().prepare(), &vk.delta_g2.prepare())]);
        });

        let (r_vec_sender, r_vec_receiver) = channel();
        s.spawn(move |_| {
            r_vec_sender
                .send(structured_scalar_power(public_inputs.len(), &r))
                .unwrap();
        });

        s.spawn(move |_| {
            let mut g_ic = vk.ic[0].into_projective();
            g_ic.mul_assign(r_sum);

            let r_vec = r_vec_receiver.recv().unwrap();
            let b = vk
                .ic
                .par_iter()
                .skip(1)
                .enumerate()
                .map(|(i, b)| (i, b.into_projective()))
                .map(|(i, b)| {
                    let ip = inner_product::scalar(
                        &public_inputs
                            .iter()
                            .map(|inputs| inputs[i].clone())
                            .collect::<Vec<E::Fr>>(),
                        &r_vec,
                    );

                    let mut b = b;
                    b.mul_assign(ip);
                    b
                })
                .reduce(
                    || E::G1::zero(),
                    |mut acc, curr| {
                        acc.add_assign(&curr);
                        acc
                    },
                );

            g_ic.add_assign(&b);
            *p2 = E::miller_loop(&[(&g_ic.into_affine().prepare(), &vk.gamma_g2.prepare())]);
        });
    });

    let p1_p2_p3 = mul!(p1, &mul!(p2, &p3));
    // if proofs are valid, proof_ab and proof_ac are equal to 1 (after final
    // exponentiation) so multiply
    // won't change the results, it's still p1_p2_p3
    let proof_and_p123 = mul!(mul!(tipa_proof_ab_valid, &tipa_proof_c_valid), &p1_p2_p3);
    let p123 = E::final_exponentiation(&proof_and_p123).unwrap();
    let ppe_valid = proof.ip_ab == p123;

    info!("aggregate verify done");
    ppe_valid
}

fn verify_with_srs_shift<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    _ck_t: &HomomorphicPlaceholderValue,
    com: (&E::Fqk, &E::Fqk, &Vec<E::Fqk>),
    proof: &PairingInnerProductABProof<E, D>,
    r_shift: &E::Fr,
) -> E::Fqk {
    info!("verify with srs shift");
    let (base_com, transcript) = gipa_verify_recursive_challenge_transcript(com, &proof.gipa_proof);
    let transcript_inverse = transcript.iter().map(|x| x.inverse().unwrap()).collect();

    // Verify commitment keys wellformed
    let (ck_a_final, ck_b_final) = &proof.final_ck;
    let (ck_a_proof, ck_b_proof) = &proof.final_ck_proof;

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        hash_input.extend_from_slice(&transcript.first().unwrap().as_bytes());
        hash_input.extend_from_slice(&ck_a_final.as_bytes());
        hash_input.extend_from_slice(&ck_b_final.as_bytes());

        if let Some(c) =
            E::Fr::from_random_bytes(&D::digest(&hash_input).as_slice()[..E::Fr::SERIALIZED_BYTES])
        {
            break c;
        };
        counter_nonce += 1;
    };

    // if valid aid == 1
    let aid = verify_commitment_key_g2_kzg_opening(
        v_srs,
        &ck_a_final,
        &ck_a_proof,
        &transcript_inverse,
        &r_shift.inverse().unwrap(),
        &c,
    );
    // if valid bid == 1
    let bid = verify_commitment_key_g1_kzg_opening(
        v_srs,
        &ck_b_final,
        &ck_b_proof,
        &transcript,
        &E::Fr::one(),
        &c,
    );

    // Verify base inner product commitment
    let (com_a, com_b, com_t) = base_com;
    let a_base = vec![proof.gipa_proof.r_base.0.clone()];
    let b_base = vec![proof.gipa_proof.r_base.1.clone()];
    let t_base = vec![inner_product::pairing::<E>(&a_base, &b_base)];

    let base_valid = {
        // LMC::verify - pairing inner product<E>
        let a = inner_product::pairing::<E>(&a_base, &vec![ck_a_final.clone()]) == com_a;
        // RMC::verify - afgho commitment G1
        let b = inner_product::pairing::<E>(&vec![ck_b_final.clone()], &b_base) == com_b;
        // IPC::verify - identity commitment<Fqk, Fr>
        let c = t_base == com_t;

        a && b && c
    };

    // we return a Fqk element that is supposed to be equal to Fqk::one when
    // put into the final exponentiation
    if base_valid {
        mul!(aid, &bid)
    } else {
        E::Fqk::zero()
    }
}

fn gipa_verify_recursive_challenge_transcript<E: Engine, D: Digest>(
    com: (&E::Fqk, &E::Fqk, &Vec<E::Fqk>),
    proof: &GIPAProof<E, D>,
) -> ((E::Fqk, E::Fqk, Vec<E::Fqk>), Vec<E::Fr>) {
    info!("gipa verify recursive challenge transcript");
    let (com_0, com_1, com_2) = com.clone();
    let (mut com_a, mut com_b, mut com_t) = (*com_0, *com_1, com_2.clone());
    let mut r_transcript = Vec::new();
    for (com_1, com_2) in proof.r_commitment_steps.iter().rev() {
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

            let d = D::digest(&hash_input);
            let c = fr_from_u128::<E::Fr>(d.as_slice());

            if let Some(c_inv) = c.inverse() {
                // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                break 'challenge (c_inv, c);
            }
            counter_nonce += 1;
        };

        #[inline]
        /// (x * c) * y * (z * c_inv)
        fn lambda<E: Engine>(
            x: E::Fqk,
            y: &E::Fqk,
            z: &E::Fqk,
            c: &E::Fr,
            c_inv: &E::Fr,
        ) -> E::Fqk {
            // x * c
            let x_c = x.pow(c.into_repr());
            // z * c_inv
            let z_c_inv = z.pow(c_inv.into_repr());

            mul!(mul!(x_c, y), &z_c_inv)
        }

        com_a = lambda::<E>(com_1.0, &com_a, &com_2.0, &c, &c_inv);
        com_b = lambda::<E>(com_1.1, &com_b, &com_2.1, &c, &c_inv);
        assert_eq!(com_1.2.len(), com_t.len());
        assert_eq!(com_2.2.len(), com_t.len());
        com_t = {
            let x_c = com_1.2.iter().map(|com_1_2| com_1_2.pow(c.into_repr()));
            let z_c_inv = com_2.2.iter().map(|com_2_2| com_2_2.pow(c_inv.into_repr()));
            x_c.zip(com_t.iter())
                .zip(z_c_inv)
                .map(|((x_c, y), z_c_inv)| mul!(mul!(x_c, y), &z_c_inv))
                .collect()
        };

        r_transcript.push(c);
    }
    r_transcript.reverse();
    ((com_a, com_b, com_t), r_transcript)
}

pub fn verify_commitment_key_g2_kzg_opening<E: Engine>(
    v_srs: &VerifierSRS<E>,
    ck_final: &E::G2,
    ck_opening: &E::G2,
    transcript: &Vec<E::Fr>,
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> E::Fqk {
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);

    let p1 = E::miller_loop(&[(
        &v_srs.g.into_affine().prepare(),
        &sub!(*ck_final, &mul!(v_srs.h, ck_polynomial_c_eval))
            .into_affine()
            .prepare(),
    )]);
    let p2 = E::miller_loop(&[(
        &sub!(v_srs.g_beta, &mul!(v_srs.g, kzg_challenge.clone()))
            .into_affine()
            .prepare(),
        &ck_opening.into_affine().prepare(),
    )]);
    let ip1 = p1.inverse().unwrap();
    mul!(ip1, &p2)
}

pub fn verify_commitment_key_g1_kzg_opening<E: Engine>(
    v_srs: &VerifierSRS<E>,
    ck_final: &E::G1,
    ck_opening: &E::G1,
    transcript: &Vec<E::Fr>,
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> E::Fqk {
    let ck_polynomial_c_eval =
        polynomial_evaluation_product_form_from_transcript(transcript, kzg_challenge, r_shift);
    let p1 = E::miller_loop(&[(
        &sub!(*ck_final, &mul!(v_srs.g, ck_polynomial_c_eval))
            .into_affine()
            .prepare(),
        &v_srs.h.into_affine().prepare(),
    )]);
    let p2 = E::miller_loop(&[(
        &ck_opening.into_affine().prepare(),
        &sub!(v_srs.h_alpha, &mul!(v_srs.h, *kzg_challenge))
            .into_affine()
            .prepare(),
    )]);
    let ip1 = p1.inverse().unwrap();
    mul!(ip1, &p2)
}

fn verify_with_structured_scalar_message<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    _ck_t: &HomomorphicPlaceholderValue,
    com: (&E::Fqk, &Vec<E::G1>),
    scalar_b: &E::Fr,
    proof: &MultiExpInnerProductCProof<E, D>,
) -> E::Fqk {
    info!("verify with structured scalar message");
    let (base_com, transcript) = gipa_with_ssm_verify_recursive_challenge_transcript(
        (com.0, scalar_b, com.1),
        &proof.gipa_proof,
    );
    let transcript_inverse = transcript.iter().map(|x| x.inverse().unwrap()).collect();

    let ck_a_final = &proof.final_ck;
    let ck_a_proof = &proof.final_ck_proof;

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        hash_input.extend_from_slice(&transcript.first().unwrap().as_bytes());
        hash_input.extend_from_slice(&ck_a_final.as_bytes());
        if let Some(c) =
            E::Fr::from_random_bytes(&D::digest(&hash_input).as_slice()[..E::Fr::SERIALIZED_BYTES])
        {
            break c;
        };
        counter_nonce += 1;
    };

    // Check commitment key
    let aid = verify_commitment_key_g2_kzg_opening(
        v_srs,
        &ck_a_final,
        &ck_a_proof,
        &transcript_inverse,
        &E::Fr::one(),
        &c,
    );

    // Compute final scalar
    let mut power_2_b = scalar_b.clone();
    let mut b_base = E::Fr::one();
    for x in transcript.iter() {
        b_base.mul_assign(&add!(
            E::Fr::one(),
            &(mul!(x.inverse().unwrap(), &power_2_b))
        ));
        power_2_b.mul_assign(&power_2_b.clone());
    }

    // Verify base inner product commitment
    let (com_a, _, com_t) = base_com;
    let a_base = vec![proof.gipa_proof.r_base.0.clone()];
    let t_base = vec![inner_product::multiexponentiation(&a_base, &vec![b_base])];
    let a = inner_product::pairing::<E>(&a_base, &vec![ck_a_final.clone()]) == com_a;
    let b = &t_base == &com_t;
    if a && b {
        aid
    } else {
        E::Fqk::zero()
    }
}

fn gipa_with_ssm_verify_recursive_challenge_transcript<E: Engine, D: Digest>(
    com: (&E::Fqk, &E::Fr, &Vec<E::G1>),
    proof: &GIPAProofWithSSM<E, D>,
) -> ((E::Fqk, E::Fr, Vec<E::G1>), Vec<E::Fr>) {
    info!("gipa ssm verify recursive challenge transcript");
    let (com_0, com_1, com_2) = com.clone();
    let (mut com_a, mut com_b, mut com_t) = (*com_0, *com_1, com_2.clone());
    let mut r_transcript = Vec::new();
    for (com_1, com_2) in proof.r_commitment_steps.iter().rev() {
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

            let d = D::digest(&hash_input);
            let c = fr_from_u128::<E::Fr>(d.as_slice());

            if let Some(c_inv) = c.inverse() {
                // Optimization for multiexponentiation to rescale G2 elements with 128-bit challenge
                // Swap 'c' and 'c_inv' since can't control bit size of c_inv
                break 'challenge (c_inv, c);
            }
            counter_nonce += 1;
        };

        com_a = mul!(
            mul!(com_1.0.pow(c.into_repr()), &com_a),
            &com_2.0.pow(c_inv.into_repr())
        );
        com_b = mul!(mul!(mul!(com_1.1, &c), &com_b), &mul!(com_2.1, &c_inv));
        assert_eq!(com_1.2.len(), com_t.len());
        assert_eq!(com_2.2.len(), com_t.len());
        com_t = com_1
            .2
            .iter()
            .zip(com_t.iter())
            .zip(com_2.2.iter())
            .map(|((com_1_2, com_t), com_2_2)| {
                let a = mul!(*com_1_2, c.into_repr());
                let b = mul!(*com_2_2, c_inv.into_repr());
                add!(add!(a, &com_t), &b)
            })
            .collect();

        r_transcript.push(c);
    }
    r_transcript.reverse();
    ((com_a, com_b, com_t), r_transcript)
}
