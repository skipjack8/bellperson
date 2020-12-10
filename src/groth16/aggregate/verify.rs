use crossbeam_channel::bounded;
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::CurveProjective;
use log::*;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    accumulator::PairingTuple,
    inner_product,
    prove::{fr_from_u128, polynomial_evaluation_product_form_from_transcript},
    structured_scalar_power, AggregateProof, GIPAProof, GIPAProofWithSSM,
    MultiExpInnerProductCProof, PairingInnerProductABProof, VerifierSRS,
};
use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::{
    multiscalar::{par_multiscalar, MultiscalarPrecomp, ScalarList},
    PreparedVerifyingKey,
};
use crate::SynthesisError;

use std::time::Instant;
pub fn verify_aggregate_proof<E: Engine + std::fmt::Debug>(
    ip_verifier_srs: &VerifierSRS<E>,
    pvk: &PreparedVerifyingKey<E>,
    public_inputs: &[Vec<E::Fr>],
    proof: &AggregateProof<E>,
) -> Result<bool, SynthesisError> {
    info!("verify_aggregate_proof");

    // Random linear combination of proofs
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);

        bincode::serialize_into(&mut hash_input, &proof.com_a).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.com_b).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.com_c).expect("vec");

        if let Some(r) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break r;
        };
        counter_nonce += 1;
    };

    for pub_input in public_inputs {
        if (pub_input.len() + 1) != pvk.ic.len() {
            return Err(SynthesisError::MalformedVerifyingKey);
        }
    }

    let (valid_send, valid_rcv) = bounded(1);
    rayon::scope(move |s| {
        // channel used to aggregate all pairing tuples
        let (send_tuple, rcv_tuple) = bounded(10);

        // 1.Check TIPA proof ab
        let tipa_ab = send_tuple.clone();
        s.spawn(move |_| {
            let now = Instant::now();
            let tuple = verify_with_srs_shift::<E>(
                ip_verifier_srs,
                (&proof.com_a, &proof.com_b, &proof.ip_ab),
                &proof.tipa_proof_ab,
                &r,
            );
            println!("TIPA AB took {} ms", now.elapsed().as_millis());
            tipa_ab.send(tuple).unwrap();
        });

        // 2.Check TIPA proof c
        let tipa_c = send_tuple.clone();
        s.spawn(move |_| {
            let now = Instant::now();
            let tuple = verify_with_structured_scalar_message::<E>(
                ip_verifier_srs,
                (&proof.com_c, &proof.agg_c),
                &r,
                &proof.tipa_proof_c,
            );
            tipa_c.send(tuple).unwrap();
            println!("TIPA proof c took {} ms", now.elapsed().as_millis());
        });

        // Check aggregate pairing product equation
        info!("checking aggregate pairing");
        let mut r_sum = r.pow(&[public_inputs.len() as u64]);
        r_sum.sub_assign(&E::Fr::one());
        let b = sub!(r, &E::Fr::one()).inverse().unwrap();
        r_sum.mul_assign(&b);

        // 3. Compute left part of the final pairing equation
        let p1 = send_tuple.clone();
        s.spawn(move |_| {
            let mut alpha_g1_r_sum = pvk.alpha_g1;
            alpha_g1_r_sum.mul_assign(r_sum);
            let tuple = E::miller_loop(&[(&alpha_g1_r_sum.into_affine().prepare(), &pvk.beta_g2)]);

            p1.send(PairingTuple::from_miller(tuple)).unwrap();
        });

        // 4. Compute right part of the final pairing equation
        let p3 = send_tuple.clone();
        s.spawn(move |_| {
            let tuple = PairingTuple::from_miller(E::miller_loop(&[(
                &proof.agg_c.into_affine().prepare(),
                &pvk.delta_g2,
            )]));
            p3.send(tuple).unwrap();
        });

        let (r_vec_sender, r_vec_receiver) = bounded(1);
        s.spawn(move |_| {
            let now = Instant::now();
            r_vec_sender
                .send(structured_scalar_power(public_inputs.len(), &r))
                .unwrap();
            let elapsed = now.elapsed().as_millis();
            println!("generation of r vector: {}ms", elapsed);
        });

        // 5. compute the middle part of the final pairing equation, the one
        //    with the public inputs
        //let p2 = send_tuple.clone();
        s.spawn(move |_| {
            // We want to compute MUL(i:0 -> l) S_i ^ (SUM(j:0 -> n) ai,j * r^j)
            // this table keeps tracks of incremental computation of each i-th
            // exponent to later multiply with S_i
            // The index of the table is i, which is an index of the public
            // input element
            // We incrementally build the r vector and the table
            // NOTE: in this version it's not r^2j but simply r^j

            let l = public_inputs[0].len();
            let mut g_ic = pvk.ic_projective[0];
            g_ic.mul_assign(r_sum);

            let powers = r_vec_receiver.recv().unwrap();

            let now = Instant::now();
            // now we do the multi exponentiation
            let getter = |i: usize| -> <E::Fr as PrimeField>::Repr {
                // i denotes the column of the public input, and j denotes which public input
                let mut c = public_inputs[0][i];
                for j in 1..public_inputs.len() {
                    let mut ai = public_inputs[j][i];
                    ai.mul_assign(&powers[j]);
                    c.add_assign(&ai);
                }
                c.into_repr()
            };

            let totsi = par_multiscalar::<_, E::G1Affine>(
                &ScalarList::Getter(getter, l),
                &pvk.multiscalar.at_point(1),
                std::mem::size_of::<<E::Fr as PrimeField>::Repr>() * 8,
            );

            g_ic.add_assign(&totsi);

            let tuple = PairingTuple::from_miller(E::miller_loop(&[(
                &g_ic.into_affine().prepare(),
                &pvk.gamma_g2,
            )]));
            let elapsed = now.elapsed().as_millis();
            println!("table generation: {}ms", elapsed);

            send_tuple.send(tuple).unwrap();
        });

        s.spawn(move |_| {
            // final value ip_ab is what we want to compare in the groth16
            // aggregated equation A * B
            let mut acc = PairingTuple::from_pair(E::Fqk::one(), proof.ip_ab.clone());
            while let Ok(tuple) = rcv_tuple.recv() {
                acc.merge(&tuple);
            }
            valid_send.send(acc.verify()).unwrap();
        });
    });

    let res = valid_rcv.recv().unwrap();
    info!("aggregate verify done");

    Ok(res)
}

fn verify_with_srs_shift<E: Engine>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::Fqk, &E::Fqk),
    proof: &PairingInnerProductABProof<E>,
    r_shift: &E::Fr,
) -> PairingTuple<E> {
    info!("verify with srs shift");
    let now = Instant::now();
    let (base_com, transcript, transcript_inverse) =
        gipa_verify_recursive_challenge_transcript(com, &proof.gipa_proof);
    println!("TIPA AB: verify recursive {}ms", now.elapsed().as_millis());

    // Verify commitment keys wellformed
    let (ck_a_final, ck_b_final) = &proof.final_ck;
    let (ck_a_proof, ck_b_proof) = &proof.final_ck_proof;

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

    let now = Instant::now();

    let mut aid = PairingTuple::new();
    let mut bid = PairingTuple::new();
    let mut a = PairingTuple::new();
    let mut b = PairingTuple::new();
    let mut t_base = PairingTuple::new();

    rayon::scope(|s| {
        let aid = &mut aid;
        s.spawn(move |_| {
            *aid = verify_commitment_key_g2_kzg_opening(
                v_srs,
                &ck_a_final,
                &ck_a_proof,
                &transcript_inverse,
                &r_shift.inverse().unwrap(),
                &c,
            );
        });

        let bid = &mut bid;
        s.spawn(move |_| {
            *bid = verify_commitment_key_g1_kzg_opening(
                v_srs,
                &ck_b_final,
                &ck_b_proof,
                &transcript,
                &E::Fr::one(),
                &c,
            );
        });

        // Verify base inner product commitment
        let (com_a, com_b, com_t) = base_com;
        let a_base = [proof.gipa_proof.r_base.0.clone()];
        let b_base = [proof.gipa_proof.r_base.1.clone()];

        let a = &mut a;
        s.spawn(move |_| {
            // LMC::verify - pairing inner product<E>
            *a = PairingTuple::from_pair(
                inner_product::pairing_miller::<E>(&a_base, &[ck_a_final.clone()]),
                com_a,
            );
        });

        let b = &mut b;
        s.spawn(move |_| {
            // RMC::verify - afgho commitment G1
            *b = PairingTuple::from_pair(
                inner_product::pairing_miller::<E>(&[ck_b_final.clone()], &b_base),
                com_b,
            );
        });

        let t_base = &mut t_base;
        s.spawn(move |_| {
            // IPC::verify - identity commitment<Fqk, Fr>
            *t_base = PairingTuple::from_pair(
                inner_product::pairing_miller::<E>(&a_base, &b_base),
                com_t,
            );
        });
    });

    println!("TIPA AB inner product: {}ms", now.elapsed().as_millis());

    let now = Instant::now();
    a.merge(&b);
    a.merge(&t_base);
    a.merge(&aid);
    a.merge(&bid);
    println!("TIPA AB merge : {}ms", now.elapsed().as_millis());
    a
}

fn gipa_verify_recursive_challenge_transcript<E: Engine>(
    com: (&E::Fqk, &E::Fqk, &E::Fqk),
    proof: &GIPAProof<E>,
) -> ((E::Fqk, E::Fqk, E::Fqk), Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa verify recursive challenge transcript");

    let now = Instant::now();

    let mut r_transcript = Vec::new();
    let mut r_transcript_inverse = Vec::new();

    let default_transcript = E::Fr::zero();

    for (com_1, com_2) in proof.r_commitment_steps.iter().rev() {
        // Fiat-Shamir challenge
        let mut counter_nonce: usize = 0;

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

        r_transcript.push(c);
        r_transcript_inverse.push(c_inv);
    }

    println!(
        "TIPA AB: challenge gen took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    let mut com_a = com.0.clone();
    let mut com_b = com.1.clone();
    let mut com_t = com.2.clone();

    let prep: Vec<(_, _, _, _, _, _)> = proof
        .r_commitment_steps
        .par_iter()
        .rev()
        .zip(r_transcript.par_iter())
        .zip(r_transcript_inverse.par_iter())
        .map(|(((com_1, com_2), c), c_inv)| {
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();

            (
                com_1.0.pow(c_repr),
                com_2.0.pow(c_inv_repr),
                com_1.1.pow(c_repr),
                com_2.1.pow(c_inv_repr),
                com_1.2.pow(c_repr),
                com_2.2.pow(c_inv_repr),
            )
        })
        .collect();
    println!("TIPA AB: prep took {}ms", now.elapsed().as_millis());

    let now = Instant::now();

    for (a_x_c, a_z_c_inv, b_x_c, b_z_c_inv, t_x_c, t_z_c_inv) in prep.iter() {
        com_a.mul_assign(a_x_c);
        com_a.mul_assign(a_z_c_inv);

        com_b.mul_assign(b_x_c);
        com_b.mul_assign(b_z_c_inv);

        com_t.mul_assign(t_x_c);
        com_t.mul_assign(t_z_c_inv);
    }
    println!("TIPA AB: recursive took {}ms", now.elapsed().as_millis());
    let now = Instant::now();
    r_transcript.reverse();
    r_transcript_inverse.reverse();

    println!(
        "TIPA AB: reversed transcript took {}ms",
        now.elapsed().as_millis()
    );
    ((com_a, com_b, com_t), r_transcript, r_transcript_inverse)
}

pub fn verify_commitment_key_g2_kzg_opening<E: Engine>(
    v_srs: &VerifierSRS<E>,
    ck_final: &E::G2,
    ck_opening: &E::G2,
    transcript: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> PairingTuple<E> {
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
    // this pair should be one when multiplied
    PairingTuple::from_miller(mul!(ip1, &p2))
}

pub fn verify_commitment_key_g1_kzg_opening<E: Engine>(
    v_srs: &VerifierSRS<E>,
    ck_final: &E::G1,
    ck_opening: &E::G1,
    transcript: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> PairingTuple<E> {
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
    PairingTuple::from_miller(mul!(ip1, &p2))
}

fn verify_with_structured_scalar_message<E: Engine>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::G1),
    scalar_b: &E::Fr,
    proof: &MultiExpInnerProductCProof<E>,
) -> PairingTuple<E> {
    info!("verify with structured scalar message");
    let now = Instant::now();
    let (base_com, transcript, transcript_inverse) =
        gipa_with_ssm_verify_recursive_challenge_transcript((com.0, com.1), &proof.gipa_proof);

    println!("TIPA AB: vssm gipa took {}ms", now.elapsed().as_millis());
    let now = Instant::now();

    let ck_a_final = &proof.final_ck;
    let ck_a_proof = &proof.final_ck_proof;

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

    println!(
        "TIPA AB: vssm challenge took {}ms",
        now.elapsed().as_millis()
    );
    let now = Instant::now();

    // Check commitment key
    let mut aid = verify_commitment_key_g2_kzg_opening(
        v_srs,
        &ck_a_final,
        &ck_a_proof,
        &transcript_inverse,
        &E::Fr::one(),
        &c,
    );

    let (com_a, com_t) = base_com;
    let a_base = [proof.gipa_proof.r_base.0.clone().into_affine()];

    let a = PairingTuple::from_pair(
        inner_product::pairing_miller_affine::<E>(&a_base, &[ck_a_final.into_affine()]),
        com_a.clone(),
    );

    println!("TIPA AB: vssm aid took {}ms", now.elapsed().as_millis());
    let now = Instant::now();

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
    let t_base = inner_product::multiexponentiation(&a_base, &[b_base]);
    let b = t_base == com_t;

    println!("TIPA AB: vssm b took {}ms", now.elapsed().as_millis());
    let now = Instant::now();

    // only check that doesn't require pairing so we can give a tuple that will
    // render the equation wrong in case it's false
    if !b {
        return PairingTuple::new_invalid();
    }

    aid.merge(&a);
    println!("TIPA AB: vssm merge took {}ms", now.elapsed().as_millis());
    aid
}

fn gipa_with_ssm_verify_recursive_challenge_transcript<E: Engine>(
    com: (&E::Fqk, &E::G1),
    proof: &GIPAProofWithSSM<E>,
) -> ((E::Fqk, E::G1), Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa ssm verify recursive challenge transcript");
    let mut r_transcript = Vec::new();
    let mut r_transcript_inverse = Vec::new();

    for (com_1, com_2) in proof.r_commitment_steps.iter().rev() {
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
        r_transcript.push(c);
        r_transcript_inverse.push(c_inv);
    }

    let mut com_a = com.0.clone();
    let mut com_t = com.1.clone();

    let now = Instant::now();

    let prep: Vec<(_, _, _, _)> = proof
        .r_commitment_steps
        .par_iter()
        .rev()
        .zip(r_transcript.par_iter())
        .zip(r_transcript_inverse.par_iter())
        .map(|(((com_1, com_2), c), c_inv)| {
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();

            let mut x = com_1.1;
            x.mul_assign(c_repr);
            let mut y = com_2.1;
            y.mul_assign(c_inv_repr);

            (com_1.0.pow(c_repr), com_2.0.pow(c_inv_repr), x, y)
        })
        .collect();

    println!(
        "TIPA AB: vssm prep took {}ms ({})",
        now.elapsed().as_millis(),
        prep.len()
    );

    for (a_x_c, a_z_c_inv, t_x_c, t_z_c_inv) in prep.iter() {
        com_a.mul_assign(a_x_c);
        com_a.mul_assign(a_z_c_inv);

        com_t.add_assign(t_x_c);
        com_t.add_assign(t_z_c_inv);
    }

    r_transcript.reverse();
    r_transcript_inverse.reverse();
    ((com_a, com_t), r_transcript, r_transcript_inverse)
}
