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
use crossbeam_channel::bounded;
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::CurveProjective;
use log::*;

pub fn verify_aggregate_proof<E: Engine + std::fmt::Debug, D: Digest + Sync>(
    ip_verifier_srs: &VerifierSRS<E>,
    pvk: &PreparedVerifyingKey<E>,
    public_inputs: &[Vec<E::Fr>],
    proof: &AggregateProof<E, D>,
) -> bool {
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
            &D::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break r;
        };
        counter_nonce += 1;
    };

    assert_eq!(pvk.ic.len(), public_inputs[0].len() + 1);

    let (valid_send, valid_rcv) = bounded(1);
    rayon::scope(move |s| {
        // channel used to aggregate all pairing tuples
        let (send_tuple, rcv_tuple) = bounded(10);

        // 1.Check TIPA proof ab
        let tipa_ab = send_tuple.clone();
        s.spawn(move |_| {
            let tuple = verify_with_srs_shift::<E, D>(
                ip_verifier_srs,
                (&proof.com_a, &proof.com_b, &proof.ip_ab),
                &proof.tipa_proof_ab,
                &r,
            );
            tipa_ab.send(tuple).unwrap();
        });

        // 2.Check TIPA proof c
        let tipa_c = send_tuple.clone();
        s.spawn(move |_| {
            let tuple = verify_with_structured_scalar_message::<E, D>(
                ip_verifier_srs,
                (&proof.com_c, &proof.agg_c),
                &r,
                &proof.tipa_proof_c,
            );
            tipa_c.send(tuple).unwrap();
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
            r_vec_sender
                .send(structured_scalar_power(public_inputs.len(), &r))
                .unwrap();
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

    res
}

fn verify_with_srs_shift<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::Fqk, &E::Fqk),
    proof: &PairingInnerProductABProof<E, D>,
    r_shift: &E::Fr,
) -> PairingTuple<E> {
    info!("verify with srs shift");
    let (base_com, transcript, transcript_inverse) =
        gipa_verify_recursive_challenge_transcript(com, &proof.gipa_proof);

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
            &D::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    let aid = verify_commitment_key_g2_kzg_opening(
        v_srs,
        &ck_a_final,
        &ck_a_proof,
        &transcript_inverse,
        &r_shift.inverse().unwrap(),
        &c,
    );
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
    let a_base = [proof.gipa_proof.r_base.0.clone()];
    let b_base = [proof.gipa_proof.r_base.1.clone()];
    // LMC::verify - pairing inner product<E>
    let mut a = PairingTuple::from_pair(
        inner_product::pairing_miller::<E>(&a_base, &[ck_a_final.clone()]),
        com_a,
    );
    // RMC::verify - afgho commitment G1
    let b = PairingTuple::from_pair(
        inner_product::pairing_miller::<E>(&[ck_b_final.clone()], &b_base),
        com_b,
    );
    // IPC::verify - identity commitment<Fqk, Fr>
    let t_base =
        PairingTuple::from_pair(inner_product::pairing_miller::<E>(&a_base, &b_base), com_t);

    a.merge(&b);
    a.merge(&t_base);
    a.merge(&aid);
    a.merge(&bid);
    a
}

fn gipa_verify_recursive_challenge_transcript<E: Engine, D: Digest>(
    com: (&E::Fqk, &E::Fqk, &E::Fqk),
    proof: &GIPAProof<E, D>,
) -> ((E::Fqk, E::Fqk, E::Fqk), Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa verify recursive challenge transcript");
    let (com_0, com_1, com_2) = com.clone();
    let (mut com_a, mut com_b, mut com_t) = (*com_0, *com_1, *com_2);
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

        com_t = {
            let x_c = com_1.2.pow(c.into_repr());
            let z_c_inv = com_2.2.pow(c_inv.into_repr());
            mul!(mul!(x_c, &com_t), &z_c_inv)
        };

        r_transcript.push(c);
        r_transcript_inverse.push(c_inv);
    }
    r_transcript.reverse();
    r_transcript_inverse.reverse();
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

fn verify_with_structured_scalar_message<E: Engine, D: Digest>(
    v_srs: &VerifierSRS<E>,
    com: (&E::Fqk, &E::G1),
    scalar_b: &E::Fr,
    proof: &MultiExpInnerProductCProof<E, D>,
) -> PairingTuple<E> {
    info!("verify with structured scalar message");
    let (base_com, transcript, transcript_inverse) =
        gipa_with_ssm_verify_recursive_challenge_transcript((com.0, com.1), &proof.gipa_proof);

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
            &D::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    // Check commitment key
    let mut aid = verify_commitment_key_g2_kzg_opening(
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
    let (com_a, com_t) = base_com;
    let a_base = [proof.gipa_proof.r_base.0.clone().into_affine()];
    let t_base = inner_product::multiexponentiation(&a_base, &[b_base]);
    let a = PairingTuple::from_pair(
        inner_product::pairing_miller_affine::<E>(&a_base, &[ck_a_final.into_affine()]),
        com_a.clone(),
    );
    let b = t_base == com_t;

    // only check that doesn't require pairing so we can give a tuple that will
    // render the equation wrong in case it's false
    if b {
        aid.merge(&a);
        aid
    } else {
        PairingTuple::new_invalid()
    }
}

fn gipa_with_ssm_verify_recursive_challenge_transcript<E: Engine, D: Digest>(
    com: (&E::Fqk, &E::G1),
    proof: &GIPAProofWithSSM<E, D>,
) -> ((E::Fqk, E::G1), Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa ssm verify recursive challenge transcript");
    let (com_0, com_1) = com.clone();
    let (mut com_a, mut com_t) = (*com_0, *com_1);
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

        com_t = {
            let a = mul!(com_1.1, c.into_repr());
            let b = mul!(com_2.1, c_inv.into_repr());
            add!(add!(a, &com_t), &b)
        };

        r_transcript.push(c);
        r_transcript_inverse.push(c_inv);
    }

    r_transcript.reverse();
    r_transcript_inverse.reverse();
    ((com_a, com_t), r_transcript, r_transcript_inverse)
}
