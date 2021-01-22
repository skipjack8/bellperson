use crossbeam_channel::bounded;
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use log::*;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    accumulator::PairingTuple,
    commit, inner_product,
    prove::{fr_from_u128, polynomial_evaluation_product_form_from_transcript},
    structured_scalar_power, AggregateProof, GipaMIPP, GipaTIPP, KZGOpening, MIPPProof, TIPPProof,
    VerifierSRS,
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
    // TODO: move that to seprate function or macro
    let mut counter_nonce: usize = 0;
    let r = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);

        bincode::serialize_into(&mut hash_input, &proof.com_ab.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.com_ab.1).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.com_c.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &proof.com_c.1).expect("vec");

        if let Some(r) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break r;
        };
        counter_nonce += 1;
    };
    println!(" ---------------------------- VERIFY Challenge r: {}", r);

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
            let tuple = verify_tipp::<E>(
                ip_verifier_srs,
                &proof.com_ab,
                &proof.ip_ab,
                &proof.proof_ab,
                &r, // we give the extra r as it's not part of the proof itself - it is simply used on top for the groth16 aggregation
            );
            println!(
                "TIPA AB took {} ms -> VERIFIED ? {}",
                now.elapsed().as_millis(),
                tuple.verify()
            );
            tipa_ab.send(tuple).unwrap();
        });

        // 2.Check TIPA proof c
        let tipa_c = send_tuple.clone();
        s.spawn(move |_| {
            let now = Instant::now();
            let tuple = verify_mipp::<E>(
                ip_verifier_srs,
                // com_c = C * v
                &proof.com_c,
                // agg_c = C ^ r
                &proof.agg_c,
                &proof.proof_c,
            );
            println!(
                "TIPA proof c took {} ms -> VERIFIED ? {}",
                now.elapsed().as_millis(),
                tuple.verify()
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
                // e(c^r vector form, h^delta)
                // let agg_c = inner_product::multiexponentiation::<E::G1Affine>(&c, r_vec)
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

/// verify_tipp returns a pairing equation to check the tipp proof. commAB is
/// the commitment output of A and B, Z is the aggregated value: $A^r * B$ as
/// described in the paper. $r$ is the randomness used to produce a random
/// linear combination of A and B.
fn verify_tipp<E: Engine>(
    v_srs: &VerifierSRS<E>,
    comAB: &commit::Output<E>,
    Z: &E::Fqk,
    proof: &TIPPProof<E>,
    r_shift: &E::Fr,
) -> PairingTuple<E> {
    info!("verify with srs shift");
    let now = Instant::now();
    // (T,U), Z, and all challenges
    let (final_ab, final_z, mut challenges, mut challenges_inv) =
        gipa_verify_tipp(comAB, Z, &proof.gipa);
    println!("\tTIPP.verify challenges: {:?}", challenges.clone());
    println!("TIPA AB: gipa verify tipp {}ms", now.elapsed().as_millis());

    // we reverse the order so the KZG polynomial have them in the expected
    // order to construct them in logn time.
    challenges.reverse();
    challenges_inv.reverse();
    // Verify commitment keys wellformed
    let fvkey = proof.gipa.final_vkey;
    let fwkey = proof.gipa.final_wkey;
    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &challenges.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &fvkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &fvkey.1).expect("vec");
        bincode::serialize_into(&mut hash_input, &fwkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &fwkey.1).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    println!("\t VERIFY TIPP KZG -> {:?}", &c);

    let now = Instant::now();
    // Section 3.4. step 5 check the opening proof for v
    let mut vtuple = verify_kzg_opening_g2(
        v_srs,
        &fvkey,
        &proof.vkey_opening,
        &challenges_inv,
        &r_shift.inverse().unwrap(),
        &c,
        true,
    );
    // Section 3.4 step 6 check the opening proof for w
    let wtuple = verify_kzg_opening_g1(
        v_srs,
        &fwkey,
        &proof.wkey_opening,
        &challenges,
        &E::Fr::one(),
        &c,
    );
    println!(
        "TIPA AB verify KZG: {}ms VERIFIED ? v {} w {}",
        now.elapsed().as_millis(),
        vtuple.verify(),
        wtuple.verify()
    );
    let now = Instant::now();

    // Section 3.4 step 2
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut out = E::Fqk::one();
    let (T, U) = final_ab;
    // final_Z = e(A,B)
    left.push(proof.gipa.final_A);
    right.push(proof.gipa.final_B);
    out.mul_assign(&final_z);
    let check = PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&left, &right),
        out,
    );
    println!(
        "TIPA AB inner product Z ONLY: {}ms -> VERIFIED ? {}",
        now.elapsed().as_millis(),
        check.verify()
    );

    //  final_AB.0 = T = e(A,v1)e(w1,B)
    left.push(proof.gipa.final_A.clone());
    right.push(fvkey.0.clone());
    let mut check = PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&left, &right),
        E::Fqk::one(),
    );
    left.push(fwkey.0.clone());
    right.push(proof.gipa.final_B.clone());
    out.mul_assign(&T);
    let check2 = PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&left, &right),
        out,
    );
    check.merge(&check2);
    println!(
        "TIPA AB inner product T ONLY: {}ms -> VERIFIED ? {}",
        now.elapsed().as_millis(),
        check.verify()
    );

    // final_AB.1 = U = e(A,v2)e(w2,B)
    left.push(proof.gipa.final_A.clone());
    right.push(fvkey.1.clone());
    left.push(fwkey.1.clone());
    right.push(proof.gipa.final_B.clone());
    out.mul_assign(&U);

    // TODO check if doing one big miller loop is faster than doing the
    // three in parallels and combine
    let check = PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&left, &right),
        out,
    );
    println!(
        "TIPA AB inner product: {}ms -> VERIFIED ? {}",
        now.elapsed().as_millis(),
        check.verify()
    );

    let now = Instant::now();
    vtuple.merge(&wtuple);
    vtuple.merge(&check);
    println!("TIPA AB merge : {}ms", now.elapsed().as_millis());
    vtuple
}

/// gipa_verify_tipp recurse on the proof and statement and produces the final
/// values to be checked by TIPP verifier, namely:
/// (T, U), Z, challenges, challenges_inv
/// T,U are the final commitment values of A and B and Z the final product
/// between A and B. Challenges are returned in inverse order as well to avoid
/// repeating the operation multiple times later on.
fn gipa_verify_tipp<E: Engine>(
    comm_AB: &commit::Output<E>,
    Z: &E::Fqk,
    proof: &GipaTIPP<E>,
) -> (commit::Output<E>, E::Fqk, Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa verify TIPP");

    let now = Instant::now();

    let mut challenges = Vec::new();
    let mut challenges_inv = Vec::new();

    let default_transcript = E::Fr::zero();

    // We first generate all challenges as this is the only consecutive process
    // that can not be parallelized then we scale the commitments in a
    // parallelized way
    for (comms_ab, z_comm) in proof.comms.iter().zip(proof.z_vec.iter()) {
        let ((T_l, U_l), (T_r, U_r)) = comms_ab;
        let (Z_l, Z_r) = z_comm;
        // Fiat-Shamir challenge
        // TODO use same function as in proving
        let mut counter_nonce: usize = 0;
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
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    println!(
        "TIPA AB: challenge gen took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    // paper names the output of the pair commitment T and U in TIPP
    let (mut T, mut U) = comm_AB.clone();
    let mut Z = Z.clone();

    // we first multiply each entry of the Z U and L vectors by the respective
    // challenges independently - step 3.4.1 (b) of paper.
    let prep: Vec<(_, _, _, _, _, _)> = proof
        .comms
        .par_iter()
        .zip(proof.z_vec.par_iter())
        .zip(challenges.par_iter())
        .zip(challenges_inv.par_iter())
        .map(|((((C_l, C_r), (Z_l, Z_r)), c), c_inv)| {
            let (T_l, U_l) = C_l;
            let (T_r, U_r) = C_r;
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();
            (
                T_l.pow(c_repr),
                T_r.pow(c_inv_repr),
                U_l.pow(c_repr),
                U_r.pow(c_inv_repr),
                Z_l.pow(c_repr),
                Z_r.pow(c_inv_repr),
            )
        })
        .collect();
    println!("TIPA AB: prep took {}ms", now.elapsed().as_millis());

    let now = Instant::now();

    for (T_l_c, T_r_cinv, U_l_c, U_l_cinv, Z_l_c, Z_l_cinv) in prep.iter() {
        // T = T_l^x . T . T_r^{x^-1}
        T.mul_assign(T_l_c);
        T.mul_assign(T_r_cinv);

        // U = U_l^x . U . U_r^{x-1}
        U.mul_assign(U_l_c);
        U.mul_assign(U_l_cinv);

        // Z = Z_l^x . Z . Z_r^{x^-1}
        Z.mul_assign(Z_l_c);
        Z.mul_assign(Z_l_cinv);
    }
    println!("TIPA AB: recursive took {}ms", now.elapsed().as_millis());
    let now = Instant::now();
    println!(
        "TIPA AB: reversed challenges took {}ms",
        now.elapsed().as_millis()
    );
    ((T, U), Z, challenges, challenges_inv)
}

/// verify_kzg_opening_g2 takes a KZG opening, the final commitment key, SRS and
/// any shift (in TIPP we shift the v commitment by r^-1) and returns a pairing
/// tuple to check if the opening is correct or not.
/// TODO potential optimization to avoid doing too many operation in Fqk
pub fn verify_kzg_opening_g2<E: Engine>(
    v_srs: &VerifierSRS<E>,
    final_vkey: &(E::G2Affine, E::G2Affine),
    vkey_opening: &KZGOpening<E::G2Affine>,
    challenges: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
    debug: bool,
) -> PairingTuple<E> {
    // f_v(z)
    let vpoly_eval_z =
        polynomial_evaluation_product_form_from_transcript(challenges, kzg_challenge, r_shift);

    // verify first part of opening - v1
    // e(g, v1 h^{-af_v(z)})
    let p1 = E::miller_loop(&[(
        &v_srs.g.into_affine().prepare(),
        // in additive notation: final_vkey = uH,
        // uH - f_v(z)H = (u - f_v)H --> v1h^{-af_v(z)}
        &sub!(
            final_vkey.0.into_projective(),
            &mul!(v_srs.h_alpha, vpoly_eval_z)
        )
        .into_affine()
        .prepare(),
    )]);
    // e(g^{a - z}, opening_1) ==> (aG) - (zG)
    let p2 = E::miller_loop(&[(
        &sub!(v_srs.g_alpha, &mul!(v_srs.g, kzg_challenge.clone()))
            .into_affine()
            .prepare(),
        &vkey_opening.0.prepare(),
    )]);
    // inverse so p1^-1 * p2 == 1
    let ip1 = p1.inverse().unwrap();
    if debug {
        let mut t1 = PairingTuple::<E>::from_miller(ip1);
        let t2 = PairingTuple::<E>::from_miller(p2);
        t1.merge(&t2);
        println!("@@@@@ Debug KZG t.valid() ? {}", t1.verify());
    }

    // verify second part of opening - v2 - similar but changing secret exponent
    // e(g, v2 h^{-bf_v(z)})
    let q1 = E::miller_loop(&[(
        &v_srs.g.into_affine().prepare(),
        // in additive notation: final_vkey = uH,
        // uH - f_v(z)H = (u - f_v)H --> v1h^{-f_v(z)}
        &sub!(
            final_vkey.1.into_projective(),
            &mul!(v_srs.h_beta, vpoly_eval_z)
        )
        .into_affine()
        .prepare(),
    )]);
    // e(g^{b - z}, opening_1)
    let q2 = E::miller_loop(&[(
        &sub!(v_srs.g_beta, &mul!(v_srs.g, kzg_challenge.clone()))
            .into_affine()
            .prepare(),
        &vkey_opening.1.prepare(),
    )]);

    let iq1 = q1.inverse().unwrap();
    // this pair should be one when multiplied
    PairingTuple::from_miller(mul!(mul!(iq1, &q2), &mul!(ip1, &p2)))
}

/// Similar to verify_kzg_opening_g2 but for g1.
pub fn verify_kzg_opening_g1<E: Engine>(
    v_srs: &VerifierSRS<E>,
    final_wkey: &(E::G1Affine, E::G1Affine),
    wkey_opening: &KZGOpening<E::G1Affine>,
    challenges: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> PairingTuple<E> {
    let wkey_poly_eval =
        polynomial_evaluation_product_form_from_transcript(challenges, kzg_challenge, r_shift);

    // first check on w1
    // let K = g^{a^{n+1}}
    // e(w1 K^{-f_w(z)},h)
    let p1 = E::miller_loop(&[(
        &sub!(
            final_wkey.0.into_projective(),
            &mul!(v_srs.g_alpha_n, wkey_poly_eval)
        )
        .into_affine()
        .prepare(),
        &v_srs.h.into_affine().prepare(),
    )]);
    // e(opening, h^{a - z})
    let p2 = E::miller_loop(&[(
        &wkey_opening.0.prepare(),
        &sub!(v_srs.h_alpha, &mul!(v_srs.h, *kzg_challenge))
            .into_affine()
            .prepare(),
    )]);
    let ip1 = p1.inverse().unwrap();
    // then do second check
    // let K = g^{b^{n+1}}
    // e(w2 K^{-f_w(z)},h)
    let q1 = E::miller_loop(&[(
        &sub!(
            final_wkey.1.into_projective(),
            &mul!(v_srs.g_beta_n, wkey_poly_eval)
        )
        .into_affine()
        .prepare(),
        &v_srs.h.into_affine().prepare(),
    )]);
    // e(opening, h^{b - z})
    let q2 = E::miller_loop(&[(
        &wkey_opening.1.prepare(),
        &sub!(v_srs.h_beta, &mul!(v_srs.h, *kzg_challenge))
            .into_affine()
            .prepare(),
    )]);
    let iq1 = q1.inverse().unwrap();

    PairingTuple::from_miller(mul!(mul!(iq1, &q2), &mul!(ip1, &p2)))
}

fn verify_mipp<E: Engine>(
    v_srs: &VerifierSRS<E>,
    com_C: &commit::Output<E>, // original (T,U) = CM(v1,v2,C) - is rescaled in gipa verify
    agg_C: &E::G1,             // original Z = C^r - is rescaled in gipa verify
    proof: &MIPPProof<E>,
) -> PairingTuple<E> {
    info!("verify with structured scalar message");
    let now = Instant::now();
    let (com_TU, com_Z, mut challenges, mut challenges_inv) =
        gipa_verify_mipp(com_C, agg_C, &proof.gipa);

    println!(
        "TIPA AB: gipa mipp verification took {}ms",
        now.elapsed().as_millis()
    );
    let now = Instant::now();

    let final_vkey = proof.gipa.final_vkey;
    // reverse the challenges so KZG polynomial is constructed correctly
    challenges.reverse();
    challenges_inv.reverse();

    // KZG challenge point
    let mut counter_nonce: usize = 0;
    let c = loop {
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(&counter_nonce.to_be_bytes()[..]);
        bincode::serialize_into(&mut hash_input, &challenges.first().unwrap()).expect("vec");
        bincode::serialize_into(&mut hash_input, &final_vkey.0).expect("vec");
        bincode::serialize_into(&mut hash_input, &final_vkey.1).expect("vec");

        if let Some(c) = E::Fr::from_random_bytes(
            &Sha256::digest(&hash_input).as_slice()
                [..std::mem::size_of::<<E::Fr as PrimeField>::Repr>()],
        ) {
            break c;
        };
        counter_nonce += 1;
    };

    println!(" +++ MIPP - verify: challenges {:?}", challenges);
    println!(" +++ MIPP - verify: c {:?}", c);

    println!(
        "TIPA AB: mipp verification challenge took {}ms",
        now.elapsed().as_millis()
    );
    let now = Instant::now();

    // final rescaled T U Z
    let (T, U) = com_TU;

    // final c from proof
    let final_C = proof.gipa.final_C.clone();
    let final_r = proof.gipa.final_r.clone();

    // Verify base inner product commitment
    // Z ==  c ^ r
    let now = Instant::now();
    let final_Z = inner_product::multiexponentiation::<E::G1Affine>(&[final_C], &[final_r]);
    let b = final_Z == com_Z;
    println!(
        "MIPP: check Z took {}ms -> valid Z ? {}",
        now.elapsed().as_millis(),
        b
    );
    // only check that doesn't require pairing so we can give a tuple that will
    // render the equation wrong in case it's false
    if !b {
        return PairingTuple::new_invalid();
    }

    let now = Instant::now();
    // Check commitment key corectness
    let mut vtuple = verify_kzg_opening_g2(
        v_srs,
        &final_vkey,
        &proof.vkey_opening,
        &challenges_inv,
        &E::Fr::one(),
        &c,
        false,
    );
    println!(
        "MIPP: check KZG took {}ms -> verified ? {}",
        now.elapsed().as_millis(),
        vtuple.verify()
    );
    let now = Instant::now();

    // Check commiment correctness 4.2.2
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut out = E::Fqk::one();
    // T = e(C,v1)
    left.push(final_C.clone());
    right.push(final_vkey.0);
    out.mul_assign(&T);
    {
        let miller_out = inner_product::pairing_miller_affine::<E>(&left, &right);
        let pair = PairingTuple::<E>::from_pair(miller_out, out);
        println!(
            " ------------MIPP - verify: vssm check took {}ms --> T CORRECT? {}",
            now.elapsed().as_millis(),
            pair.verify()
        );
        left.clear();
        right.clear();
        out = E::Fqk::one();
        // U = e(A,v2)
        left.push(final_C.clone());
        right.push(final_vkey.1);
        out.mul_assign(&U);
        let miller_out = inner_product::pairing_miller_affine::<E>(&left, &right);
        let pair = PairingTuple::<E>::from_pair(miller_out, out);
        println!(
            "-------------MIPP - verify: check  U=e(A,v2) took {}ms -> verified {}",
            now.elapsed().as_millis(),
            pair.verify()
        );
    }
    // U = e(A,v2)
    left.push(final_C.clone());
    right.push(final_vkey.1);
    out.mul_assign(&U);

    let miller_out = inner_product::pairing_miller_affine::<E>(&left, &right);
    let pair = PairingTuple::from_pair(miller_out, out);
    println!(
        "MIPP - verify: check  U=e(A,v2) took {}ms -> verified {}",
        now.elapsed().as_millis(),
        pair.verify()
    );

    let now = Instant::now();
    vtuple.merge(&pair);
    println!(
        " ---- MIPP: final merge took {}ms -- end verify? {}",
        now.elapsed().as_millis(),
        vtuple.verify()
    );
    vtuple
}

/// gipa_verify_mipp returns the final reconstructed Z T U values, as described
/// in section 4.2.1 as well as all challenges generated.
fn gipa_verify_mipp<E: Engine>(
    com_C: &commit::Output<E>,
    Z: &E::G1,
    proof: &GipaMIPP<E>,
) -> (commit::Output<E>, E::G1, Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa ssm verify recursive challenge challenges");
    let mut challenges = Vec::new();
    let mut challenges_inv = Vec::new();

    for ((TU_l, TU_r), (Z_l, Z_r)) in proof.comms.iter().zip(proof.z_vec.iter()) {
        // Fiat-Shamir challenge
        // TODO use same code for prover and verifier
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
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    let (mut comm_T, mut comm_U) = com_C.clone();
    let mut Z = Z.clone();

    let now = Instant::now();

    // Prepare the final commitment section 4.2. - steps 1.b
    let prep: Vec<(_, _, _, _, _, _)> = proof
        .comms
        .par_iter()
        .zip(proof.z_vec.par_iter())
        .zip(challenges.par_iter())
        .zip(challenges_inv.par_iter())
        .map(|((((C_l, C_r), (Z_l, Z_r)), c), c_inv)| {
            let (T_l, U_l) = C_l;
            let (T_r, U_r) = C_r;
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();

            // Z_l^x
            let mut Z_l = Z_l.clone();
            Z_l.mul_assign(c_repr);
            // Z_r^{x^{-1}}
            let mut Z_r = Z_r.clone();
            Z_r.mul_assign(c_inv_repr);

            // U_r^x  , U_l^x^-1
            (
                T_l.pow(c_repr),
                U_l.pow(c_repr),
                T_r.pow(c_inv_repr),
                U_r.pow(c_inv_repr),
                Z_l,
                Z_r,
            )
        })
        .collect();

    println!(
        "MIPP C: GIPA preparation took {}ms ({})",
        now.elapsed().as_millis(),
        prep.len()
    );

    for (T_l_c, U_l_c, T_r_cinv, U_r_cinv, Z_l_c, Z_r_cinv) in prep.iter() {
        comm_T.mul_assign(T_l_c);
        comm_T.mul_assign(T_r_cinv);
        comm_U.mul_assign(U_l_c);
        comm_U.mul_assign(U_r_cinv);

        Z.add_assign(Z_l_c);
        Z.add_assign(Z_r_cinv);
    }

    ((comm_T, comm_U), Z, challenges, challenges_inv)
}
