use crossbeam_channel::bounded;
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use log::*;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    accumulator::PairingTuple, commit, inner_product,
    prove::polynomial_evaluation_product_form_from_transcript, structured_scalar_power,
    AggregateProof, GipaMIPP, GipaTIPP, KZGOpening, MIPPProof, TIPPProof, VerifierSRS,
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
    let r = oracle!(
        &proof.com_ab.0,
        &proof.com_ab.1,
        &proof.com_c.0,
        &proof.com_c.1
    );

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
            println!("TIPP took {} ms", now.elapsed().as_millis());
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
            println!("MIPP proof took {} ms", now.elapsed().as_millis(),);
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
    comm_ab: &commit::Output<E>,
    z: &E::Fqk,
    proof: &TIPPProof<E>,
    r_shift: &E::Fr,
) -> PairingTuple<E> {
    info!("verify with srs shift");
    let now = Instant::now();
    // (T,U), Z, and all challenges
    let (final_ab, final_z, mut challenges, mut challenges_inv) =
        gipa_verify_tipp(comm_ab, z, &proof.gipa);
    println!(
        "TIPP verify: gipa verify tipp {}ms",
        now.elapsed().as_millis()
    );

    // we reverse the order so the KZG polynomial have them in the expected
    // order to construct them in logn time.
    challenges.reverse();
    challenges_inv.reverse();
    // Verify commitment keys wellformed
    let fvkey = proof.gipa.final_vkey;
    let fwkey = proof.gipa.final_wkey;
    // KZG challenge point
    let c = oracle!(
        &challenges.first().unwrap(),
        &fvkey.0,
        &fvkey.1,
        &fwkey.0,
        &fwkey.1
    );

    let (t, u) = final_ab;
    let now = Instant::now();
    par! {
        // Section 3.4. step 5 check the opening proof for v
        let vtuple = verify_kzg_opening_g2(
            v_srs,
            &fvkey,
            &proof.vkey_opening,
            &challenges_inv,
            &r_shift.inverse().unwrap(),
            &c,
        ),
        // Section 3.4 step 6 check the opening proof for w
        let wtuple = verify_kzg_opening_g1(
            v_srs,
            &fwkey,
            &proof.wkey_opening,
            &challenges,
            &E::Fr::one(),
            &c,
        ),
        // Section 3.4 step 2
        // final_z = e(A,B)
        let check_z = make_tuple(&proof.gipa.final_a, &proof.gipa.final_b, &final_z),
        //  final_aB.0 = T = e(A,v1)e(w1,B)
        let check_ab11 = make_tuple(&proof.gipa.final_a, &fvkey.0, &E::Fqk::one()),
        let check_ab12 = make_tuple(&fwkey.0, &proof.gipa.final_b, &t),
        let check_ab21 = make_tuple(&proof.gipa.final_a, &fvkey.1, &E::Fqk::one()),
        let check_ab22 = make_tuple(&fwkey.1, &proof.gipa.final_b, &u)
    };

    println!(
        "TIPP verify: parallel checks before merge: {}ms",
        now.elapsed().as_millis(),
    );

    let now = Instant::now();
    let mut acc = vtuple;
    acc.merge(&wtuple);
    acc.merge(&check_z);
    acc.merge(&check_ab11);
    acc.merge(&check_ab12);
    acc.merge(&check_ab21);
    acc.merge(&check_ab22);
    println!("TIPP verify: final merge {}ms", now.elapsed().as_millis());
    acc
}

/// gipa_verify_tipp recurse on the proof and statement and produces the final
/// values to be checked by TIPP verifier, namely:
/// (T, U), Z, challenges, challenges_inv
/// T,U are the final commitment values of A and B and Z the final product
/// between A and B. Challenges are returned in inverse order as well to avoid
/// repeating the operation multiple times later on.
fn gipa_verify_tipp<E: Engine>(
    comm_ab: &commit::Output<E>,
    z: &E::Fqk,
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
        let ((t_l, u_l), (t_r, u_r)) = comms_ab;
        let (z_l, z_r) = z_comm;
        // Fiat-Shamir challenge
        let transcript = challenges.last().unwrap_or(&default_transcript);
        let c_inv = oracle!(&transcript, &t_l, &u_l, &t_r, &u_r, &z_r, &z_l);
        let c = c_inv.inverse().unwrap();
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    println!(
        "TIPP verify: gipa challenge gen took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    // paper names the output of the pair commitment T and U in TIPP
    let (mut t, mut u) = comm_ab.clone();
    let mut z = z.clone();

    // we first multiply each entry of the Z U and L vectors by the respective
    // challenges independently - step 3.4.1 (b) of paper.
    let prep: Vec<(_, _, _, _, _, _)> = proof
        .comms
        .par_iter()
        .zip(proof.z_vec.par_iter())
        .zip(challenges.par_iter())
        .zip(challenges_inv.par_iter())
        .map(|((((c_l, c_r), (z_l, z_r)), c), c_inv)| {
            let (t_l, u_l) = c_l;
            let (t_r, u_r) = c_r;
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();
            (
                t_l.pow(c_repr),
                t_r.pow(c_inv_repr),
                u_l.pow(c_repr),
                u_r.pow(c_inv_repr),
                z_l.pow(c_repr),
                z_r.pow(c_inv_repr),
            )
        })
        .collect();
    println!(
        "TIPP verify: gipa prep took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();

    for (t_l_c, t_r_cinv, u_l_c, u_l_cinv, z_l_c, z_l_cinv) in prep.iter() {
        // T = t_l^x . T . t_r^{x^-1}
        t.mul_assign(t_l_c);
        t.mul_assign(t_r_cinv);

        // U = u_l^x . U . u_r^{x-1}
        u.mul_assign(u_l_c);
        u.mul_assign(u_l_cinv);

        // Z = z_l^x . Z . z_r^{x^-1}
        z.mul_assign(z_l_c);
        z.mul_assign(z_l_cinv);
    }
    println!(
        "TIPP verify: gipa accumulate step took {}ms",
        now.elapsed().as_millis()
    );
    ((t, u), z, challenges, challenges_inv)
}

/// verify_kzg_opening_g2 takes a KZG opening, the final commitment key, SRS and
/// any shift (in TIPP we shift the v commitment by r^-1) and returns a pairing
/// tuple to check if the opening is correct or not.
/// TODO optimization to do all in one miller loop maybe
pub fn verify_kzg_opening_g2<E: Engine>(
    v_srs: &VerifierSRS<E>,
    final_vkey: &(E::G2Affine, E::G2Affine),
    vkey_opening: &KZGOpening<E::G2Affine>,
    challenges: &[E::Fr],
    r_shift: &E::Fr,
    kzg_challenge: &E::Fr,
) -> PairingTuple<E> {
    // f_v(z)
    let vpoly_eval_z =
        polynomial_evaluation_product_form_from_transcript(challenges, kzg_challenge, r_shift);
    // -g such that when we test a pairing equation we only need to check if
    // it's equal 1 at the end:
    // e(a,b) = e(c,d) <=> e(a,b)e(-c,d) = 1
    let mut ng = v_srs.g.clone();
    ng.negate();
    par! {
        // verify first part of opening - v1
        // e(g, v1 h^{-af_v(z)})
        let p1 = E::miller_loop(&[(
            &ng.into_affine().prepare(),
            // in additive notation: final_vkey = uH,
            // uH - f_v(z)H = (u - f_v)H --> v1h^{-af_v(z)}
            &sub!(
                final_vkey.0.into_projective(),
                &mul!(v_srs.h_alpha, vpoly_eval_z)
            )
            .into_affine()
            .prepare(),
        )]),
        // e(g^{a - z}, opening_1) ==> (aG) - (zG)
        let p2 = E::miller_loop(&[(
            &sub!(v_srs.g_alpha, &mul!(v_srs.g, kzg_challenge.clone()))
                .into_affine()
                .prepare(),
            &vkey_opening.0.prepare(),
        )]),

        // verify second part of opening - v2 - similar but changing secret exponent
        // e(g, v2 h^{-bf_v(z)})
        let q1 = E::miller_loop(&[(
            &ng.into_affine().prepare(),
            // in additive notation: final_vkey = uH,
            // uH - f_v(z)H = (u - f_v)H --> v1h^{-f_v(z)}
            &sub!(
                final_vkey.1.into_projective(),
                &mul!(v_srs.h_beta, vpoly_eval_z)
            )
            .into_affine()
            .prepare(),
        )]),
        // e(g^{b - z}, opening_1)
        let q2 = E::miller_loop(&[(
            &sub!(v_srs.g_beta, &mul!(v_srs.g, kzg_challenge.clone()))
                .into_affine()
                .prepare(),
            &vkey_opening.1.prepare(),
        )])
    };

    // this pair should be one when multiplied
    let (l, r) = rayon::join(|| mul!(q1, &q2), || mul!(p1, &p2));
    PairingTuple::from_miller(mul!(l, &r))
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

    // -h such that when we test a pairing equation we only need to check if
    // it's equal 1 at the end:
    // e(a,b) = e(c,d) <=> e(a,b)e(c,-d) = 1
    let mut nh = v_srs.h.clone();
    nh.negate();

    par! {
        // first check on w1
        // let K = g^{a^{n+1}}
        // e(w1 K^{-f_w(z)},h)
        let p1 = E::miller_loop(&[(
            &sub!(
                final_wkey.0.into_projective(),
                &mul!(v_srs.g_alpha_n1, wkey_poly_eval)
            )
            .into_affine()
            .prepare(),
            &nh.into_affine().prepare(),
        )]),
        // e(opening, h^{a - z})
        let p2 = E::miller_loop(&[(
            &wkey_opening.0.prepare(),
            &sub!(v_srs.h_alpha, &mul!(v_srs.h, *kzg_challenge))
                .into_affine()
                .prepare(),
        )]),
        // then do second check
        // let K = g^{b^{n+1}}
        // e(w2 K^{-f_w(z)},h)
        let q1 = E::miller_loop(&[(
            &sub!(
                final_wkey.1.into_projective(),
                &mul!(v_srs.g_beta_n1, wkey_poly_eval)
            )
            .into_affine()
            .prepare(),
            &nh.into_affine().prepare(),
        )]),
        // e(opening, h^{b - z})
        let q2 = E::miller_loop(&[(
            &wkey_opening.1.prepare(),
            &sub!(v_srs.h_beta, &mul!(v_srs.h, *kzg_challenge))
                .into_affine()
                .prepare(),
        )])
    };
    let (l, r) = rayon::join(|| mul!(q1, &q2), || mul!(p1, &p2));
    PairingTuple::from_miller(mul!(l, &r))
}

fn verify_mipp<E: Engine>(
    v_srs: &VerifierSRS<E>,
    com_c: &commit::Output<E>, // original (T,U) = CM(v1,v2,C) - is rescaled in gipa verify
    agg_c: &E::G1,             // original Z = C^r - is rescaled in gipa verify
    proof: &MIPPProof<E>,
) -> PairingTuple<E> {
    info!("verify with structured scalar message");
    let now = Instant::now();
    let (com_tu, com_z, mut challenges, mut challenges_inv) =
        gipa_verify_mipp(com_c, agg_c, &proof.gipa);

    println!(
        "MIPP verify: gipa mipp verification took {}ms",
        now.elapsed().as_millis()
    );
    let now = Instant::now();

    let final_vkey = proof.gipa.final_vkey;
    // reverse the challenges so KZG polynomial is constructed correctly
    challenges.reverse();
    challenges_inv.reverse();

    // KZG challenge point
    let c = oracle!(&challenges.first().unwrap(), &final_vkey.0, &final_vkey.1);

    println!(
        "MIPP verify: mipp verification challenge took {}ms",
        now.elapsed().as_millis()
    );

    // final rescaled T U Z
    let (t, u) = com_tu;

    // final c from proof
    let final_c = proof.gipa.final_c.clone();
    let final_r = proof.gipa.final_r.clone();
    let now = Instant::now();

    par! {
        // Verify base inner product commitment
        // Z ==  c ^ r
        let final_z = inner_product::multiexponentiation::<E::G1Affine>(&[final_c], &[final_r]),
        // Check commitment key corectness
        let vtuple = verify_kzg_opening_g2(
            v_srs,
            &final_vkey,
            &proof.vkey_opening,
            &challenges_inv,
            &E::Fr::one(),
            &c,
        ),

        // Check commiment correctness 4.2.2
        // T = e(C,v1)
        let check_t = make_tuple(&final_c,&final_vkey.0,&t),
        // U = e(A,v2)
        let check_u = make_tuple(&final_c,&final_vkey.1,&u)
    };

    println!(
        "MIPP verify: parallel checks kZG + T & U took {}ms",
        now.elapsed().as_millis(),
    );
    let now = Instant::now();

    let b = final_z == com_z;
    // only check that doesn't require pairing so we can give a tuple that will
    // render the equation wrong in case it's false
    if !b {
        return PairingTuple::new_invalid();
    }

    let mut acc = vtuple;
    acc.merge(&check_t);
    acc.merge(&check_u);
    println!(
        "MIPP verify: final merge took {}ms",
        now.elapsed().as_millis(),
    );
    acc
}

/// gipa_verify_mipp returns the final reconstructed Z T U values, as described
/// in section 4.2.1 as well as all challenges generated.
fn gipa_verify_mipp<E: Engine>(
    com_c: &commit::Output<E>,
    z: &E::G1,
    proof: &GipaMIPP<E>,
) -> (commit::Output<E>, E::G1, Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa ssm verify recursive challenge challenges");
    let mut challenges = Vec::new();
    let mut challenges_inv = Vec::new();

    for ((tu_l, tu_r), (z_l, z_r)) in proof.comms.iter().zip(proof.z_vec.iter()) {
        // Fiat-Shamir challenge
        let default_transcript = E::Fr::zero();
        let transcript = challenges.last().unwrap_or(&default_transcript);
        let c_inv = oracle!(&transcript, &tu_r.0, &tu_r.1, &tu_l.0, &tu_l.1, &z_r, &z_l);
        let c = c_inv.inverse().unwrap();
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    println!(
        "MIPP verify: gipa challenge gen took {}ms",
        now.elapsed().as_millis()
    );
    let (mut comm_t, mut comm_u) = com_c.clone();
    let mut z = z.clone();

    let now = Instant::now();

    // Prepare the final commitment section 4.2. - steps 1.b
    let prep: Vec<(_, _, _, _, _, _)> = proof
        .comms
        .par_iter()
        .zip(proof.z_vec.par_iter())
        .zip(challenges.par_iter())
        .zip(challenges_inv.par_iter())
        .map(|((((c_l, c_r), (z_l, z_r)), c), c_inv)| {
            let (t_l, u_l) = c_l;
            let (t_r, u_r) = c_r;
            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();

            // z_l^x
            let mut z_l = z_l.clone();
            z_l.mul_assign(c_repr);
            // z_r^{x^{-1}}
            let mut z_r = z_r.clone();
            z_r.mul_assign(c_inv_repr);

            // u_r^x  , u_l^x^-1
            (
                t_l.pow(c_repr),
                u_l.pow(c_repr),
                t_r.pow(c_inv_repr),
                u_r.pow(c_inv_repr),
                z_l,
                z_r,
            )
        })
        .collect();

    println!(
        "MIPP verify: gipa preparation took {}ms ({})",
        now.elapsed().as_millis(),
        prep.len()
    );

    let now = Instant::now();
    for (t_l_c, u_l_c, t_r_cinv, u_r_cinv, z_l_c, z_c_cinv) in prep.iter() {
        // TODO parallelize ?
        comm_t.mul_assign(t_l_c);
        comm_t.mul_assign(t_r_cinv);
        comm_u.mul_assign(u_l_c);
        comm_u.mul_assign(u_r_cinv);
        z.add_assign(z_l_c);
        z.add_assign(z_c_cinv);
    }
    println!(
        "MIPP verify: gipa accumulation step took {}ms",
        now.elapsed().as_millis()
    );

    ((comm_t, comm_u), z, challenges, challenges_inv)
}

fn make_tuple<E: Engine>(left: &E::G1Affine, right: &E::G2Affine, out: &E::Fqk) -> PairingTuple<E> {
    PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&[left.clone()], &[right.clone()]),
        out.clone(),
    )
}
