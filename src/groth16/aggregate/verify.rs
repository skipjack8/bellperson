use crossbeam_channel::bounded;
use digest::Digest;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use log::*;
use rayon::prelude::*;
use sha2::Sha256;

use super::{
    accumulator::PairingTuple, inner_product,
    prove::polynomial_evaluation_product_form_from_transcript, structured_scalar_power,
    AggregateProof, KZGOpening, VerifierSRS,
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
        // 2.Check TIPA proof c
        let tipa_ab = send_tuple.clone();
        s.spawn(move |_| {
            let now = Instant::now();
            let tuple = verify_tipp_mipp::<E>(
                ip_verifier_srs,
                proof,
                &r, // we give the extra r as it's not part of the proof itself - it is simply used on top for the groth16 aggregation
            );
            println!("TIPP took {} ms", now.elapsed().as_millis());
            tipa_ab.send(tuple).unwrap();
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
fn verify_tipp_mipp<E: Engine>(
    v_srs: &VerifierSRS<E>,
    proof: &AggregateProof<E>,
    r_shift: &E::Fr,
) -> PairingTuple<E> {
    info!("verify with srs shift");
    let now = Instant::now();
    // (T,U), Z for TIPP and MIPP  and all challenges
    let (final_res, mut challenges, mut challenges_inv) = gipa_verify_tipp(&proof);
    println!(
        "TIPP verify: gipa verify tipp {}ms",
        now.elapsed().as_millis()
    );

    // we reverse the order so the KZG polynomial have them in the expected
    // order to construct them in logn time.
    challenges.reverse();
    challenges_inv.reverse();
    // Verify commitment keys wellformed
    let fvkey = proof.tmipp.gipa.final_vkey;
    let fwkey = proof.tmipp.gipa.final_wkey;
    // KZG challenge point
    let c = oracle!(
        &challenges.first().unwrap(),
        &fvkey.0,
        &fvkey.1,
        &fwkey.0,
        &fwkey.1
    );

    let final_a = &proof.tmipp.gipa.final_a;
    let final_b = &proof.tmipp.gipa.final_b;
    let final_c = &proof.tmipp.gipa.final_c;
    let final_r = &proof.tmipp.gipa.final_r;
    let final_zab = &final_res.zab;
    let final_tab = &final_res.tab;
    let final_uab = &final_res.uab;
    let final_tc = &final_res.tc;
    let final_uc = &final_res.uc;

    let now = Instant::now();
    par! {
        // Section 3.4. step 5 check the opening proof for v
        let vtuple = verify_kzg_opening_g2(
            v_srs,
            &fvkey,
            &proof.tmipp.vkey_opening,
            &challenges_inv,
            &E::Fr::one(),
            &c,
        ),
        // Section 3.4 step 6 check the opening proof for w
        let wtuple = verify_kzg_opening_g1(
            v_srs,
            &fwkey,
            &proof.tmipp.wkey_opening,
            &challenges,
            &r_shift.inverse().unwrap(),
            &c,
        ),
        // TIPP
        // Section 3.4 step 2
        // z = e(A,B)
        let check_z = make_tuple(final_a, final_b, final_zab),
        //  final_aB.0 = T = e(A,v1)e(w1,B)
        let check_ab11 = make_tuple(final_a, &fvkey.0, &E::Fqk::one()),
        let check_ab12 = make_tuple(&fwkey.0, final_b, final_tab),
        let check_ab21 = make_tuple(final_a, &fvkey.1, &E::Fqk::one()),
        let check_ab22 = make_tuple(&fwkey.1, final_b,
            final_uab),

        // MIPP
        // Verify base inner product commitment
        // Z ==  c ^ r
        let final_z =
            inner_product::multiexponentiation::<E::G1Affine>(&[final_c.clone()],
            &[final_r.clone()]),
        // Check commiment correctness 4.2.2
        // T = e(C,v1)
        let check_t = make_tuple(final_c,&fvkey.0,final_tc),
        // U = e(A,v2)
        let check_u = make_tuple(final_c,&fvkey.1,final_uc)
    };

    println!(
        "TIPP verify: parallel checks before merge: {}ms",
        now.elapsed().as_millis(),
    );

    let b = final_z == final_res.zc;
    // only check that doesn't require pairing so we can give a tuple that will
    // render the equation wrong in case it's false
    if !b {
        return PairingTuple::new_invalid();
    }
    let now = Instant::now();
    let mut acc = vtuple;
    acc.merge(&check_z);
    acc.merge(&check_ab11);
    acc.merge(&check_ab12);
    acc.merge(&check_ab21);
    acc.merge(&check_ab22);
    acc.merge(&check_t);
    acc.merge(&check_u);
    acc.merge(&wtuple);
    println!("TIPP verify: final merge {}ms", now.elapsed().as_millis());
    acc
}

/// gipa_verify_tipp recurse on the proof and statement and produces the final
/// values to be checked by TIPP verifier, namely:
/// (T, U), Z, challenges, challenges_inv
/// T,U are the final commitment values of A and B and Z the final product
/// between A and B. Challenges are returned in inverse order as well to avoid
/// repeating the operation multiple times later on.
fn gipa_verify_tipp<E: Engine>(proof: &AggregateProof<E>) -> (GipaTUZ<E>, Vec<E::Fr>, Vec<E::Fr>) {
    info!("gipa verify TIPP");
    let gipa = &proof.tmipp.gipa;
    let comms_ab = &gipa.comms_ab;
    let comms_c = &gipa.comms_c;
    let zs_ab = &gipa.z_ab;
    let zs_c = &gipa.z_c;

    let now = Instant::now();

    let mut challenges = Vec::new();
    let mut challenges_inv = Vec::new();

    let default_transcript = E::Fr::zero();

    // We first generate all challenges as this is the only consecutive process
    // that can not be parallelized then we scale the commitments in a
    // parallelized way
    for ((comm_ab, z_ab), (comm_c, z_c)) in comms_ab
        .iter()
        .zip(zs_ab.iter())
        .zip(comms_c.iter().zip(zs_c.iter()))
    {
        let (tab_l, tab_r) = comm_ab;
        let (zab_l, zab_r) = z_ab;
        let (tc_l, tc_r) = comm_c;
        let (zc_l, zc_r) = z_c;
        // Fiat-Shamir challenge
        let transcript = challenges.last().unwrap_or(&default_transcript);
        let c_inv = oracle!(
            &transcript,
            &tab_l.0,
            &tab_l.1,
            &tab_r.0,
            &tab_r.1,
            &zab_l,
            &zab_r,
            &zc_l,
            &zc_r,
            &tc_l.0,
            &tc_l.1,
            &tc_r.0,
            &tc_r.1
        );
        let c = c_inv.inverse().unwrap();
        challenges.push(c);
        challenges_inv.push(c_inv);
    }

    println!(
        "TIPP verify: gipa challenge gen took {}ms",
        now.elapsed().as_millis()
    );

    let now = Instant::now();
    // output of the pair commitment T and U in TIPP -> COM((v,w),A,B)
    let (t_ab, u_ab) = proof.com_ab.clone();
    let z_ab = proof.ip_ab; // in the end must be equal to Z = A^r * B

    // COM(v,C)
    let (t_c, u_c) = proof.com_c.clone();
    let z_c = proof.agg_c; // in the end must be equal to Z = C^r

    let mut final_res = GipaTUZ {
        tab: t_ab,
        uab: u_ab,
        zab: z_ab,
        tc: t_c,
        uc: u_c,
        zc: z_c,
    };

    // we first multiply each entry of the Z U and L vectors by the respective
    // challenges independently
    // Since at the end we want to multiple all "t" values together, we do
    // multiply all of them in parrallel and then merge then back at the end.
    // same for u and z.
    enum Op<'a, E: Engine> {
        TAB(&'a E::Fqk, <E::Fr as PrimeField>::Repr),
        UAB(&'a E::Fqk, <E::Fr as PrimeField>::Repr),
        ZAB(&'a E::Fqk, <E::Fr as PrimeField>::Repr),
        TC(&'a E::Fqk, <E::Fr as PrimeField>::Repr),
        UC(&'a E::Fqk, <E::Fr as PrimeField>::Repr),
        ZC(&'a E::G1, <E::Fr as PrimeField>::Repr),
    }

    let res = comms_ab
        .par_iter()
        .zip(zs_ab.par_iter())
        .zip(comms_c.par_iter().zip(zs_c.par_iter()))
        .zip(challenges.par_iter().zip(challenges_inv.par_iter()))
        .flat_map(|(((comm_ab, z_ab), (comm_c, z_c)), (c, c_inv))| {
            // T and U values for right and left for AB part
            let ((tab_l, uab_l), (tab_r, uab_r)) = comm_ab;
            let (zab_l, zab_r) = z_ab;
            // T and U values for right and left for C part
            let ((tc_l, uc_l), (tc_r, uc_r)) = comm_c;
            let (zc_l, zc_r) = z_c;

            let c_repr = c.into_repr();
            let c_inv_repr = c_inv.into_repr();

            // we multiple left side by x and right side by x^-1
            vec![
                Op::TAB::<E>(tab_l, c_repr),
                Op::TAB(tab_r, c_inv_repr),
                Op::UAB(uab_l, c_repr),
                Op::UAB(uab_r, c_inv_repr),
                Op::ZAB(zab_l, c_repr),
                Op::ZAB(zab_r, c_inv_repr),
                Op::TC::<E>(tc_l, c_repr),
                Op::TC(tc_r, c_inv_repr),
                Op::UC(uc_l, c_repr),
                Op::UC(uc_r, c_inv_repr),
                Op::ZC(zc_l, c_repr),
                Op::ZC(zc_r, c_inv_repr),
            ]
        })
        .fold(GipaTUZ::<E>::empty, |mut res, op: Op<E>| {
            match op {
                Op::TAB(tx, c) => {
                    let tx: E::Fqk = tx.pow(c);
                    res.tab.mul_assign(&tx);
                }
                Op::UAB(ux, c) => {
                    let ux: E::Fqk = ux.pow(c);
                    res.uab.mul_assign(&ux);
                }
                Op::ZAB(zx, c) => {
                    let zx: E::Fqk = zx.pow(c);
                    res.zab.mul_assign(&zx);
                }
                Op::TC(tx, c) => {
                    let tx: E::Fqk = tx.pow(c);
                    res.tc.mul_assign(&tx);
                }
                Op::UC(ux, c) => {
                    let ux: E::Fqk = ux.pow(c);
                    res.uc.mul_assign(&ux);
                }
                Op::ZC(zx, c) => {
                    let mut zx = *zx;
                    zx.mul_assign(c);
                    res.zc.add_assign(&zx);
                }
            }
            res
        })
        .reduce(GipaTUZ::empty, |mut acc_res, res| {
            acc_res.merge(&res);
            acc_res
        });

    final_res.merge(&res);
    println!(
        "TIPP verify: gipa prep and accumulate took {}ms",
        now.elapsed().as_millis()
    );
    (final_res, challenges, challenges_inv)
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

fn make_tuple<E: Engine>(left: &E::G1Affine, right: &E::G2Affine, out: &E::Fqk) -> PairingTuple<E> {
    PairingTuple::<E>::from_pair(
        inner_product::pairing_miller_affine::<E>(&[left.clone()], &[right.clone()]),
        out.clone(),
    )
}

/// Keeps track of the variables that have been sent by the prover and must
/// be multiplied together by the verifier. Both MIPP and TIPP are merged
/// together.
struct GipaTUZ<E: Engine> {
    pub tab: E::Fqk,
    pub uab: E::Fqk,
    pub zab: E::Fqk,
    pub tc: E::Fqk,
    pub uc: E::Fqk,
    pub zc: E::G1,
}

impl<E> GipaTUZ<E>
where
    E: Engine,
{
    fn empty() -> Self {
        Self {
            tab: E::Fqk::one(),
            uab: E::Fqk::one(),
            zab: E::Fqk::one(),
            tc: E::Fqk::one(),
            uc: E::Fqk::one(),
            zc: E::G1::zero(),
        }
    }
    fn merge(&mut self, other: &Self) {
        self.tab.mul_assign(&other.tab);
        self.uab.mul_assign(&other.uab);
        self.zab.mul_assign(&other.zab);
        self.tc.mul_assign(&other.tc);
        self.uc.mul_assign(&other.uc);
        self.zc.add_assign(&other.zc);
    }
}
