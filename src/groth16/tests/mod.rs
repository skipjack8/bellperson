use blstrs::*;

mod dummy_engine;
use self::dummy_engine::*;

use super::{
    create_proof, create_proof_batch, generate_parameters, prepare_verifying_key, verify_proof,
};
use crate::{Circuit, ConstraintSystem, SynthesisError};

#[derive(Clone)]
struct XORDemo {
    a: Option<bool>,
    b: Option<bool>,
}

impl Circuit for XORDemo {
    fn synthesize<CS: ConstraintSystem>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let a_var = cs.alloc(
            || "a",
            || {
                if self.a.is_some() {
                    if self.a.unwrap() {
                        Ok(Scalar::one())
                    } else {
                        Ok(Scalar::zero())
                    }
                } else {
                    Err(SynthesisError::AssignmentMissing)
                }
            },
        )?;

        cs.enforce(
            || "a_boolean_constraint",
            |lc| lc + CS::one() - a_var,
            |lc| lc + a_var,
            |lc| lc,
        );

        let b_var = cs.alloc(
            || "b",
            || {
                if self.b.is_some() {
                    if self.b.unwrap() {
                        Ok(Scalar::one())
                    } else {
                        Ok(Scalar::zero())
                    }
                } else {
                    Err(SynthesisError::AssignmentMissing)
                }
            },
        )?;

        cs.enforce(
            || "b_boolean_constraint",
            |lc| lc + CS::one() - b_var,
            |lc| lc + b_var,
            |lc| lc,
        );

        let c_var = cs.alloc_input(
            || "c",
            || {
                if self.a.is_some() && self.b.is_some() {
                    if self.a.unwrap() ^ self.b.unwrap() {
                        Ok(Scalar::one())
                    } else {
                        Ok(Scalar::zero())
                    }
                } else {
                    Err(SynthesisError::AssignmentMissing)
                }
            },
        )?;

        cs.enforce(
            || "c_xor_constraint",
            |lc| lc + a_var + a_var,
            |lc| lc + b_var,
            |lc| lc + a_var + b_var - c_var,
        );

        Ok(())
    }
}

#[test]
fn test_xordemo() {
    let g1 = Scalar::one();
    let g2 = Scalar::one();
    let alpha = Scalar::from_str("48577").unwrap();
    let beta = Scalar::from_str("22580").unwrap();
    let gamma = Scalar::from_str("53332").unwrap();
    let delta = Scalar::from_str("5481").unwrap();
    let tau = Scalar::from_str("3673").unwrap();

    let params = {
        let c = XORDemo::<DummyEngine> { a: None, b: None };

        generate_parameters(c, g1, g2, alpha, beta, gamma, delta, tau).unwrap()
    };

    // This will synthesize the constraint system:
    //
    // public inputs: a_0 = 1, a_1 = c
    // aux inputs: a_2 = a, a_3 = b
    // constraints:
    //     (a_0 - a_2) * (a_2) = 0
    //     (a_0 - a_3) * (a_3) = 0
    //     (a_2 + a_2) * (a_3) = (a_2 + a_3 - a_1)
    //     (a_0) * 0 = 0
    //     (a_1) * 0 = 0

    // The evaluation domain is 8. The H query should
    // have 7 elements (it's a quotient polynomial)
    assert_eq!(7, params.h.len());

    let mut root_of_unity = Scalar::root_of_unity();

    // We expect this to be a 2^10 root of unity
    assert_eq!(Scalar::one(), root_of_unity.pow(&[1 << 10]));

    // Let's turn it into a 2^3 root of unity.
    root_of_unity = root_of_unity.pow(&[1 << 7]);
    assert_eq!(Scalar::one(), root_of_unity.pow(&[1 << 3]));
    assert_eq!(Scalar::from_str("20201").unwrap(), root_of_unity);

    // Let's compute all the points in our evaluation domain.
    let mut points = Vec::with_capacity(8);
    for i in 0..8 {
        points.push(root_of_unity.pow(&[i]));
    }

    // Let's compute t(tau) = (tau - p_0)(tau - p_1)...
    //                      = tau^8 - 1
    let mut t_at_tau = tau.pow(&[8, 0, 0, 0]);
    t_at_tau.sub_assign(&Scalar::one());
    {
        let mut tmp = Scalar::one();
        for p in &points {
            let mut term = tau;
            term.sub_assign(p);
            tmp *= &term;
        }
        assert_eq!(tmp, t_at_tau);
    }

    // We expect our H query to be 7 elements of the form...
    // {tau^i t(tau) / delta}
    let delta_inverse = delta.inverse().unwrap();
    let gamma_inverse = gamma.inverse().unwrap();
    {
        let mut coeff = delta_inverse;
        coeff *= &t_at_tau;

        let mut cur = Scalar::one();
        for h in params.h.iter() {
            let mut tmp = cur;
            tmp *= &coeff;

            assert_eq!(*h, tmp);

            cur *= &tau;
        }
    }

    // The density of the IC query is 2 (2 inputs)
    assert_eq!(2, params.vk.ic.len());

    // The density of the L query is 2 (2 aux variables)
    assert_eq!(2, params.l.len());

    // The density of the A query is 4 (each variable is in at least one A term)
    assert_eq!(4, params.a.len());

    // The density of the B query is 2 (two variables are in at least one B term)
    assert_eq!(2, params.b_g1.len());
    assert_eq!(2, params.b_g2.len());

    /*
    Lagrange interpolation polynomials in our evaluation domain:

    ,-------------------------------. ,-------------------------------. ,-------------------------------.
    |            A TERM             | |            B TERM             | |            C TERM             |
    `-------------------------------. `-------------------------------' `-------------------------------'
    | a_0   | a_1   | a_2   | a_3   | | a_0   | a_1   | a_2   | a_3   | | a_0   | a_1   | a_2   | a_3   |
    | 1     | 0     | 64512 | 0     | | 0     | 0     | 1     | 0     | | 0     | 0     | 0     | 0     |
    | 1     | 0     | 0     | 64512 | | 0     | 0     | 0     | 1     | | 0     | 0     | 0     | 0     |
    | 0     | 0     | 2     | 0     | | 0     | 0     | 0     | 1     | | 0     | 64512 | 1     | 1     |
    | 1     | 0     | 0     | 0     | | 0     | 0     | 0     | 0     | | 0     | 0     | 0     | 0     |
    | 0     | 1     | 0     | 0     | | 0     | 0     | 0     | 0     | | 0     | 0     | 0     | 0     |
    `-------'-------'-------'-------' `-------'-------'-------'-------' `-------'-------'-------'-------'

    Example for u_0:

    sage: r = 64513
    sage: Fr = GF(r)
    sage: omega = (Fr(5)^63)^(2^7)
    sage: tau = Fr(3673)
    sage: R.<x> = PolynomialRing(Fr, 'x')
    sage: def eval(tau, c0, c1, c2, c3, c4):
    ....:     p = R.lagrange_polynomial([(omega^0, c0), (omega^1, c1), (omega^2, c2), (omega^3, c3), (omega^4, c4), (omega^5, 0), (omega^6, 0), (omega^7, 0)])
    ....:     return p.substitute(tau)
    sage: eval(tau, 1, 1, 0, 1, 0)
    59158
    */

    let u_i = [59158, 48317, 21767, 10402]
        .iter()
        .map(|e| Scalar::from_str(&format!("{}", e)).unwrap())
        .collect::<Vec<Fr>>();
    let v_i = [0, 0, 60619, 30791]
        .iter()
        .map(|e| Scalar::from_str(&format!("{}", e)).unwrap())
        .collect::<Vec<Fr>>();
    let w_i = [0, 23320, 41193, 41193]
        .iter()
        .map(|e| Scalar::from_str(&format!("{}", e)).unwrap())
        .collect::<Vec<Fr>>();

    for (u, a) in u_i.iter().zip(&params.a[..]) {
        assert_eq!(u, a);
    }

    for (v, b) in v_i
        .iter()
        .filter(|&&e| e != Scalar::zero())
        .zip(&params.b_g1[..])
    {
        assert_eq!(v, b);
    }

    for (v, b) in v_i
        .iter()
        .filter(|&&e| e != Scalar::zero())
        .zip(&params.b_g2[..])
    {
        assert_eq!(v, b);
    }

    for i in 0..4 {
        let mut tmp1 = beta;
        tmp1 *= &u_i[i];

        let mut tmp2 = alpha;
        tmp2 *= &v_i[i];

        tmp1 += &tmp2;
        tmp1 += &w_i[i];

        if i < 2 {
            // Check the correctness of the IC query elements
            tmp1 *= &gamma_inverse;

            assert_eq!(tmp1, params.vk.ic[i]);
        } else {
            // Check the correctness of the L query elements
            tmp1 *= &delta_inverse;

            assert_eq!(tmp1, params.l[i - 2]);
        }
    }

    // Check consistency of the other elements
    assert_eq!(alpha, params.vk.alpha_g1);
    assert_eq!(beta, params.vk.beta_g1);
    assert_eq!(beta, params.vk.beta_g2);
    assert_eq!(gamma, params.vk.gamma_g2);
    assert_eq!(delta, params.vk.delta_g1);
    assert_eq!(delta, params.vk.delta_g2);

    let pvk = prepare_verifying_key(&params.vk);

    let r = Scalar::from_str("27134").unwrap();
    let s = Scalar::from_str("17146").unwrap();

    let proof = {
        let c = XORDemo {
            a: Some(true),
            b: Some(false),
        };

        create_proof(c, &params, r, s).unwrap()
    };

    // A(x) =
    //  a_0 * (44865*x^7 + 56449*x^6 + 44865*x^5 + 8064*x^4 + 3520*x^3 + 56449*x^2 + 3520*x + 40321) +
    //  a_1 * (8064*x^7 + 56449*x^6 + 8064*x^5 + 56449*x^4 + 8064*x^3 + 56449*x^2 + 8064*x + 56449) +
    //  a_2 * (16983*x^7 + 24192*x^6 + 63658*x^5 + 56449*x^4 + 16983*x^3 + 24192*x^2 + 63658*x + 56449) +
    //  a_3 * (5539*x^7 + 27797*x^6 + 6045*x^5 + 56449*x^4 + 58974*x^3 + 36716*x^2 + 58468*x + 8064) +
    {
        // proof A = alpha + A(tau) + delta * r
        let mut expected_a = delta;
        expected_a *= &r;
        expected_a += &alpha;
        expected_a += &u_i[0]; // a_0 = 1
        expected_a += &u_i[1]; // a_1 = 1
        expected_a += &u_i[2]; // a_2 = 1
                               // a_3 = 0
        assert_eq!(proof.a, expected_a);
    }

    // B(x) =
    // a_0 * (0) +
    // a_1 * (0) +
    // a_2 * (56449*x^7 + 56449*x^6 + 56449*x^5 + 56449*x^4 + 56449*x^3 + 56449*x^2 + 56449*x + 56449) +
    // a_3 * (31177*x^7 + 44780*x^6 + 21752*x^5 + 42255*x^3 + 35861*x^2 + 33842*x + 48385)
    {
        // proof B = beta + B(tau) + delta * s
        let mut expected_b = delta;
        expected_b *= &s;
        expected_b += &beta;
        expected_b += &v_i[0]; // a_0 = 1
        expected_b += &v_i[1]; // a_1 = 1
        expected_b += &v_i[2]; // a_2 = 1
                               // a_3 = 0
        assert_eq!(proof.b, expected_b);
    }

    // C(x) =
    // a_0 * (0) +
    // a_1 * (27797*x^7 + 56449*x^6 + 36716*x^5 + 8064*x^4 + 27797*x^3 + 56449*x^2 + 36716*x + 8064) +
    // a_2 * (36716*x^7 + 8064*x^6 + 27797*x^5 + 56449*x^4 + 36716*x^3 + 8064*x^2 + 27797*x + 56449) +
    // a_3 * (36716*x^7 + 8064*x^6 + 27797*x^5 + 56449*x^4 + 36716*x^3 + 8064*x^2 + 27797*x + 56449)
    //
    // If A * B = C at each point in the domain, then the following polynomial...
    // P(x) = A(x) * B(x) - C(x)
    //      = 49752*x^14 + 13914*x^13 + 29243*x^12 + 27227*x^11 + 62362*x^10 + 35703*x^9 + 4032*x^8 + 14761*x^6 + 50599*x^5 + 35270*x^4 + 37286*x^3 + 2151*x^2 + 28810*x + 60481
    //
    // ... should be divisible by t(x), producing the quotient polynomial:
    // h(x) = P(x) / t(x)
    //      = 49752*x^6 + 13914*x^5 + 29243*x^4 + 27227*x^3 + 62362*x^2 + 35703*x + 4032
    {
        let mut expected_c = Scalar::zero();

        // A * s
        let mut tmp = proof.a;
        tmp *= &s;
        expected_c += &tmp;

        // B * r
        let mut tmp = proof.b;
        tmp *= &r;
        expected_c += &tmp;

        // delta * r * s
        let mut tmp = delta;
        tmp *= &r;
        tmp *= &s;
        expected_c.sub_assign(&tmp);

        // L query answer
        // a_2 = 1, a_3 = 0
        expected_c += &params.l[0];

        // H query answer
        for (i, coeff) in [5040, 11763, 10755, 63633, 128, 9747, 8739]
            .iter()
            .enumerate()
        {
            let coeff = Scalar::from_str(&format!("{}", coeff)).unwrap();

            let mut tmp = params.h[i];
            tmp *= &coeff;
            expected_c += &tmp;
        }

        assert_eq!(expected_c, proof.c);
    }

    assert!(verify_proof(&pvk, &proof, &[Scalar::one()]).unwrap());
}

#[test]
fn test_create_batch_single() {
    // test consistency between single and batch creation
    let g1 = G1Projective::one();
    let g2 = G2Projective::one();
    let alpha = Scalar::from(48577);
    let beta = Scalar::from(22580);
    let gamma = Scalar::from(53332);
    let delta = Scalar::from(5481);
    let tau = Scalar::from(3673);

    let params = {
        let c = XORDemo::<DummyEngine> { a: None, b: None };

        generate_parameters(c, g1, g2, alpha, beta, gamma, delta, tau).unwrap()
    };

    let pvk = prepare_verifying_key(&params.vk);

    let r1 = Scalar::from_str("27134").unwrap();
    let s1 = Scalar::from_str("17146").unwrap();

    let r2 = Scalar::from_str("27132").unwrap();
    let s2 = Scalar::from_str("17142").unwrap();

    let c = XORDemo {
        a: Some(true),
        b: Some(false),
    };
    let proof_single_1 = create_proof(c.clone(), &params, r1, s1).unwrap();
    let proof_single_2 = create_proof(c.clone(), &params, r2, s2).unwrap();

    let proof_batch = create_proof_batch(
        vec![c.clone(), c.clone()],
        &params,
        vec![r1, r2],
        vec![s1, s2],
    )
    .unwrap();

    assert_eq!(proof_batch[0], proof_single_1);
    assert_eq!(proof_batch[1], proof_single_2);

    assert!(verify_proof(&pvk, &proof_single_1, &[Scalar::one()]).unwrap());
    assert!(verify_proof(&pvk, &proof_single_2, &[Scalar::one()]).unwrap());
    for proof in &proof_batch {
        assert!(verify_proof(&pvk, &proof, &[Scalar::one()]).unwrap());
    }
}
