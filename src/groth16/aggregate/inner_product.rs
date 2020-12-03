use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use super::msm;
use crate::bls::{Engine, PairingCurveAffine};

pub fn pairing_miller_affine<E: Engine>(left: &[E::G1Affine], right: &[E::G2Affine]) -> E::Fqk {
    assert_eq!(left.len(), right.len());
    let pairs = left
        .par_iter()
        .map(|e| e.prepare())
        .zip(right.par_iter().map(|e| e.prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    E::miller_loop(pairs_ref.iter())
}

pub fn pairing_miller<E: Engine>(left: &[E::G1], right: &[E::G2]) -> E::Fqk {
    assert_eq!(left.len(), right.len());
    let pairs = left
        .par_iter()
        .map(|e| e.into_affine().prepare())
        .zip(right.par_iter().map(|e| e.into_affine().prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    E::miller_loop(pairs_ref.iter())
}

/// Returns the miller loop result of the inner pairing product
pub fn pairing<E: Engine>(left: &[E::G1], right: &[E::G2]) -> E::Fqk {
    E::final_exponentiation(&pairing_miller::<E>(left, right)).expect("invalid pairing")
}

/// Returns the miller loop result of the inner pairing product
pub fn pairing_affine<E: Engine>(left: &[E::G1Affine], right: &[E::G2Affine]) -> E::Fqk {
    E::final_exponentiation(&pairing_miller_affine::<E>(left, right)).expect("invalid pairing")
}

pub fn multiexponentiation<G: CurveProjective>(left: &[G], right: &[G::Scalar]) -> G {
    assert_eq!(left.len(), right.len());
    msm::variable_base::multi_scalar_mul(
        &left.par_iter().map(|b| b.into_affine()).collect::<Vec<_>>(),
        &right.par_iter().map(|b| b.into_repr()).collect::<Vec<_>>(),
    )
}

pub fn multiexponentiation_affine<G: CurveAffine>(
    left: &[G],
    right: &[G::Scalar],
) -> G::Projective {
    assert_eq!(left.len(), right.len());
    msm::variable_base::multi_scalar_mul(
        &left,
        &right.par_iter().map(|b| b.into_repr()).collect::<Vec<_>>(),
    )
}

pub fn scalar<F: Field>(left: &[F], right: &[F]) -> F {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(x, y)| {
            let mut x = *x;
            x.mul_assign(y);
            x
        })
        .fold(F::zero(), |mut acc, curr| {
            acc.add_assign(&curr);
            acc
        })
}
