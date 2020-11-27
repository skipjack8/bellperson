use ff::{Field, PrimeField};
use groupy::CurveProjective;

use super::msm;
use crate::bls::{Engine, PairingCurveAffine};

pub fn pairing<E: Engine>(left: &[E::G1], right: &[E::G2]) -> E::Fqk {
    assert_eq!(left.len(), right.len());
    let pairs = left
        .iter()
        .map(|e| e.into_affine())
        .zip(right.iter().map(|e| e.into_affine()))
        .map(|(a, b)| (a.prepare(), b.prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    let ml: E::Fqk = E::miller_loop(pairs_ref.iter());
    E::final_exponentiation(&ml).expect("invalid pairing")
}

pub fn multiexponentiation<G: CurveProjective>(left: &[G], right: &[G::Scalar]) -> G {
    assert_eq!(left.len(), right.len());
    msm::variable_base::multi_scalar_mul(
        &left.iter().map(|b| b.into_affine()).collect::<Vec<_>>(),
        &right.iter().map(|b| b.into_repr()).collect::<Vec<_>>(),
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
