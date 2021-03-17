use ff::PrimeField;
use groupy::CurveAffine;
use rayon::prelude::*;

use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::multiscalar::*;

/// Returns the miller loop evaluated on inputs, i.e.
/// e(l_1,r_1)e(l_2,r_2)...
/// NOTE: the result is not in the final subgroup, one must run
/// `E::final_exponentiation` to use the final result.
pub fn pairing_miller_affine<E: Engine>(left: &[E::G1Affine], right: &[E::G2Affine]) -> E::Fqk {
    debug_assert_eq!(left.len(), right.len());
    let pairs = left
        .par_iter()
        .map(|e| e.prepare())
        .zip(right.par_iter().map(|e| e.prepare()))
        .collect::<Vec<_>>();
    let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();

    E::miller_loop(pairs_ref.iter())
}

/// Returns the miller loop result of the inner pairing product
pub fn pairing<E: Engine>(left: &[E::G1Affine], right: &[E::G2Affine]) -> E::Fqk {
    E::final_exponentiation(&pairing_miller_affine::<E>(left, right)).expect("invalid pairing")
}

pub fn multiexponentiation<G: CurveAffine>(left: &[G], right: &[G::Scalar]) -> G::Projective {
    debug_assert_eq!(left.len(), right.len());

    let table = precompute_fixed_window::<G>(&left, WINDOW_SIZE);
    multiexponentiation_with_table::<G>(&table, right)
}

pub fn multiexponentiation_with_table<G: CurveAffine>(
    table: &dyn MultiscalarPrecomp<G>,
    right: &[G::Scalar],
) -> G::Projective {
    let getter = |i: usize| -> <G::Scalar as PrimeField>::Repr { right[i].into_repr() };
    par_multiscalar::<_, G>(
        &ScalarList::Getter(getter, right.len()),
        table,
        std::mem::size_of::<<G::Scalar as PrimeField>::Repr>() * 8,
    )
}
