/// TODO global commitment doc and usage
use ff::PrimeField;
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::aggregate::accumulator::PairingTuple;
use crate::groth16::aggregate::inner_product;
use crate::groth16::multiscalar::*;

/// VKey is a commitment key used by the "single" commitment on G1 values as
/// well as in the "pair" commtitment.
type VKey struct<E: Engine>{
    /// $\{h^a^i\}_{i=1}^n$
    pub v1: &[E::G2],
    /// $\{h^b^i\}_{i=1}^n$
    pub v2: &[E::G2],
}

/// WKey is a commitment key used by the "pair" commitment. Note the sequence of
/// powers starts at $n$ already.
type WKey struct<E :Engine>{
    /// $\{g^{a^{n+i}}\}_{i=1}^n$
    pub w1: &[E::G1],
    /// $\{g^{b^{n+i}}\}_{i=1}^n$
    pub w2: &[E::G2],
}

/// Both commitment outputs a pair of $F_q^k$ element.
type Output<E: Engine> = (E::Fqk, E::Fqk);

/// single_g1 commits to a single vector of G1 elements in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})$
/// Output is $(T,U)$
pub fn single_g1<E: Engine>(vkey: &VKey<E>, A: &[E::G1]) -> Output<E> {
    let T = inner_product::pairing_miller(A, v1);
    let U = inner_product::pairing_miller(A, v2);
    return Output(T, U);
}

/// pair commits to a tuple of G1 vector and G2 vector in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})e(B_i,w_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})e(B_i,w_{2,i})$
/// Output is $(T,U)$
pub fn pair<E: Engine>(
    vkey: &VKey<E>, wkey: &WKey<E>,
    A: &[E::G1],
    B: &[E::G2],
) -> Output<E> {
    let mut T1 = inner_product::pairing_miller(A, v1);
    let T2 = inner_product::pairing_miller(w1, B);
    let mut U1 = inner_product::pairing_miller(A, v2);
    let U2 = inner_product::pairing_miller(w2, B);
    T1.mul_assign(&T2);
    U1.mul_assign(&U2);
    return Output(T1, U1);
}
