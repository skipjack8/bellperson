/// TODO global commitment doc and usage
use ff::PrimeField;
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::aggregate::accumulator::PairingTuple;
use crate::groth16::aggregate::inner_product;
use crate::groth16::multiscalar::*;

/// Key is a generic commitment key that is instanciated with g and h as basis,
/// and a and b as powers.
#[derive(Clone,Debug)]
pub type Key struct<G: CurveProjective> {
    /// Exponent is a
    pub a: Vec<G>,
    /// Exponent is b
    pub b: Vec<G>,
}
/// VKey is a commitment key used by the "single" commitment on G1 values as
/// well as in the "pair" commtitment.
/// It contains $\{h^a^i\}_{i=1}^n$ and $\{h^b^i\}_{i=1}^n$
pub type VKey<E: Engine> = Key<E::G2>;

/// WKey is a commitment key used by the "pair" commitment. Note the sequence of
/// powers starts at $n$ already.
/// It contains $\{g^{a^{n+i}}\}_{i=1}^n$ and $\{g^{b^{n+i}}\}_{i=1}^n$
pub type WKey<E: Engine> = Key<E::G1>;

impl<G> Key<G> where G: CurveProjective {
    /// correct_len returns true if both commitment keys have the same size as
    /// the argument. It is necessary for the IPP scheme to work that commitment
    /// key have the exact same number of arguments as the number of proofs to
    /// aggregate.
    pub fn correct_len(&self,n:usize) -> bool {
        self.cka.len() == n && self.ckb.len() == n 
    }

    /// scale returns both vectors scaled by the given vector entrywise.
    /// In other words, it returns $\{v_i^{s_i}\}$
    pub fn scale(&self,s_vec: &[G::Scalar]) -> Self {
        let a = self.a.par_iter()
                    .zip(r_vec.par_iter())
                    .map(|(a, s)| mul!(a, s))
                    .collect::<Vec<_>>();
        let b = self.b.par_iter()
                    .zip(s_vec.par_iter())
                    .map(|(b, s)| mul!(a, s))
                    .collect::<Vec<_>>();
        Self { a: a, b: b }
    }

    /// split returns the left and right commitment key part. It makes copy.
    /// TODO: remove the copy
    pub fn split(&self, at: usize) -> (Self,Self) {
        let a_l,a_r = self.a.split_at(at);
        let b_l,b_r = self.b.split_at(at);
        (
            Self{a:a_l,b:b_l},
            Self{a:a_r,b:b_r}
        )
    }

    /// Compress takes a left and right commitment key and returns a commitment
    /// key $left \circ right^{scale} = (left_i*right_i^{scale} ...)$. This is
    /// required step during GIPA recursion.
    pub fn compress(left: &Self, right: &Self, scale: &G::Scalar) -> Self {
        let (a,b) = rayon::join(
            || left.a.par_iter().zip(right.a.par_iter()).map(|(left,right)| {
                let mut g = right.mul_assign(scale);
                g.add_assign(left);
                g
            }).collect::<Vec<_>>(),
            || left.b.par_iter().zip(right.b.par_iter()).map(|(left,right)| {
                let mut g = right.mul_assign(scale);
                g.add_assign(left);
                g
            }).collect::<Vec<_>>());
        Self{ a:a, b:b }
    }

    /// first returns the first values in the vector of v1 and v2 (respectively
    /// w1 and w2). When commitment key is of size one, it's a proxy to get the
    /// final values.
    pub fn first(&self) -> (G,G) {
        (self.a[0],self.b[0])
    }
}

/// Both commitment outputs a pair of $F_q^k$ element.
type Output<E: Engine> = (E::Fqk, E::Fqk);

/// single_g1 commits to a single vector of G1 elements in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})$
/// Output is $(T,U)$
pub fn single_g1<E: Engine>(vkey: &VKey<E>, A: &[E::G1]) -> Output<E> {
    let T = inner_product::pairing_miller(A, vkey.a);
    let U = inner_product::pairing_miller(A, vkey.b);
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
    // TODO parralelize that
    let mut T1 = inner_product::pairing_miller(A, vkey.a);
    let T2 = inner_product::pairing_miller(wkey.a, B);
    let mut U1 = inner_product::pairing_miller(A, vkey.b);
    let U2 = inner_product::pairing_miller(wkey.b, B);
    T1.mul_assign(&T2);
    U1.mul_assign(&U2);
    return Output(T1, U1);
}
