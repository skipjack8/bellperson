/// TODO global commitment doc and usage
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use crate::bls::{Engine, PairingCurveAffine};
use crate::groth16::aggregate::accumulator::PairingTuple;
use crate::groth16::aggregate::inner_product;
use crate::groth16::multiscalar::*;

/// Key is a generic commitment key that is instanciated with g and h as basis,
/// and a and b as powers.
#[derive(Clone, Debug)]
pub struct Key<G: CurveAffine> {
    /// Exponent is a
    pub a: Vec<G>,
    /// Exponent is b
    pub b: Vec<G>,
}
/// VKey is a commitment key used by the "single" commitment on G1 values as
/// well as in the "pair" commtitment.
/// It contains $\{h^a^i\}_{i=1}^n$ and $\{h^b^i\}_{i=1}^n$
pub type VKey<E: Engine> = Key<E::G2Affine>;

/// WKey is a commitment key used by the "pair" commitment. Note the sequence of
/// powers starts at $n$ already.
/// It contains $\{g^{a^{n+i}}\}_{i=1}^n$ and $\{g^{b^{n+i}}\}_{i=1}^n$
pub type WKey<E: Engine> = Key<E::G1Affine>;

impl<G> Key<G>
where
    G: CurveAffine,
{
    /// correct_len returns true if both commitment keys have the same size as
    /// the argument. It is necessary for the IPP scheme to work that commitment
    /// key have the exact same number of arguments as the number of proofs to
    /// aggregate.
    pub fn correct_len(&self, n: usize) -> bool {
        self.a.len() == n && self.b.len() == n
    }

    /// scale returns both vectors scaled by the given vector entrywise.
    /// In other words, it returns $\{v_i^{s_i}\}$
    pub fn scale(&self, s_vec: &[G::Scalar]) -> Self {
        let (a, b) = self
            .a
            .par_iter()
            .zip(self.b.par_iter())
            .zip(s_vec.par_iter())
            .map(|((ap, bp), s)| {
                let mut xa = ap.clone();
                let mut xb = bp.clone();
                xa.mul(s.into_repr());
                xb.mul(s.into_repr());
                (xa, xb)
            })
            .unzip();

        Self { a: a, b: b }
    }

    /// split returns the left and right commitment key part. It makes copy.
    /// TODO: remove the copy
    pub fn split(&self, at: usize) -> (Self, Self) {
        let (a_l, a_r) = self.a.split_at(at);
        let (b_l, b_r) = self.b.split_at(at);
        (
            Self {
                a: a_l.to_vec(),
                b: b_l.to_vec(),
            },
            Self {
                a: a_r.to_vec(),
                b: b_r.to_vec(),
            },
        )
    }

    /// Compress takes a left and right commitment key and returns a commitment
    /// key $left \circ right^{scale} = (left_i*right_i^{scale} ...)$. This is
    /// required step during GIPA recursion.
    pub fn compress(left: &Self, right: &Self, scale: &G::Scalar) -> Self {
        assert!(left.a.len() == right.a.len());
        let (a, b): (Vec<G>, Vec<G>) = left
            .a
            .par_iter()
            .zip(left.b.par_iter())
            .zip(right.a.par_iter())
            .zip(right.b.par_iter())
            .map(|(((left_a, left_b), right_a), right_b)| {
                let mut ra = mul!(right_a.into_projective(), scale.into_repr());
                let mut rb = mul!(right_b.into_projective(), scale.into_repr());
                ra.add_assign_mixed(left_a);
                rb.add_assign_mixed(left_b);
                (ra.into_affine(), rb.into_affine())
            })
            .unzip();

        assert!(a.len() == left.a.len());
        assert!(b.len() == left.a.len());
        Self { a: a, b: b }
    }

    /// first returns the first values in the vector of v1 and v2 (respectively
    /// w1 and w2). When commitment key is of size one, it's a proxy to get the
    /// final values.
    pub fn first(&self) -> (G, G) {
        (self.a[0].clone(), self.b[0].clone())
    }
}

/// Both commitment outputs a pair of $F_q^k$ element.
pub type Output<E: Engine> = (E::Fqk, E::Fqk);

/// single_g1 commits to a single vector of G1 elements in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})$
/// Output is $(T,U)$
pub fn single_g1<E: Engine>(vkey: &VKey<E>, A: &[E::G1Affine]) -> Output<E> {
    let T = inner_product::pairing_miller_affine::<E>(A, &vkey.a);
    let U = inner_product::pairing_miller_affine::<E>(A, &vkey.b);
    return (T, U);
}

/// pair commits to a tuple of G1 vector and G2 vector in the following way:
/// $T = \prod_{i=0}^n e(A_i, v_{1,i})e(B_i,w_{1,i})$
/// $U = \prod_{i=0}^n e(A_i, v_{2,i})e(B_i,w_{2,i})$
/// Output is $(T,U)$
pub fn pair<E: Engine>(
    vkey: &VKey<E>,
    wkey: &WKey<E>,
    A: &[E::G1Affine],
    B: &[E::G2Affine],
) -> Output<E> {
    let ((mut T1, T2), (mut U1, U2)) = rayon::join(
        || {
            rayon::join(
                || inner_product::pairing_miller_affine::<E>(A, &vkey.a),
                || inner_product::pairing_miller_affine::<E>(&wkey.a, B),
            )
        },
        || {
            rayon::join(
                || inner_product::pairing_miller_affine::<E>(A, &vkey.b),
                || inner_product::pairing_miller_affine::<E>(&wkey.b, B),
            )
        },
    );
    T1.mul_assign(&T2);
    U1.mul_assign(&U2);
    return (T1, U1);
}
