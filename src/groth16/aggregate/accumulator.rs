use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use paired::PairingCurveAffine;
use rand::thread_rng;

/// PairingTuple is an alias to a pair of
/// - a miller loop result that is to be multiplied by other miller loop results
/// before going into a final exponentiation result
/// - a right side result which is already in the right subgroup Gt which is to
/// be compared to the left side when "final_exponentiatiat"-ed
pub struct PairingTuple<E: Engine>(E::Fqk, E::Fqk);

impl<E> PairingTuple<E>
where
    E: Engine,
{
    pub fn new() -> PairingTuple<E> {
        Self(E::Fqk::one(), E::Fqk::one())
    }

    pub fn new_invalid() -> PairingTuple<E> {
        Self(E::Fqk::one(), E::Fqk::zero())
    }

    pub fn from_pair(result: E::Fqk, exp: E::Fqk) -> PairingTuple<E> {
        Self(result, exp)
    }

    pub fn from_miller_one(result: E::Fqk) -> PairingTuple<E> {
        Self(result, E::Fqk::one())
    }

    /// returns a pairing tuple that is scaled by a random element. Specifically
    /// we have e(A,B)e(C,D)... = out <=> e(g,h)^{ab + cd} = out
    /// We rescale using a random element $r$ to give
    /// e(rA,B)e(rC,D) ... = out^r <=>
    /// e(A,B)^r e(C,D)^r = out^r <=> e(g,h)^{abr + cdr} = out^r
    /// (e(g,h)^{ab + cd})^r = out^r
    pub fn from_miller_inputs<'a, I>(it: I, out: &'a E::Fqk) -> PairingTuple<E>
    where
        I: IntoIterator<
            Item = &'a (
                &'a E::G1Affine,
                &'a <E::G2Affine as PairingCurveAffine>::Prepared,
            ),
        >,
    {
        let mut rng = thread_rng();
        let coeff = E::Fr::random(&mut rng);
        let (g1scaled, g2): (Vec<_>, Vec<_>) = it
            .into_iter()
            .map(|&(a, b)| {
                let na = a.mul(coeff).into_affine();
                (na.prepare(), b)
            })
            .unzip();
        let pairs = g1scaled
            .iter()
            .zip(g2.iter())
            .map(|(ar, &b)| (ar, b))
            .collect::<Vec<_>>();
        let miller_out = E::miller_loop(pairs.iter());
        let mut outt = out.clone();
        if out != &E::Fqk::one() {
            outt = outt.pow(&coeff.into_repr());
        }
        PairingTuple(miller_out, outt)
    }

    /// takes another pairing tuple and combine both sides together as a random
    /// linear combination.
    pub fn merge(&mut self, p2: &PairingTuple<E>) {
        // multiply miller loop results together
        self.0.mul_assign(&p2.0);
        // multiply  right side in GT together
        self.1.mul_assign(&p2.1);
    }

    pub fn verify(&self) -> bool {
        E::final_exponentiation(&self.0).unwrap() == self.1
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bls::{Bls12, Fr, G1Projective, G2Projective};
    use crate::groth16::aggregate::structured_generators_scalar_power;
    use ff::Field;
    use groupy::CurveProjective;
    use rand_core::SeedableRng;

    #[test]
    fn test_pairing_randomize() {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let g1r = G1Projective::random(&mut rng);
        let g2r = G2Projective::random(&mut rng);
        let exp = Bls12::pairing(g1r.clone(), g2r.clone());
        let tuple = PairingTuple::<Bls12>::from_miller_inputs(
            &[(&g1r.into_affine(), &g2r.into_affine().prepare())],
            &exp,
        );
        assert!(tuple.verify());
    }
}
