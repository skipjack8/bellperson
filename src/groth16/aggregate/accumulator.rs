use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rand::thread_rng;

#[cfg(feature = "pairing")]
use paired::{Engine, PairingCurveAffine};

#[cfg(feature = "blst")]
use crate::bls::Engine;
#[cfg(feature = "blst")]
use blstrs::PairingCurveAffine;

/// PairingCheck represents a check of the form e(A,B)e(C,D)... = T. Checks can
/// be aggregated together using random linear combination. The efficiency comes
/// from keeping the results from the miller loop output before proceding to a final
/// exponentiation when verifying if all checks are verified.
/// It is a tuple:
/// - a miller loop result that is to be multiplied by other miller loop results
/// before going into a final exponentiation result
/// - a right side result which is already in the right subgroup Gt which is to
/// be compared to the left side when "final_exponentiatiat"-ed
pub struct PairingCheck<E: Engine>(E::Fqk, E::Fqk);

impl<E> PairingCheck<E>
where
    E: Engine,
{
    pub fn new() -> PairingCheck<E> {
        Self(E::Fqk::one(), E::Fqk::one())
    }

    pub fn new_invalid() -> PairingCheck<E> {
        Self(E::Fqk::one(), E::Fqk::zero())
    }

    pub fn from_pair(result: E::Fqk, exp: E::Fqk) -> PairingCheck<E> {
        Self(result, exp)
    }

    pub fn from_miller_one(result: E::Fqk) -> PairingCheck<E> {
        Self(result, E::Fqk::one())
    }

    /// returns a pairing tuple that is scaled by a random element.
    /// When aggregating pairing checks, this creates a random linear
    /// combination of all checks so that it is secure. Specifically
    /// we have e(A,B)e(C,D)... = out <=> e(g,h)^{ab + cd} = out
    /// We rescale using a random element $r$ to give
    /// e(rA,B)e(rC,D) ... = out^r <=>
    /// e(A,B)^r e(C,D)^r = out^r <=> e(g,h)^{abr + cdr} = out^r
    /// (e(g,h)^{ab + cd})^r = out^r
    pub fn from_miller_inputs<'a, I>(it: I, out: &'a E::Fqk) -> PairingCheck<E>
    where
        I: IntoIterator<Item = &'a (&'a E::G1Affine, &'a E::G2Affine)>,
    {
        let coeff = derive_non_zero::<E>();
        let pairs = it
            .into_iter()
            .map(|&(a, b)| {
                let na = a.mul(coeff).into_affine();
                (na.prepare(), b.prepare())
            })
            .collect::<Vec<_>>();
        let pairs_ref: Vec<_> = pairs.iter().map(|(a, b)| (a, b)).collect();
        let miller_out = E::miller_loop(pairs_ref.iter());
        let mut outt = out.clone();
        if out != &E::Fqk::one() {
            // we only need to make this expensive operation is the output is
            // not one since 1^r = 1
            outt = outt.pow(&coeff.into_repr());
        }
        PairingCheck(miller_out, outt)
    }

    /// takes another pairing tuple and combine both sides together as a random
    /// linear combination.
    pub fn merge(&mut self, p2: &PairingCheck<E>) {
        // multiply miller loop results together
        self.0.mul_assign(&p2.0);
        // multiply  right side in GT together
        self.1.mul_assign(&p2.1);
    }

    pub fn verify(&self) -> bool {
        E::final_exponentiation(&self.0).unwrap() == self.1
    }
}

fn derive_non_zero<E: Engine>() -> E::Fr {
    let mut rng = thread_rng();
    loop {
        let coeff = E::Fr::random(&mut rng);
        if coeff != E::Fr::zero() {
            return coeff;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use groupy::CurveProjective;
    use rand_core::SeedableRng;

    #[cfg(feature = "blst")]
    use crate::bls::{Bls12, G1Projective, G2Projective};

    #[cfg(feature = "pairing")]
    use paired::bls12_381::{Bls12, G1 as G1Projective, G2 as G2Projective};

    #[test]
    fn test_pairing_randomize() {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        let g1r = G1Projective::random(&mut rng);
        let g2r = G2Projective::random(&mut rng);
        let exp = Bls12::pairing(g1r.clone(), g2r.clone());
        let tuple = PairingCheck::<Bls12>::from_miller_inputs(
            &[(&g1r.into_affine(), &g2r.into_affine())],
            &exp,
        );
        assert!(tuple.verify());
    }
}
