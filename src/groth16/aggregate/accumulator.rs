use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::CurveAffine;
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
                let na = a.clone();
                na.mul(coeff);
                (a.prepare(), b)
            })
            .unzip();
        let pairs = g1scaled
            .iter()
            .zip(g2.iter())
            .map(|(ar, &b)| (ar, b))
            .collect::<Vec<_>>();
        //let pairs_ref: Vec<_> = pairs.into_iter().map(|(a, b)| (a, b)).collect();
        //let miller_out = E::miller_loop(pairs_ref.iter());
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
