use crate::bls::Engine;
use ff::{Field, PrimeField};
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
        // "1" when final exponentiated / in target group, will be equal to 1 !
        Self(E::Fqk::one(), E::Fqk::one())
    }

    pub fn new_invalid() -> PairingTuple<E> {
        Self(E::Fqk::one(), E::Fqk::zero())
    }

    pub fn from_miller(miller: E::Fqk) -> PairingTuple<E> {
        Self(miller, E::Fqk::one())
    }

    pub fn from_pair(miller: E::Fqk, exp: E::Fqk) -> PairingTuple<E> {
        Self(miller, exp)
    }

    /// takes another pairing tuple and combine both sides together as a random
    /// linear combination.
    pub fn merge(&mut self, p2: &PairingTuple<E>) {
        let mut rng = thread_rng();
        let coeff = E::Fr::random(&mut rng);
        p2.0.pow(&coeff.into_repr());
        p2.1.pow(&coeff.into_repr());
        // multiply miller loop results together
        self.0.mul_assign(&p2.0);
        // multiply  right side in GT together
        self.1.mul_assign(&p2.1);
    }

    pub fn verify(&self) -> bool {
        E::final_exponentiation(&self.0).unwrap() == self.1
    }
}
