use crate::bls::{Engine, PairingCurveAffine};
use ff::Field;

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
        return Self(E::Fqk::one(), E::Fqk::one());
    }

    pub fn new_invalid() -> PairingTuple<E> {
        return Self(E::Fqk::one(), E::Fqk::zero());
    }

    pub fn from_miller(miller: E::Fqk) -> PairingTuple<E> {
        return Self(miller, E::Fqk::one());
    }

    pub fn from_pair(miller: E::Fqk, exp: E::Fqk) -> PairingTuple<E> {
        return Self(miller, exp);
    }

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
