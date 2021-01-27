use ff::Field;

#[macro_use]
mod macros;

mod accumulator;
mod commit;
mod inner_product;
mod msm;
mod poly;
mod proof;
mod prove;
mod srs;
mod verify;

pub use self::commit::*;
pub use self::proof::*;
pub use self::prove::*;
pub use self::srs::*;
pub use self::verify::*;

/// Returns the vector used for the linear combination fo the inner pairing product
/// between A and B for the Groth16 aggregation: A^r * B. It is required as it
/// is not enough to simply prove the ipp of A*B, we need a random linear
/// combination of those.
fn structured_scalar_power<F: Field>(num: usize, s: &F) -> Vec<F> {
    let mut powers = vec![F::one()];
    for i in 1..num {
        powers.push(mul!(powers[i - 1], s));
    }
    powers
}
