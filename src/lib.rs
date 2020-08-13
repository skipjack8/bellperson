//! `bellperson` is a crate for building zk-SNARK circuits. It provides circuit
//! traits and and primitive structures, as well as basic gadget implementations
//! such as booleans and number abstractions.
//!
//! # Example circuit
//!
//! Say we want to write a circuit that proves we know the preimage to some hash
//! computed using SHA-256d (calling SHA-256 twice). The preimage must have a
//! fixed length known in advance (because the circuit parameters will depend on
//! it), but can otherwise have any value. We take the following strategy:
//!
//! - Witness each bit of the preimage.
//! - Compute `hash = SHA-256d(preimage)` inside the circuit.
//! - Expose `hash` as a public input using multiscalar packing.
//!
//! ```
//! use bellperson::{
//!     gadgets::{
//!         boolean::{AllocatedBit, Boolean},
//!         multipack,
//!         sha256::sha256,
//!     },
//!     groth16, Circuit, ConstraintSystem, SynthesisError,
//! };
//! use paired::{bls12_381::Bls12, Engine};
//! use rand::rngs::OsRng;
//! use sha2::{Digest, Sha256};
//!
//! /// Our own SHA-256d gadget. Input and output are in little-endian bit order.
//! fn sha256d<E: Engine, CS: ConstraintSystem>(
//!     mut cs: CS,
//!     data: &[Boolean],
//! ) -> Result<Vec<Boolean>, SynthesisError> {
//!     // Flip endianness of each input byte
//!     let input: Vec<_> = data
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect();
//!
//!     let mid = sha256(cs.namespace(|| "SHA-256(input)"), &input)?;
//!     let res = sha256(cs.namespace(|| "SHA-256(mid)"), &mid)?;
//!
//!     // Flip endianness of each output byte
//!     Ok(res
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect())
//! }
//!
//! struct MyCircuit {
//!     /// The input to SHA-256d we are proving that we know. Set to `None` when we
//!     /// are verifying a proof (and do not have the witness data).
//!     preimage: Option<[u8; 80]>,
//! }
//!
//! impl<E: Engine> Circuit for MyCircuit {
//!     fn synthesize<CS: ConstraintSystem>(self, cs: &mut CS) -> Result<(), SynthesisError> {
//!         // Compute the values for the bits of the preimage. If we are verifying a proof,
//!         // we still need to create the same constraints, so we return an equivalent-size
//!         // Vec of None (indicating that the value of each bit is unknown).
//!         let bit_values = if let Some(preimage) = self.preimage {
//!             preimage
//!                 .iter()
//!                 .map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
//!                 .flatten()
//!                 .map(|b| Some(b))
//!                 .collect()
//!         } else {
//!             vec![None; 80 * 8]
//!         };
//!         assert_eq!(bit_values.len(), 80 * 8);
//!
//!         // Witness the bits of the preimage.
//!         let preimage_bits = bit_values
//!             .into_iter()
//!             .enumerate()
//!             // Allocate each bit.
//!             .map(|(i, b)| {
//!                 AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {}", i)), b)
//!             })
//!             // Convert the AllocatedBits into Booleans (required for the sha256 gadget).
//!             .map(|b| b.map(Boolean::from))
//!             .collect::<Result<Vec<_>, _>>()?;
//!
//!         // Compute hash = SHA-256d(preimage).
//!         let hash = sha256d(cs.namespace(|| "SHA-256d(preimage)"), &preimage_bits)?;
//!
//!         // Expose the vector of 32 boolean variables as compact public inputs.
//!         multipack::pack_into_inputs(cs.namespace(|| "pack hash"), &hash)
//!     }
//! }
//!
//! // Create parameters for our circuit. In a production deployment these would
//! // be generated securely using a multiparty computation.
//! let params = {
//!     let c = MyCircuit { preimage: None };
//!     groth16::generate_random_parameters::<Bls12, _, _>(c, &mut OsRng).unwrap()
//! };
//!
//! // Prepare the verification key (for proof verification).
//! let pvk = groth16::prepare_verifying_key(&params.vk);
//!
//! // Pick a preimage and compute its hash.
//! let preimage = [42; 80];
//! let hash = Sha256::digest(&Sha256::digest(&preimage));
//!
//! // Create an instance of our circuit (with the preimage as a witness).
//! let c = MyCircuit {
//!     preimage: Some(preimage),
//! };
//!
//! // Create a Groth16 proof with our parameters.
//! let proof = groth16::create_random_proof(c, &params, &mut OsRng).unwrap();
//!
//! // Pack the hash as inputs for proof verification.
//! let hash_bits = multipack::bytes_to_bits_le(&hash);
//! let inputs = multipack::compute_multipacking::<Bls12>(&hash_bits);
//!
//! // Check the proof!
//! assert!(groth16::verify_proof(&pvk, &proof, &inputs).unwrap());
//! ```
//!
//! # Roadmap
//!
//! `bellperson` is being refactored into a generic proving library. Currently it
//! is pairing-specific, and different types of proving systems need to be
//! implemented as sub-modules. After the refactor, `bellperson` will be generic
//! using the [`ff`] and [`group`] crates, while specific proving systems will
//! be separate crates that pull in the dependencies they require.

// Catch documentation errors caused by code changes.
#![deny(intra_doc_link_resolution_failure)]

#[cfg(test)]
#[macro_use]
extern crate hex_literal;

pub mod domain;
// FIXME
// pub mod gadgets;
pub mod gpu;
#[cfg(feature = "groth16")]
pub mod groth16;
pub mod multicore;
pub mod multiexp;

// FIXME
// pub mod util_cs;

use ahash::AHashMap as HashMap;
use std::io;
use std::ops::{Add, Sub};

use blstrs::*;

const BELLMAN_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Computations are expressed in terms of arithmetic circuits, in particular
/// rank-1 quadratic constraint systems. The `Circuit` trait represents a
/// circuit that can be synthesized. The `synthesize` method is called during
/// CRS generation and during proving.
pub trait Circuit {
    /// Synthesize the circuit into a rank-1 quadratic constraint system.
    fn synthesize<CS: ConstraintSystem>(self, cs: &mut CS) -> Result<(), SynthesisError>;
}

/// Represents a variable in our constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable(Index);

impl Variable {
    /// This constructs a variable with an arbitrary index.
    /// Circuit implementations are not recommended to use this.
    pub fn new_unchecked(idx: Index) -> Variable {
        Variable(idx)
    }

    /// This returns the index underlying the variable.
    /// Circuit implementations are not recommended to use this.
    pub fn get_unchecked(&self) -> Index {
        self.0
    }
}

/// Represents the index of either an input variable or
/// auxiliary variable.
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

/// This represents a linear combination of some variables, with coefficients
/// in the scalar field of a pairing-friendly elliptic curve group.
#[derive(Clone)]
pub struct LinearCombination(HashMap<Variable, Scalar>);
impl Default for LinearCombination {
    fn default() -> Self {
        Self::zero()
    }
}

impl LinearCombination {
    pub fn zero() -> LinearCombination {
        LinearCombination(HashMap::new())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Variable, &Scalar)> + '_ {
        self.0.iter()
    }

    pub fn add_unsimplified(mut self, (coeff, var): (Scalar, Variable)) -> LinearCombination {
        *self.0.entry(var).or_insert(Scalar::zero()) += &coeff;

        self
    }
}

impl Add<(Scalar, Variable)> for LinearCombination {
    type Output = LinearCombination;

    fn add(mut self, (coeff, var): (Scalar, Variable)) -> LinearCombination {
        *self.0.entry(var).or_insert(Scalar::zero()) += &coeff;

        self
    }
}

impl Sub<(Scalar, Variable)> for LinearCombination {
    type Output = LinearCombination;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, (mut coeff, var): (Scalar, Variable)) -> LinearCombination {
        coeff = -coeff;

        self + (coeff, var)
    }
}

impl Add<Variable> for LinearCombination {
    type Output = LinearCombination;

    fn add(self, other: Variable) -> LinearCombination {
        self + (Scalar::one(), other)
    }
}

impl Sub<Variable> for LinearCombination {
    type Output = LinearCombination;

    fn sub(self, other: Variable) -> LinearCombination {
        self - (Scalar::one(), other)
    }
}

impl<'a> Add<&'a LinearCombination> for LinearCombination {
    type Output = LinearCombination;

    fn add(mut self, other: &'a LinearCombination) -> LinearCombination {
        for (var, val) in &other.0 {
            *self.0.entry(*var).or_insert(Scalar::zero()) += val;
        }

        self
    }
}

impl<'a> Sub<&'a LinearCombination> for LinearCombination {
    type Output = LinearCombination;

    fn sub(mut self, other: &'a LinearCombination) -> LinearCombination {
        for (var, val) in &other.0 {
            self = self - (*val, *var);
        }

        self
    }
}

impl<'a> Add<(Scalar, &'a LinearCombination)> for LinearCombination {
    type Output = LinearCombination;

    fn add(mut self, (coeff, other): (Scalar, &'a LinearCombination)) -> LinearCombination {
        for s in &other.0 {
            let mut tmp = *s.1;
            tmp *= &coeff;
            self = self + (tmp, *s.0);
        }

        self
    }
}

impl<'a> Sub<(Scalar, &'a LinearCombination)> for LinearCombination {
    type Output = LinearCombination;

    fn sub(mut self, (coeff, other): (Scalar, &'a LinearCombination)) -> LinearCombination {
        for s in &other.0 {
            let mut tmp = *s.1;
            tmp *= &coeff;
            self = self - (tmp, *s.0);
        }

        self
    }
}

/// This is an error that could occur during circuit synthesis contexts,
/// such as CRS generation, proving or verification.
#[derive(thiserror::Error, Debug)]
pub enum SynthesisError {
    /// During synthesis, we lacked knowledge of a variable assignment.
    #[error("an assignment for a variable could not be computed")]
    AssignmentMissing,
    /// During synthesis, we divided by zero.
    #[error("division by zero")]
    DivisionByZero,
    /// During synthesis, we constructed an unsatisfiable constraint system.
    #[error("unsatisfiable constraint system")]
    Unsatisfiable,
    /// During synthesis, our polynomials ended up being too high of degree
    #[error("polynomial degree is too large")]
    PolynomialDegreeTooLarge,
    /// During proof generation, we encountered an identity in the CRS
    #[error("encountered an identity element in the CRS")]
    UnexpectedIdentity,
    /// During proof generation, we encountered an I/O error with the CRS
    #[error("encountered an I/O error: {0}")]
    IoError(#[from] io::Error),
    /// During verification, our verifying key was malformed.
    #[error("malformed verifying key")]
    MalformedVerifyingKey,
    /// During CRS generation, we observed an unconstrained auxiliary variable
    #[error("auxiliary variable was unconstrained")]
    UnconstrainedVariable,
    /// During GPU multiexp/fft, some GPU related error happened
    #[error("encountered a GPU error: {0}")]
    GPUError(#[from] gpu::GPUError),
}

/// Represents a constraint system which can have new variables
/// allocated and constrains between them formed.
pub trait ConstraintSystem: Sized + Send {
    /// Represents the type of the "root" of this constraint system
    /// so that nested namespaces can minimize indirection.
    type Root: ConstraintSystem;

    fn new() -> Self {
        unimplemented!(
            "ConstraintSystem::new must be implemented for extensible types implementing ConstraintSystem"
        );
    }

    /// Return the "one" input variable
    fn one() -> Variable {
        Variable::new_unchecked(Index::Input(0))
    }

    /// Allocate a private variable in the constraint system. The provided function is used to
    /// determine the assignment of the variable. The given `annotation` function is invoked
    /// in testing contexts in order to derive a unique name for this variable in the current
    /// namespace.
    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    /// Allocate a public variable in the constraint system. The provided function is used to
    /// determine the assignment of the variable.
    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    /// Enforce that `A` * `B` = `C`. The `annotation` function is invoked in testing contexts
    /// in order to derive a unique name for the constraint in the current namespace.
    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination) -> LinearCombination,
        LB: FnOnce(LinearCombination) -> LinearCombination,
        LC: FnOnce(LinearCombination) -> LinearCombination;

    /// Create a new (sub)namespace and enter into it. Not intended
    /// for downstream use; use `namespace` instead.
    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR;

    /// Exit out of the existing namespace. Not intended for
    /// downstream use; use `namespace` instead.
    fn pop_namespace(&mut self);

    /// Gets the "root" constraint system, bypassing the namespacing.
    /// Not intended for downstream use; use `namespace` instead.
    fn get_root(&mut self) -> &mut Self::Root;

    /// Begin a namespace for this constraint system.
    fn namespace<NR, N>(&mut self, name_fn: N) -> Namespace<'_, Self::Root>
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.get_root().push_namespace(name_fn);

        Namespace(self.get_root())
    }

    /// Most implementations of ConstraintSystem are not 'extensible': they won't implement a specialized
    /// version of `extend` and should therefore also keep the default implementation of `is_extensible`
    /// so callers which optionally make use of `extend` can know to avoid relying on it when unimplemented.
    fn is_extensible() -> bool {
        false
    }

    /// Extend concatenates thew  `other` constraint systems to the receiver, modifying the receiver, whose
    /// inputs, allocated variables, and constraints will precede those of the `other` constraint system.
    /// The primary use case for this is parallel synthesis of circuits which can be decomposed into
    /// entirely independent sub-circuits. Each can be synthesized in its own thread, then the
    /// original `ConstraintSystem` can be extended with each, in the same order they would have
    /// been synthesized sequentially.
    fn extend(&mut self, _other: Self) {
        unimplemented!(
            "ConstraintSystem::extend must be implemented for types implementing ConstraintSystem"
        );
    }
}

/// This is a "namespaced" constraint system which borrows a constraint system (pushing
/// a namespace context) and, when dropped, pops out of the namespace context.
pub struct Namespace<'a, CS: ConstraintSystem>(&'a mut CS);

impl<'cs, CS: ConstraintSystem> ConstraintSystem for Namespace<'cs, CS> {
    type Root = CS::Root;

    fn one() -> Variable {
        CS::one()
    }

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.0.alloc(annotation, f)
    }

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.0.alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination) -> LinearCombination,
        LB: FnOnce(LinearCombination) -> LinearCombination,
        LC: FnOnce(LinearCombination) -> LinearCombination,
    {
        self.0.enforce(annotation, a, b, c)
    }

    // Downstream users who use `namespace` will never interact with these
    // functions and they will never be invoked because the namespace is
    // never a root constraint system.

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        panic!("only the root's push_namespace should be called");
    }

    fn pop_namespace(&mut self) {
        panic!("only the root's pop_namespace should be called");
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self.0.get_root()
    }
}

impl<'a, CS: ConstraintSystem> Drop for Namespace<'a, CS> {
    fn drop(&mut self) {
        self.get_root().pop_namespace()
    }
}

/// Convenience implementation of ConstraintSystem for mutable references to
/// constraint systems.
impl<'cs, CS: ConstraintSystem> ConstraintSystem for &'cs mut CS {
    type Root = CS::Root;

    fn one() -> Variable {
        CS::one()
    }

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        (**self).alloc(annotation, f)
    }

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<Scalar, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        (**self).alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination) -> LinearCombination,
        LB: FnOnce(LinearCombination) -> LinearCombination,
        LC: FnOnce(LinearCombination) -> LinearCombination,
    {
        (**self).enforce(annotation, a, b, c)
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        (**self).push_namespace(name_fn)
    }

    fn pop_namespace(&mut self) {
        (**self).pop_namespace()
    }

    fn get_root(&mut self) -> &mut Self::Root {
        (**self).get_root()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add_simplify() {
        let n = 5;

        let mut lc = LinearCombination::zero();

        let mut expected_sums = vec![Scalar::zero(); n];
        let mut total_additions = 0;
        for i in 0..n {
            for _ in 0..i + 1 {
                let coeff = Scalar::one();
                lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
                let mut tmp = expected_sums[i];
                tmp += &coeff;
                expected_sums[i] = tmp;
                total_additions += 1;
            }
        }

        // There are only as many terms as distinct variable Indexes — not one per addition operation.
        assert_eq!(n, lc.0.len());
        assert!(lc.0.len() != total_additions);

        // Each variable has the expected coefficient, the sume of those added by its Index.
        lc.0.iter().for_each(|(var, coeff)| match var.0 {
            Index::Aux(i) => assert_eq!(expected_sums[i], *coeff),
            _ => panic!("unexpected variable type"),
        });
    }
}
