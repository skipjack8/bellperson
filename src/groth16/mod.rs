//! The [Groth16] proving system.
//!
//! [Groth16]: https://eprint.iacr.org/2016/260

use std::io::{self, Read, Write};

#[cfg(test)]
mod tests;

mod ext;
mod generator;
mod mapped_params;
mod params;
mod prover;
mod verifier;
mod verifying_key;

pub use self::ext::*;
pub use self::generator::*;
pub use self::mapped_params::*;
pub use self::prover::*;
pub use self::verifier::*;
pub use self::verifying_key::*;
pub use params::*;

use blstrs::*;

#[derive(Clone, Debug)]
pub struct Proof {
    pub a: G1Affine,
    pub b: G2Affine,
    pub c: G1Affine,
}

impl PartialEq for Proof {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl Proof {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.a.to_compressed().as_ref())?;
        writer.write_all(self.b.to_compressed().as_ref())?;
        writer.write_all(self.c.to_compressed().as_ref())?;

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = [0u8; G1Affine::compressed_size()];
        let mut g2_repr = [0u8; G2Affine::compressed_size()];

        reader.read_exact(g1_repr.as_mut())?;
        let a = G1Affine::from_compressed(&g1_repr)
            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "invalid"))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g2_repr.as_mut())?;
        let b = G2Affine::from_compressed(&g2_repr)
            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "invalid"))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        reader.read_exact(g1_repr.as_mut())?;
        let c = G1Affine::from_compressed(&g1_repr)
            .ok_or(io::Error::new(io::ErrorKind::InvalidData, "invalid"))
            .and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

        Ok(Proof { a, b, c })
    }
}

#[cfg(test)]
mod test_with_bls12_381 {
    use super::*;
    use crate::{Circuit, ConstraintSystem, SynthesisError};

    use rand::thread_rng;

    #[test]
    fn serialization() {
        struct MySillyCircuit {
            a: Option<Scalar>,
            b: Option<Scalar>,
        }

        impl Circuit for MySillyCircuit {
            fn synthesize<CS: ConstraintSystem>(self, cs: &mut CS) -> Result<(), SynthesisError> {
                let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
                let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
                let c = cs.alloc_input(
                    || "c",
                    || {
                        let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
                        let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

                        a.mul_assign(&b);
                        Ok(a)
                    },
                )?;

                cs.enforce(|| "a*b=c", |lc| lc + a, |lc| lc + b, |lc| lc + c);

                Ok(())
            }
        }

        let rng = &mut thread_rng();

        let params =
            generate_random_parameters::<_, _>(MySillyCircuit { a: None, b: None }, rng).unwrap();

        {
            let mut v = vec![];

            params.write(&mut v).unwrap();
            assert_eq!(v.len(), 2136);

            let de_params = Parameters::read(&v[..], true).unwrap();
            assert!(params == de_params);

            let de_params = Parameters::read(&v[..], false).unwrap();
            assert!(params == de_params);
        }

        let pvk = prepare_verifying_key(&params.vk);

        for _ in 0..100 {
            let a = Scalar::random(rng);
            let b = Scalar::random(rng);
            let mut c = a;
            c.mul_assign(&b);

            let proof = create_random_proof(
                MySillyCircuit {
                    a: Some(a),
                    b: Some(b),
                },
                &params,
                rng,
            )
            .unwrap();

            let mut v = vec![];
            proof.write(&mut v).unwrap();

            assert_eq!(v.len(), 192);

            let de_proof = Proof::read(&v[..]).unwrap();
            assert!(proof == de_proof);

            assert!(verify_proof(&pvk, &proof, &[c]).unwrap());
            assert!(!verify_proof(&pvk, &proof, &[a]).unwrap());
        }
    }
}
