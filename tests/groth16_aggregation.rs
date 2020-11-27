use std::time::{Duration, Instant};

use bellperson::bls::{Bls12, Engine, Fr, FrRepr};
use bellperson::gadgets::num::AllocatedNum;
use bellperson::groth16::{
    aggregate_proofs, create_random_proof, generate_random_parameters, prepare_verifying_key,
    setup_inner_product, verify_aggregate_proof, verify_proof, verify_proofs_batch,
};
use bellperson::{Circuit, ConstraintSystem, SynthesisError};
use blake2::Blake2b;
use ff::{Field, PrimeField, ScalarEngine};
use rand::{thread_rng, SeedableRng};

const MIMC_ROUNDS: usize = 322;

/// This is an implementation of MiMC, specifically a
/// variant named `LongsightF322p3` for BLS12-381.
/// See http://eprint.iacr.org/2016/492 for more
/// information about this construction.
///
/// ```
/// function LongsightF322p3(xL ⦂ Fp, xR ⦂ Fp) {
///     for i from 0 up to 321 {
///         xL, xR := xR + (xL + Ci)^3, xL
///     }
///     return xL
/// }
/// ```
fn mimc<E: Engine>(mut xl: E::Fr, mut xr: E::Fr, constants: &[E::Fr]) -> E::Fr {
    assert_eq!(constants.len(), MIMC_ROUNDS);

    for i in 0..MIMC_ROUNDS {
        let mut tmp1 = xl;
        tmp1.add_assign(&constants[i]);
        let mut tmp2 = tmp1;
        tmp2.square();
        tmp2.mul_assign(&tmp1);
        tmp2.add_assign(&xr);
        xr = xl;
        xl = tmp2;
    }

    xl
}

#[derive(Clone)]
struct MiMCDemo<'a, E: Engine> {
    xl: Option<E::Fr>,
    xr: Option<E::Fr>,
    constants: &'a [E::Fr],
}

impl<'a, E: Engine> Circuit<E> for MiMCDemo<'a, E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        assert_eq!(self.constants.len(), MIMC_ROUNDS);

        // Allocate the first component of the preimage.
        let mut xl_value = self.xl;
        let mut xl = cs.alloc(
            || "preimage xl",
            || xl_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        // Allocate the second component of the preimage.
        let mut xr_value = self.xr;
        let mut xr = cs.alloc(
            || "preimage xr",
            || xr_value.ok_or(SynthesisError::AssignmentMissing),
        )?;

        for i in 0..MIMC_ROUNDS {
            // xL, xR := xR + (xL + Ci)^3, xL
            let cs = &mut cs.namespace(|| format!("round {}", i));

            // tmp = (xL + Ci)^2
            let tmp_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.square();
                e
            });
            let tmp = cs.alloc(
                || "tmp",
                || tmp_value.ok_or(SynthesisError::AssignmentMissing),
            )?;

            cs.enforce(
                || "tmp = (xL + Ci)^2",
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + tmp,
            );

            // new_xL = xR + (xL + Ci)^3
            // new_xL = xR + tmp * (xL + Ci)
            // new_xL - xR = tmp * (xL + Ci)
            let new_xl_value = xl_value.map(|mut e| {
                e.add_assign(&self.constants[i]);
                e.mul_assign(&tmp_value.unwrap());
                e.add_assign(&xr_value.unwrap());
                e
            });

            let new_xl = if i == (MIMC_ROUNDS - 1) {
                // This is the last round, xL is our image and so
                // we allocate a public input.
                cs.alloc_input(
                    || "image",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            } else {
                cs.alloc(
                    || "new_xl",
                    || new_xl_value.ok_or(SynthesisError::AssignmentMissing),
                )?
            };

            cs.enforce(
                || "new_xL = xR + (xL + Ci)^3",
                |lc| lc + tmp,
                |lc| lc + xl + (self.constants[i], CS::one()),
                |lc| lc + new_xl - xr,
            );

            // xR = xL
            xr = xl;
            xr_value = xl_value;

            // xL = new_xL
            xl = new_xl;
            xl_value = new_xl_value;
        }

        Ok(())
    }
}

#[derive(Clone)]
struct TestCircuit<E: Engine> {
    public_inputs: Vec<Option<E::Fr>>,
    witness_input: Option<E::Fr>,
    public_product: Option<E::Fr>,
}
impl<E: Engine> Circuit<E> for TestCircuit<E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, mut cs: &mut CS) -> Result<(), SynthesisError> {
        let input_variables: Vec<_> = self
            .public_inputs
            .iter()
            .enumerate()
            .map(|(_i, input)| -> Result<AllocatedNum<_>, SynthesisError> {
                let num = AllocatedNum::alloc(&mut cs, || {
                    input.ok_or(SynthesisError::AssignmentMissing)
                })?;
                num.inputize(&mut cs)?;
                Ok(num)
            })
            .collect::<Result<_, _>>()?;
        let product = AllocatedNum::alloc(&mut cs, || {
            self.public_product.ok_or(SynthesisError::AssignmentMissing)
        })?;
        product.inputize(&mut cs)?;
        let witness = AllocatedNum::alloc(&mut cs, || {
            self.witness_input.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let mut computed_product = witness;
        for x in &input_variables {
            computed_product = computed_product.mul(&mut cs, x)?;
        }
        cs.enforce(
            || "product = computed product",
            |lc| lc + CS::one(),
            |lc| lc + computed_product.get_variable(),
            |lc| lc + product.get_variable(),
        );

        Ok(())
    }
}

#[test]
fn test_groth16_aggregation_min() {
    const NUM_PUBLIC_INPUTS: usize = 2;
    const NUM_PROOFS_TO_AGGREGATE: usize = 2; //1024;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    println!("Creating parameters...");

    // Generate parameters for inner product aggregation
    let srs = setup_inner_product(&mut rng, NUM_PROOFS_TO_AGGREGATE);

    // Create parameters for our circuit
    let params = {
        let c = TestCircuit::<Bls12> {
            public_inputs: vec![Default::default(); NUM_PUBLIC_INPUTS],
            public_product: Default::default(),
            witness_input: Default::default(),
        };

        generate_random_parameters(c, &mut rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    println!("Creating proofs...");

    // Generate proofs
    println!("Generating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);

    let mut proofs = Vec::new();
    let mut statements = Vec::new();
    let mut generation_time = Duration::new(0, 0);
    let mut individual_verification_time = Duration::new(0, 0);

    for _ in 0..NUM_PROOFS_TO_AGGREGATE {
        // Generate random inputs to product together
        let mut public_inputs = Vec::new();
        let mut statement = Vec::new();
        let mut prod = Fr::one();
        for _i in 0..NUM_PUBLIC_INPUTS {
            let x = Fr::from_str("4").unwrap();
            public_inputs.push(Some(x));
            statement.push(x);
            prod.mul_assign(&x);
        }
        let w = Fr::from_repr(FrRepr::from(3)).unwrap();

        let mut product: Fr = w.clone();
        product.mul_assign(&prod);
        statement.push(product.clone());

        let start = Instant::now();
        // Create an instance of our circuit (with the
        // witness)
        let c = TestCircuit {
            public_inputs,
            public_product: Some(product),
            witness_input: Some(w),
        };

        // Create a groth16 proof with our parameters.
        let proof = create_random_proof(c, &params, &mut rng).unwrap();
        generation_time += start.elapsed();

        assert!(verify_proof(&pvk, &proof, &statement).unwrap());
        individual_verification_time += start.elapsed();

        proofs.push(proof);
        statements.push(statement);
    }

    // Aggregate proofs using inner product proofs
    let start = Instant::now();
    println!("Aggregating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);
    let aggregate_proof = aggregate_proofs::<Bls12, Blake2b>(&srs, &proofs);
    let prover_time = start.elapsed().as_millis();

    println!("Verifying aggregated proof...");
    let start = Instant::now();
    let result = verify_aggregate_proof(
        &srs.get_verifier_key(),
        &params.vk,
        &statements,
        &aggregate_proof,
    );
    let verifier_time = start.elapsed().as_millis();
    assert!(result);

    println!("Proof generation time: {} ms", generation_time.as_millis());
    println!("Proof aggregation time: {} ms", prover_time);
    println!("Proof aggregation verification time: {} ms", verifier_time);
    println!(
        "Proof individual verification time: {} ms",
        individual_verification_time.as_millis()
    );
}

#[test]
fn test_groth16_aggregation_mimc() {
    const NUM_PROOFS_TO_AGGREGATE: usize = 1024;
    let rng = &mut thread_rng();

    // Generate the MiMC round constants
    let constants = (0..MIMC_ROUNDS)
        .map(|_| <Bls12 as ScalarEngine>::Fr::random(rng))
        .collect::<Vec<_>>();

    println!("Creating parameters...");

    // Create parameters for our circuit
    let params = {
        let c = MiMCDemo::<Bls12> {
            xl: None,
            xr: None,
            constants: &constants,
        };

        generate_random_parameters(c, rng).unwrap()
    };

    // Prepare the verification key (for proof verification)
    let pvk = prepare_verifying_key(&params.vk);

    // Generate parameters for inner product aggregation
    let srs = setup_inner_product(rng, NUM_PROOFS_TO_AGGREGATE);

    println!("Creating proofs...");

    // Generate proofs
    println!("Generating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);

    let mut proofs = Vec::new();
    let mut images = Vec::new();
    let mut generation_time = Duration::new(0, 0);
    let mut individual_verification_time = Duration::new(0, 0);

    for _ in 0..NUM_PROOFS_TO_AGGREGATE {
        // Generate a random preimage and compute the image
        let xl = <Bls12 as ScalarEngine>::Fr::random(rng);
        let xr = <Bls12 as ScalarEngine>::Fr::random(rng);
        let image = mimc::<Bls12>(xl, xr, &constants);

        let start = Instant::now();
        // Create an instance of our circuit (with the
        // witness)
        let c = MiMCDemo {
            xl: Some(xl),
            xr: Some(xr),
            constants: &constants,
        };

        // Create a groth16 proof with our parameters.
        let proof = create_random_proof(c, &params, rng).unwrap();
        generation_time += start.elapsed();

        assert!(verify_proof(&pvk, &proof, &[image]).unwrap());
        individual_verification_time += start.elapsed();

        proofs.push(proof);
        images.push(vec![image]);
    }

    // Aggregate proofs using inner product proofs
    let start = Instant::now();
    println!("Aggregating {} Groth16 proofs...", NUM_PROOFS_TO_AGGREGATE);
    let aggregate_proof = aggregate_proofs::<Bls12, Blake2b>(&srs, &proofs);
    let prover_time = start.elapsed().as_millis();

    println!("Verifying aggregated proof...");
    let start = Instant::now();
    let result = verify_aggregate_proof(
        &srs.get_verifier_key(),
        &params.vk,
        &images,
        &aggregate_proof,
    );
    let verifier_time = start.elapsed().as_millis();
    assert!(result);

    let start = Instant::now();
    let proofs: Vec<_> = proofs.iter().collect();
    assert!(verify_proofs_batch(&pvk, rng, &proofs, &images).unwrap());
    let batch_verifier_time = start.elapsed().as_millis();

    println!("Proof generation time: {} ms", generation_time.as_millis());
    println!("Proof aggregation time: {} ms", prover_time);
    println!("Proof aggregation verification time: {} ms", verifier_time);
    println!(
        "Proof individual verification time: {} ms",
        individual_verification_time.as_millis()
    );

    println!("Proof batch verification time: {} ms", batch_verifier_time);
}
