use super::{create_proof_batch_priority, create_random_proof_batch_priority};
use super::{ParameterSource, Proof};
use crate::{Circuit, SynthesisError};

use rand_core::RngCore;

use blstrs::*;

pub fn create_proof<C, P: ParameterSource>(
    circuit: C,
    params: P,
    r: Scalar,
    s: Scalar,
) -> Result<Proof, SynthesisError>
where
    C: Circuit + Send,
{
    let proofs =
        create_proof_batch_priority::<C, P>(vec![circuit], params, vec![r], vec![s], false)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_random_proof<C, R, P: ParameterSource>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof, SynthesisError>
where
    C: Circuit + Send,
    R: RngCore,
{
    let proofs = create_random_proof_batch_priority::<C, R, P>(vec![circuit], params, rng, false)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_proof_batch<C, P: ParameterSource>(
    circuits: Vec<C>,
    params: P,
    r: Vec<Scalar>,
    s: Vec<Scalar>,
) -> Result<Vec<Proof>, SynthesisError>
where
    C: Circuit + Send,
{
    create_proof_batch_priority::<C, P>(circuits, params, r, s, false)
}

pub fn create_random_proof_batch<C, R, P: ParameterSource>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
) -> Result<Vec<Proof>, SynthesisError>
where
    C: Circuit + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<C, R, P>(circuits, params, rng, false)
}

pub fn create_proof_in_priority<C, P: ParameterSource>(
    circuit: C,
    params: P,
    r: Scalar,
    s: Scalar,
) -> Result<Proof, SynthesisError>
where
    C: Circuit + Send,
{
    let proofs =
        create_proof_batch_priority::<C, P>(vec![circuit], params, vec![r], vec![s], true)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_random_proof_in_priority<C, R, P: ParameterSource>(
    circuit: C,
    params: P,
    rng: &mut R,
) -> Result<Proof, SynthesisError>
where
    C: Circuit + Send,
    R: RngCore,
{
    let proofs = create_random_proof_batch_priority::<C, R, P>(vec![circuit], params, rng, true)?;
    Ok(proofs.into_iter().next().unwrap())
}

pub fn create_proof_batch_in_priority<C, P: ParameterSource>(
    circuits: Vec<C>,
    params: P,
    r: Vec<Scalar>,
    s: Vec<Scalar>,
) -> Result<Vec<Proof>, SynthesisError>
where
    C: Circuit + Send,
{
    create_proof_batch_priority::<C, P>(circuits, params, r, s, true)
}

pub fn create_random_proof_batch_in_priority<C, R, P: ParameterSource>(
    circuits: Vec<C>,
    params: P,
    rng: &mut R,
) -> Result<Vec<Proof>, SynthesisError>
where
    C: Circuit + Send,
    R: RngCore,
{
    create_random_proof_batch_priority::<C, R, P>(circuits, params, rng, true)
}
