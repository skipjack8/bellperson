use crate::bls::Engine;
use crate::groth16::aggregate::commit;
use groupy::CurveAffine;
use serde::{Deserialize, Serialize};

/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
pub struct AggregateProof<E: Engine> {
    /// commitment to A and B using the pair commitment scheme needed to verify
    /// TIPP relation.
    pub com_ab: commit::Output<E>,
    /// commit to C separate since we use it only in MIPP
    pub com_c: commit::Output<E>,
    /// $A^r * B = Z$ is the left value on the aggregated Groth16 equation
    pub ip_ab: E::Fqk,
    /// $C^r$ is used on the right side of the aggregated Groth16 equation
    pub agg_c: E::G1,
    /// tipp proof for proving correct aggregation of A and B
    pub proof_ab: TIPPProof<E>,
    /// mipp proof for proving correct scaling of C
    pub proof_c: MIPPProof<E>,
}

/// GipaTIPP proof contains all the information necessary to verify the Gipa
/// TIPP statement. All GIPA related elements are given in order of generation
/// by the prover.
pub struct GipaTIPP<E: Engine> {
    /// ((T_L, U_L),(T_R,U_R)) values accross all steps
    pub comms: Vec<(commit::Output<E>, commit::Output<E>)>,
    /// Z values left and right
    pub z_vec: Vec<(E::Fqk, E::Fqk)>,
    /// final values of A and B at the end of the recursion
    pub final_A: E::G1Affine,
    pub final_B: E::G2Affine,
    /// final commitment keys $v$ and $w$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    pub final_vkey: (E::G2Affine, E::G2Affine),
    pub final_wkey: (E::G1Affine, E::G1Affine),
}

/// TIPPProof contains a GIPA proof as well as the opening of the rescaled
/// commitment keys - to let the verifier prove sucinctly the rpvoer has
/// correctly performed the recursion on the commitment keys as well.
pub struct TIPPProof<E: Engine> {
    pub gipa: GipaTIPP<E>,
    pub vkey_opening: KZGOpening<E::G2Affine>,
    pub wkey_opening: KZGOpening<E::G1Affine>,
}

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
pub type KZGOpening<G: CurveAffine> = (G, G);

/// GipaMIPP is similar to GipaTIPP: it contains information to verify the
/// GIPA recursion using the commitment of MIPP. Section 4 of the paper.
pub struct GipaMIPP<E: Engine> {
    /// ((T_L, U_L),(T_R,U_R)) values accross all steps
    pub comms: Vec<(commit::Output<E>, commit::Output<E>)>,
    /// Z values left and right
    pub z_vec: Vec<(E::G1, E::G1)>,
    /// final values of C at the end of the recursion
    pub final_C: E::G1Affine,
    pub final_r: E::Fr,
    /// final commitment keys $v$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    pub final_vkey: (E::G2Affine, E::G2Affine),
}

/// MIPPProof contains the GIPA proof as well as the opening information to be
/// able to verify the correctness of the commitment keys.
pub struct MIPPProof<E: Engine> {
    pub gipa: GipaMIPP<E>,
    pub vkey_opening: KZGOpening<E::G2Affine>,
}
