use serde::{Deserialize, Serialize};

use crate::bls::Engine;
use crate::groth16::aggregagte::commit;

/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
#[derive(Serialize, Deserialize)]
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
    pub final_A: E::G1,
    pub final_B: E::G2,
    /// final commitment keys $v$ and $w$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    pub final_vkey: (E::G2, E::G2),
    pub final_wkey: (E::G1, E::G1),
}

/// TIPPProof contains a GIPA proof as well as the opening of the rescaled
/// commitment keys - to let the verifier prove sucinctly the rpvoer has
/// correctly performed the recursion on the commitment keys as well.
pub struct TIPPProof<E: Engine> {
    pub gipa: GipaTIPP<E>,
    pub vkey_opening: KZGOpening<E::G2>,
    pub wkey_opening: KZGOpening<E::G1>,
}

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
type KZGOpening<G: CurveProjective> = (G, G);

/// GipaMIPP is similar to GipaTIPP: it contains information to verify the
/// GIPA recursion using the commitment of MIPP. Section 4 of the paper.
pub struct GipaMIPP<E: Engine> {
    /// ((T_L, U_L),(T_R,U_R)) values accross all steps
    pub comms: Vec<(commit::Output<E>, commit::Output<E>)>,
    /// Z values left and right
    pub z_vec: Vec<(E::Fqk, E::Fqk)>,
    /// final values of C at the end of the recursion
    pub final_C: E::G1,
    pub final_r: E::Fr,
    /// final commitment keys $v$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    pub final_vkey: (E::G2, E::G2),
}

/// MIPPProof contains the GIPA proof as well as the opening information to be
/// able to verify the correctness of the commitment keys.
pub struct MIPPProof<E: Engine> {
    pub gipa: GipaMIPP<E>,
    pub vkey_opening: KZGOpening<E::G2>,
}

#[derive(Serialize, Deserialize)]
pub struct PairingInnerProductABProof<E: Engine> {
    #[serde(bound(
        serialize = "GIPAProof<E>: Serialize",
        deserialize = "GIPAProof<E>: Deserialize<'de>",
    ))]
    pub gipa_proof: GIPAProof<E>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub final_ck: (E::G2, E::G1),
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub final_ck_proof: (E::G2, E::G1),
}

#[derive(Serialize, Deserialize)]
pub struct GIPAProof<E: Engine> {
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fr: Serialize,E::G1: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fr: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub r_commitment_steps: Vec<((E::Fqk, E::Fqk, E::Fqk), (E::Fqk, E::Fqk, E::Fqk))>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G2: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G2: Deserialize<'de>",
    ))]
    pub r_base: (E::G1, E::G2),
}

#[derive(Serialize, Deserialize)]
pub struct MultiExpInnerProductCProof<E: Engine> {
    #[serde(bound(
        serialize = "GIPAProofWithSSM<E>: Serialize",
        deserialize = "GIPAProofWithSSM<E>: Deserialize<'de>",
    ))]
    pub gipa_proof: GIPAProofWithSSM<E>,
    pub final_ck: E::G2,
    pub final_ck_proof: E::G2,
}

#[derive(Serialize, Deserialize)]
pub struct GIPAProofWithSSM<E: Engine> {
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fr: Serialize,E::G1: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fr: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub r_commitment_steps: Vec<((E::Fqk, E::G1), (E::Fqk, E::G1))>,
    pub r_base: (E::G1, E::Fr),
}
