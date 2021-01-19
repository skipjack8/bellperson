use serde::{Deserialize, Serialize};

use crate::bls::Engine;
use crate::groth16::aggregagte::commit;

#[derive(Serialize, Deserialize)]
pub struct AggregateProof<E: Engine> {
    pub com_a: E::Fqk,
    pub com_b: E::Fqk,
    pub com_c: E::Fqk,
    pub ip_ab: E::Fqk,
    pub agg_c: E::G1,
    #[serde(bound(
        serialize = "PairingInnerProductABProof<E>: Serialize",
        deserialize = "PairingInnerProductABProof<E>: Deserialize<'de>",
    ))]
    pub tipa_proof_ab: PairingInnerProductABProof<E>,
    #[serde(bound(
        serialize = "MultiExpInnerProductCProof<E>: Serialize",
        deserialize = "MultiExpInnerProductCProof<E>: Deserialize<'de>",
    ))]
    pub tipa_proof_c: MultiExpInnerProductCProof<E>,
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

pub struct TIPPProof<E: Engine> {
    pub gipa: GipaTIPP<E>,
    pub vkey_opening: KZGOpening<E::G2>,
    pub wkey_opening: KZGOpening<E::G1>,
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
