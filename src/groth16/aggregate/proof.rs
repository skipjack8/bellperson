use crate::bls::Engine;
use crate::groth16::aggregate::commit;
use serde::{Deserialize, Serialize};
/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
#[derive(Serialize, Deserialize)]
pub struct AggregateProof<E: Engine> {
    /// commitment to A and B using the pair commitment scheme needed to verify
    /// TIPP relation.
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub com_ab: commit::Output<E>,
    /// commit to C separate since we use it only in MIPP
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub com_c: commit::Output<E>,
    /// $A^r * B = Z$ is the left value on the aggregated Groth16 equation
    pub ip_ab: E::Fqk,
    /// $C^r$ is used on the right side of the aggregated Groth16 equation
    pub agg_c: E::G1,
    #[serde(bound(
        serialize = "TippMippProof<E>: Serialize",
        deserialize = "TippMippProof<E>: Deserialize<'de>",
    ))]
    pub tmipp: TippMippProof<E>,
}

/// It contains all elements derived in the GIPA loop for both TIPP and MIPP at
/// the same time.
#[derive(Serialize, Deserialize)]
pub struct GipaProof<E: Engine> {
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub comms_ab: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub comms_c: Vec<(commit::Output<E>, commit::Output<E>)>,
    #[serde(bound(
        serialize = "E::Fqk: Serialize, E::Fqk: Serialize",
        deserialize = "E::Fqk: Deserialize<'de>, E::Fqk: Deserialize<'de>",
    ))]
    pub z_ab: Vec<(E::Fqk, E::Fqk)>,
    #[serde(bound(
        serialize = "E::G1: Serialize, E::G1: Serialize",
        deserialize = "E::G1: Deserialize<'de>, E::G1: Deserialize<'de>",
    ))]
    pub z_c: Vec<(E::G1, E::G1)>,
    pub final_a: E::G1Affine,
    pub final_b: E::G2Affine,
    pub final_c: E::G1Affine,
    pub final_r: E::Fr,
    /// final commitment keys $v$ and $w$ - there is only one element at the
    /// end for v1 and v2 hence it's a tuple.
    #[serde(bound(
        serialize = "E::G2Affine: Serialize, E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>, E::G2Affine: Deserialize<'de>",
    ))]
    pub final_vkey: (E::G2Affine, E::G2Affine),
    #[serde(bound(
        serialize = "E::G1Affine: Serialize, E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>, E::G1Affine: Deserialize<'de>",
    ))]
    pub final_wkey: (E::G1Affine, E::G1Affine),
}

/// It contains the GIPA recursive elements as well as the KZG openings for v
/// and w
#[derive(Serialize, Deserialize)]
pub struct TippMippProof<E: Engine> {
    #[serde(bound(
        serialize = "GipaProof<E>: Serialize",
        deserialize = "GipaProof<E>: Deserialize<'de>",
    ))]
    pub gipa: GipaProof<E>,
    #[serde(bound(
        serialize = "E::G2Affine: Serialize",
        deserialize = "E::G2Affine: Deserialize<'de>",
    ))]
    pub vkey_opening: KZGOpening<E::G2Affine>,
    #[serde(bound(
        serialize = "E::G1Affine: Serialize",
        deserialize = "E::G1Affine: Deserialize<'de>",
    ))]
    pub wkey_opening: KZGOpening<E::G1Affine>,
}

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
pub type KZGOpening<G> = (G, G);
