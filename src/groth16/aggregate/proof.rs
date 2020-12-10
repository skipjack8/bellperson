use ff::PrimeField;
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use serde::{Deserialize, Serialize};
use std::mem;

use crate::bls::Engine;

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
