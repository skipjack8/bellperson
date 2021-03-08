use std::io::Write;

use ff::{PrimeField, PrimeFieldRepr};
use groupy::{CurveAffine, CurveProjective};
use serde::{Deserialize, Serialize};

use blstrs::Compress;

use crate::bls::Engine;
use crate::groth16::aggregate::commit;

/// AggregateProof contains all elements to verify n aggregated Groth16 proofs
/// using inner pairing product arguments. This proof can be created by any
/// party in possession of valid Groth16 proofs.
#[derive(Serialize, Deserialize, Debug)]
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
#[derive(Serialize, Deserialize, Debug)]
impl<E: Engine> AggregateProof<E> {
    /// Writes the agggregated proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // com_ab
        self.com_ab.0.write_compressed(&mut out)?;
        self.com_ab.1.write_compressed(&mut out)?;

        // com_c
        self.com_c.0.write_compressed(&mut out)?;
        self.com_c.1.write_compressed(&mut out)?;

        // ip_ab
        self.ip_ab.write_compressed(&mut out)?;

        // agg_c
        let agg_c = self.agg_c.into_affine().into_compressed();
        out.write_all(agg_c.as_ref())?;

        // tmpip
        self.tmipp.write(&mut out)?;

        Ok(())
    }

    /// Returns the number of bytes this proof is serialized to.
    pub fn serialized_len(&self) -> usize {
        // TODO: calculate
        let mut out = Vec::new();
        self.write(&mut out).unwrap();

        out.len()
    }
}

#[derive(Serialize, Deserialize)]
pub struct GipaProof<E: Engine> {
    pub nproofs: u32,
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
#[derive(Serialize, Deserialize, Debug)]
impl<E: Engine> GipaProof<E> {
    fn log_proofs(&self) -> usize {
        (self.nproofs as f32).log2().ceil() as usize
    }

    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // number of proofs
        out.write_all(&self.nproofs.to_le_bytes()[..])?;

        assert_eq!(self.comms_ab.len(), self.log_proofs());
        // comms_ab
        for (x, y) in &self.comms_ab {
            x.0.write_compressed(&mut out)?;
            x.1.write_compressed(&mut out)?;
            y.0.write_compressed(&mut out)?;
            y.1.write_compressed(&mut out)?;
        }

        assert_eq!(self.comms_c.len(), self.log_proofs());
        // comms_c
        for (x, y) in &self.comms_c {
            x.0.write_compressed(&mut out)?;
            x.1.write_compressed(&mut out)?;
            y.0.write_compressed(&mut out)?;
            y.1.write_compressed(&mut out)?;
        }

        assert_eq!(self.z_ab.len(), self.log_proofs());
        // z_ab
        for (x, y) in &self.z_ab {
            x.write_compressed(&mut out)?;
            y.write_compressed(&mut out)?;
        }

        assert_eq!(self.z_c.len(), self.log_proofs());
        // z_c
        for (x, y) in &self.z_c {
            out.write_all(x.into_affine().into_compressed().as_ref())?;
            out.write_all(y.into_affine().into_compressed().as_ref())?;
        }

        // final_a
        out.write_all(self.final_a.into_compressed().as_ref())?;

        // final_b
        out.write_all(self.final_b.into_compressed().as_ref())?;

        // final_c
        out.write_all(self.final_c.into_compressed().as_ref())?;

        // final_r
        self.final_r.into_repr().write_le(&mut out)?;

        // final_vkey
        out.write_all(self.final_vkey.0.into_compressed().as_ref())?;
        out.write_all(self.final_vkey.1.into_compressed().as_ref())?;

        // final_wkey
        out.write_all(self.final_wkey.0.into_compressed().as_ref())?;
        out.write_all(self.final_wkey.1.into_compressed().as_ref())?;

        Ok(())
    }
}

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

impl<E: Engine> TippMippProof<E> {
    /// Writes the  proof into the provided buffer.
    pub fn write(&self, mut out: impl Write) -> std::io::Result<()> {
        // gipa
        self.gipa.write(&mut out)?;

        // vkey_opening
        let x0 = self.vkey_opening.0.into_compressed();
        let x1 = self.vkey_opening.1.into_compressed();

        out.write_all(x0.as_ref())?;
        out.write_all(x1.as_ref())?;

        // wkey_opening
        let x0 = self.wkey_opening.0.into_compressed();
        let x1 = self.wkey_opening.1.into_compressed();

        out.write_all(x0.as_ref())?;
        out.write_all(x1.as_ref())?;

        Ok(())
    }
}

/// KZGOpening represents the KZG opening of a commitment key (which is a tuple
/// given commitment keys are a tuple).
pub type KZGOpening<G> = (G, G);
