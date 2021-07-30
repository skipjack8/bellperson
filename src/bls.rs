pub use blstrs::{
    Bls12, Compress, Fp as Fq, Fp12 as Fq12, Fp2 as Fq2, G1Affine, G1Compressed, G1Projective,
    G1Uncompressed, G2Affine, G2Compressed, G2Prepared, G2Projective, G2Uncompressed, Scalar as Fr,
};

use crate::EngineExt;

impl EngineExt for Bls12 {
    type Fq = Fq;
    type Fqe = Fq2;
}
