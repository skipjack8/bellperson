use blstrs::{
    Bls12 as Bls12rs, Compress, Engine as Ers, G1Affine, G1Projective, G2Affine, G2Projective,
    PairingCurveAffine as PCAr,
};
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use paired::{
    bls12_381::{Bls12 as Bls12p, Fq12, G1Affine as G1AffineP, G2Affine as G2AffineP},
    Compress as CompressP, Engine, PairingCurveAffine as PCAp,
};
use rand_core::{RngCore, SeedableRng};
use std::io::{Cursor, Read, Write};

#[derive(PartialEq, Clone, Debug)]
struct TestVector {
    g1: Vec<u8>,
    g2: Vec<u8>,
    gt: Vec<u8>,
}

fn compute_blstrs_vector<R: RngCore>(r: &mut R) -> TestVector {
    let g1: G1Affine = G1Projective::random(r).into();
    let g2: G2Affine = G2Projective::random(r).into();
    let gt =
        Bls12rs::final_exponentiation(&Bls12rs::miller_loop(&[(&g1.prepare(), &g2.prepare())]))
            .unwrap();
    let mut eg1: Vec<u8> = Vec::new();
    eg1.write_all(g1.into_compressed().as_ref()).unwrap();
    let mut eg2: Vec<u8> = Vec::new();
    eg2.write_all(g2.into_compressed().as_ref()).unwrap();
    let mut egt: Vec<u8> = Vec::new();
    gt.write_compressed(&mut egt).unwrap();
    TestVector {
        g1: eg1,
        g2: eg2,
        gt: egt,
    }
}

fn verify_testvector_with_paired(t: TestVector) {
    let mut g1c = <G1AffineP as CurveAffine>::Compressed::empty();
    Cursor::new(t.g1).read_exact(g1c.as_mut()).unwrap();
    let g1 = g1c.into_affine().unwrap();

    let mut g2c = <G2AffineP as CurveAffine>::Compressed::empty();
    Cursor::new(t.g2).read_exact(g2c.as_mut()).unwrap();
    let g2 = g2c.into_affine().unwrap();

    let gtexp =
        Bls12p::final_exponentiation(&Bls12p::miller_loop(&[(&g1.prepare(), &g2.prepare())]))
            .unwrap();
    let gt = <Fq12 as CompressP>::read_compressed(&mut Cursor::new(t.gt)).unwrap();
    assert_eq!(gtexp, gt);
}

#[test]
fn test_compat() {
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let v1 = compute_blstrs_vector(&mut rng);
    verify_testvector_with_paired(v1);
}
