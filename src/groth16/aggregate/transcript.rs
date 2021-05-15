use crate::bls::{Compress, Engine, Fq12, Fr, G1Affine, G2Affine};
use ff::{Field, PrimeField, PrimeFieldRepr};
use groupy::CurveAffine;
use sha2::{Digest, Sha256};

pub(crate) trait Writable {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()>;
}

// LEADS TO :
// the trait ` groth16::aggregate::transcript::Writable` is not implemented for
// `&<E as paired::Engin e>::Fqk` while E::Fqk IS a Fq12
// -
// - Fqk implements the Compress trait. Unfortunately it is not possible to use
// because that would lead to a conflict in trait resolution: G1Affine might
// implement Compress as wellas all the other type, so it wont be able to know
// which one. Unfortunately Fqk is not extended by Compress trait
/*impl<E, T> Writable for T*/
//where
//E: Engine<Fqk = T>,
//T: Compress,
//{
//fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
//// TODO: anyway to write a non compressed version without using serde?
//self.write_compressed(out)
//}
/*}*/

impl Writable for Fq12 {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // TODO: anyway to write a non compressed version without using serde?
        self.write_compressed(out)
    }
}
impl Writable for &Fq12 {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        // TODO: anyway to write a non compressed version without using serde?
        self.write_compressed(out)
    }
}
impl Writable for Fr {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        self.into_repr().write_be(out)
    }
}
impl Writable for &Fr {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        self.into_repr().write_be(out)
    }
}
/// NOTE: I wish doing `impl<T: CurveAffine>` were possible but it raise
/// conflicting implementation error since an external type might implement
/// CurveAffine for Fr as well.
impl Writable for G1Affine {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        out.write_all(self.into_compressed().as_ref())
    }
}
impl Writable for &G1Affine {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        out.write_all(self.into_compressed().as_ref())
    }
}
impl Writable for G2Affine {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        out.write_all(self.into_compressed().as_ref())
    }
}
impl Writable for &G2Affine {
    fn write_to<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        out.write_all(self.into_compressed().as_ref())
    }
}

pub(crate) struct Transcript {
    internal: Sha256,
}

impl Transcript {
    pub fn new(application_tag: &str) -> Self {
        let mut internal = sha2::Sha256::new();
        internal.update(application_tag);
        Transcript { internal }
    }

    pub fn append<C: Writable>(&mut self, e: &C) -> std::io::Result<()> {
        let mut buff = Vec::new();
        e.write_to(&mut buff)?;
        //e.write_compressed(&mut buff)?;
        self.internal.update(&buff);
        Ok(())
    }
    pub fn append_vec<'a, I, W: 'a>(&mut self, v: I) -> std::io::Result<()>
    where
        I: IntoIterator<Item = &'a W>,
        W: Writable,
    {
        let mut buff = Vec::new();
        for e in v {
            e.write_to(&mut buff)?;
        }
        self.internal.update(&buff);
        Ok(())
    }
    pub fn domain_sep(&mut self, tag: &str) {
        self.internal.update(tag);
    }
    pub fn derive_challenge<F: PrimeField>(&self) -> F {
        let mut state = self.internal.clone();
        let mut counter_nonce: usize = 0;
        let one = F::one();
        let r = loop {
            counter_nonce += 1;
            state.update(&counter_nonce.to_be_bytes()[..]);
            let curr_state = state.clone();
            let digest = curr_state.finalize();
            if let Some(c) = F::from_random_bytes(&digest) {
                if c == one {
                    continue;
                }
                if let Some(_) = c.inverse() {
                    break c;
                }
            }
        };
        r
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bls::{Bls12, Fr, G1Affine};
    use crate::bls::{Engine, PairingCurveAffine};
    use ff::Field;

    #[test]
    fn test_transcript() {
        let mut t = Transcript::new("test");
        let g1 = G1Affine::one();
        let g2 = G2Affine::one();
        let gt = <Bls12 as Engine>::final_exponentiation(&<Bls12 as Engine>::miller_loop(&[(
            &g1.prepare(),
            &g2.prepare(),
        )]))
        .expect("pairing failed");
        t.domain_sep("testing domain1");
        t.append(&g1).expect("this should have worked");
        t.append(&g2).expect("this should have worked");
        t.append(&gt).expect("woups");
        t.append(&Fr::one()).expect("this should have worked");
        t.append_vec(&[&g1, &g1, &g1]).expect("woups");
        t.append_vec(&[&g2, &g2, &g2]).expect("woups");
        t.append_vec(&[&gt, &gt, &gt]).expect("woups");
        let c1: Fr = t.derive_challenge();
        t.domain_sep("testing domain2");
        let c2: Fr = t.derive_challenge();
        assert!(c1 != c2);
    }
}
