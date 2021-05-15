use crate::bls::{Compress, Engine, Fq12, Fr, G1Affine, G2Affine};
use ff::{Field, PrimeField, PrimeFieldRepr};
use groupy::CurveAffine;
use sha2::{Digest, Sha256};

pub(crate) struct Transcript {
    internal: Sha256,
}

impl Transcript {
    pub fn new(application_tag: &str) -> Self {
        let mut internal = sha2::Sha256::new();
        internal.update(application_tag);
        Transcript { internal }
    }

    pub fn append(&mut self, buff: &[u8]) {
        self.internal.update(&buff);
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
        let input = tov!(&g1, &g2, &gt, &Fr::one());
        t.append(&input);
        let c1: Fr = t.derive_challenge();
        let c11: Fr = t.derive_challenge();
        assert_eq!(c1, c11);
        t.domain_sep("testing domain2");
        let c2: Fr = t.derive_challenge();
        assert!(c1 != c2);

        let mut t2 = Transcript::new("test");
        t2.domain_sep("testing domain1");
        t2.append(&input);
        let c12 = t2.derive_challenge();
        assert_eq!(c1, c12);
    }
}
