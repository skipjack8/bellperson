use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use memmap::Mmap;
use std::io::{self, Read, Write};
use std::mem;

use blstrs::*;

#[derive(Clone)]
pub struct VerifyingKey {
    // alpha in g1 for verifying and for creating A/C elements of
    // proof. Never the point at infinity.
    pub alpha_g1: G1Affine,

    // beta in g1 and g2 for verifying and for creating B/C elements
    // of proof. Never the point at infinity.
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,

    // gamma in g2 for verifying. Never the point at infinity.
    pub gamma_g2: G2Affine,

    // delta in g1/g2 for verifying and proving, essentially the magic
    // trapdoor that forces the prover to evaluate the C element of the
    // proof with only components from the CRS. Never the point at
    // infinity.
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,

    // Elements of the form (beta * u_i(tau) + alpha v_i(tau) + w_i(tau)) / gamma
    // for all public inputs. Because all public inputs have a dummy constraint,
    // this is the same size as the number of inputs, and never contains points
    // at infinity.
    pub ic: Vec<G1Affine>,
}

impl PartialEq for VerifyingKey {
    fn eq(&self, other: &Self) -> bool {
        self.alpha_g1 == other.alpha_g1
            && self.beta_g1 == other.beta_g1
            && self.beta_g2 == other.beta_g2
            && self.gamma_g2 == other.gamma_g2
            && self.delta_g1 == other.delta_g1
            && self.delta_g2 == other.delta_g2
            && self.ic == other.ic
    }
}

impl VerifyingKey {
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writer.write_all(self.alpha_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.beta_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.gamma_g2.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g1.to_uncompressed().as_ref())?;
        writer.write_all(self.delta_g2.to_uncompressed().as_ref())?;
        writer.write_u32::<BigEndian>(self.ic.len() as u32)?;
        for ic in &self.ic {
            writer.write_all(ic.to_uncompressed().as_ref())?;
        }

        Ok(())
    }

    pub fn read<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut g1_repr = [0u8; G1Affine::uncompressed_size()];
        let mut g2_repr = [0u8; G2Affine::uncompressed_size()];

        reader.read_exact(g1_repr.as_mut())?;
        let alpha_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let beta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let beta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let gamma_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g1_repr.as_mut())?;
        let delta_g1 = g1_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        reader.read_exact(g2_repr.as_mut())?;
        let delta_g2 = g2_repr
            .into_affine()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let ic_len = reader.read_u32::<BigEndian>()? as usize;

        let mut ic = vec![];

        for _ in 0..ic_len {
            reader.read_exact(g1_repr.as_mut())?;
            let g1 = g1_repr
                .into_affine()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                .and_then(|e| {
                    if e.is_zero() {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "point at infinity",
                        ))
                    } else {
                        Ok(e)
                    }
                })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }

    pub fn read_mmap(mmap: &Mmap, offset: &mut usize) -> io::Result<Self> {
        let u32_len = mem::size_of::<u32>();
        let g1_len = G1Affine::uncompressed_size();
        let g2_len = G2Affine::uncompressed_size();

        let read_g1 = |mmap: &Mmap, offset: &mut usize| -> Result<G1Affine, std::io::Error> {
            let ptr = &mmap[*offset..*offset + g1_len];
            // Safety: this operation is safe, because it's simply
            // casting to a known struct at the correct offset, given
            // the structure of the on-disk data.
            let g1_repr: [u8; G1Affine::uncompressed_size()] =
                unsafe { *(ptr as *const [u8] as *const _) };

            *offset += g1_len;
            G1Affine::fom_uncompressed(g1_repr)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        };

        let read_g2 = |mmap: &Mmap, offset: &mut usize| -> Result<G2Affine, std::io::Error> {
            let ptr = &mmap[*offset..*offset + g2_len];
            // Safety: this operation is safe, because it's simply
            // casting to a known struct at the correct offset, given
            // the structure of the on-disk data.
            let g2_repr: [u8; G2Affine::uncompressed_size()] =
                unsafe { *(ptr as *const [u8] as *const _) };

            *offset += g2_len;
            G2Affine::from_uncompressed(g2_repr)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        };

        let alpha_g1 = read_g1(&mmap, &mut *offset)?;
        let beta_g1 = read_g1(&mmap, &mut *offset)?;
        let beta_g2 = read_g2(&mmap, &mut *offset)?;
        let gamma_g2 = read_g2(&mmap, &mut *offset)?;
        let delta_g1 = read_g1(&mmap, &mut *offset)?;
        let delta_g2 = read_g2(&mmap, &mut *offset)?;

        let mut raw_ic_len = &mmap[*offset..*offset + u32_len];
        let ic_len = raw_ic_len.read_u32::<BigEndian>()? as usize;
        *offset += u32_len;

        let mut ic = vec![];

        for _ in 0..ic_len {
            let g1_repr = read_g1(&mmap, &mut *offset);
            let g1 = g1_repr.and_then(|e| {
                if e.is_zero() {
                    Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "point at infinity",
                    ))
                } else {
                    Ok(e)
                }
            })?;

            ic.push(g1);
        }

        Ok(VerifyingKey {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        })
    }
}

pub struct PreparedVerifyingKey {
    /// Pairing result of alpha*beta
    pub(crate) alpha_g1_beta_g2: Fp12,
    /// -gamma in G2
    pub(crate) neg_gamma_g2: G2Prepared,
    /// -delta in G2
    pub(crate) neg_delta_g2: G2Prepared,
    /// Copy of IC from `VerifiyingKey`.
    pub(crate) ic: Vec<G1Affine>,
}

pub struct BatchPreparedVerifyingKey {
    /// Pairing result of alpha*beta
    pub(crate) alpha_g1_beta_g2: Fp12,
    /// gamma in G2
    pub(crate) gamma_g2: G2Prepared,
    /// delta in G2
    pub(crate) delta_g2: G2Prepared,
    /// Copy of IC from `VerifiyingKey`.
    pub(crate) ic: Vec<G1Affine>,
}
