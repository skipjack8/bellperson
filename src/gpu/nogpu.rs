use super::error::{GPUError, GPUResult};
use crate::multicore::Worker;

use std::sync::Arc;

use blstrs::*;

// This module is compiled instead of `fft.rs` and `multiexp.rs` if `gpu` feature is disabled.
pub struct FFTKernel;

impl FFTKernel {
    pub fn create(_: bool) -> GPUResult<FFTKernel> {
        return Err(GPUError::GPUDisabled);
    }

    pub fn radix_fft(&mut self, _: &mut [Scalar], _: &Scalar, _: u32) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }
}

pub struct MultiexpKernel;

impl MultiexpKernel {
    pub fn create(_: bool) -> GPUResult<MultiexpKernel> {
        return Err(GPUError::GPUDisabled);
    }

    // TODO: allow for both g1 and g2
    pub fn multiexp<C: crate::multiexp::CurveAffine>(
        &mut self,
        _: &Worker,
        _: Arc<Vec<Scalar>>,
        _: Arc<Vec<Scalar>>,
        _: usize,
        _: usize,
    ) -> GPUResult<C::Projective> {
        return Err(GPUError::GPUDisabled);
    }
}

macro_rules! locked_kernel {
    ($class:ident) => {
        pub struct $class;

        impl $class {
            pub fn new(_: usize, _: bool) -> $class {
                $class
            }

            pub fn with<F, R, K>(&mut self, _: F) -> GPUResult<R>
            where
                F: FnMut(&mut K) -> GPUResult<R>,
            {
                return Err(GPUError::GPUDisabled);
            }
        }
    };
}

locked_kernel!(LockedFFTKernel);
locked_kernel!(LockedMultiexpKernel);
