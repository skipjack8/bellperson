use std::env;

use ff_cl_gen as ffgen;
use log::info;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
use rust_gpu_tools::{opencl, Device, Framework, GPUError as GpuToolsError, Program, Vendor};

use crate::bls::Engine;
#[cfg(not(feature = "cuda"))]
use crate::gpu::error::GPUError;
use crate::gpu::error::GPUResult;

// Instead of having a very large OpenCL program written for a specific curve, with a lot of
// rudandant codes (As OpenCL doesn't have generic types or templates), this module will dynamically
// generate OpenCL codes given different PrimeFields and curves.

static FFT_SRC: &str = include_str!("fft/fft.cl");
static FIELD2_SRC: &str = include_str!("multiexp/field2.cl");
static EC_SRC: &str = include_str!("multiexp/ec.cl");
static MULTIEXP_SRC: &str = include_str!("multiexp/multiexp.cl");

fn field2(field2: &str, field: &str) -> String {
    String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}

#[cfg(not(feature = "blstrs"))]
const BLSTRS_DEF: &str = "";
#[cfg(feature = "blstrs")]
const BLSTRS_DEF: &str = "#define BLSTRS";

fn ec(field: &str, point: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
        .replace("__BLSTRS__", BLSTRS_DEF)
}

fn multiexp(point: &str, exp: &str) -> String {
    String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
}

// WARNING: This function works only with Short Weierstrass Jacobian curves with Fq2 extension field.
pub fn kernel<E>(limb64: bool) -> String
where
    E: Engine,
{
    vec![
        if limb64 {
            ffgen::field::<E::Fr, ffgen::Limb64>("Fr")
        } else {
            ffgen::field::<E::Fr, ffgen::Limb32>("Fr")
        },
        fft("Fr"),
        if limb64 {
            ffgen::field::<E::Fq, ffgen::Limb64>("Fq")
        } else {
            ffgen::field::<E::Fq, ffgen::Limb32>("Fq")
        },
        ec("Fq", "G1"),
        multiexp("G1", "Fr"),
        field2("Fq2", "Fq"),
        ec("Fq2", "G2"),
        multiexp("G2", "Fr"),
    ]
    .join("\n\n")
}

/// Returns the program for the preferred [`rust_gpu_tools::device::Framework`].
///
/// If the device supports CUDA, then CUDA is used, else OpenCL. You can force a selection with
/// the environment variable `BELLMAN_GPU_FRAMEWORK`, which can be set either to `cuda` or `opencl`.
pub fn program<E>(device: &Device) -> GPUResult<Program>
where
    E: Engine,
{
    let framework = match env::var("BELLMAN_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Framework::Cuda
                }

                #[cfg(not(feature = "cuda"))]
                return Err(GPUError::Simple("CUDA framework is not supported, please compile withthe `cuda` feature enabled."));
            }
            "opencl" => Framework::Opencl,
            _ => device.framework(),
        },
        Err(_) => device.framework(),
    };
    program_use_framework::<E>(device, &framework)
}

/// Returns the program for the specified [`rust_gpu_tools::device::Framework`].
pub fn program_use_framework<E>(device: &Device, framework: &Framework) -> GPUResult<Program>
where
    E: Engine,
{
    match framework {
        #[cfg(feature = "cuda")]
        Framework::Cuda => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_MULTIEXP_FATBIN"));
            let cuda_device = device.cuda_device().ok_or(GpuToolsError::DeviceNotFound)?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)?;
            Ok(Program::Cuda(program))
        }
        Framework::Opencl => {
            info!("Using kernel on OpenCL.");
            let src = kernel::<E>(device.vendor() == Vendor::Nvidia);
            let opencl_device = device
                .opencl_device()
                .ok_or(GpuToolsError::DeviceNotFound)?;
            let program = opencl::Program::from_opencl(opencl_device, &src)?;
            Ok(Program::Opencl(program))
        }
    }
}
