use super::error::{GPUError, GPUResult};
use super::locks;
use super::sources;
use super::utils;
use crate::bls::Engine;
use crate::multicore::Worker;
use crate::multiexp::{multiexp as cpu_multiexp, FullDensity};
use ff::{PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::{error, info, warn};
use rayon::prelude::*;
use rust_gpu_tools::*;
use std::any::TypeId;
use std::sync::Arc;

use lazy_static::lazy_static;
use rustacuda::memory::AsyncCopyDestination;
use rustacuda::prelude::*;
use std::ffi::CStr;

const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
const MEMORY_PADDING: f64 = 0.2f64; // Let 20% of GPU memory be free

fn gpu_init() -> Vec<Device> {
    rustacuda::init(CudaFlags::empty())
        .and_then(|_| {
            let devices: Vec<Device> = Device::devices()?
                .filter_map(|device| device.ok())
                .collect();
            Ok(devices)
        })
        .unwrap_or_else(|err| {
            warn!("failed to init cuda: {:?}", err);
            Vec::new()
        })
}

lazy_static! {
    static ref CUDA_GPUS: Vec<Device> = gpu_init();
}

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                Ok(0f64)
            }
        })
        .unwrap_or(0f64)
        .max(0f64)
        .min(1f64)
}

// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
where
    E: Engine,
{
    program: opencl::Program,

    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when num_groups * num_windows ~= 2 * CUDA_CORES
    2 * core_count / num_windows
}

fn calc_window_size(n: usize, exp_bits: usize, core_count: usize) -> usize {
    // window_size = ln(n / num_groups)
    // num_windows = exp_bits / window_size
    // num_groups = 2 * core_count / num_windows = 2 * core_count * window_size / exp_bits
    // window_size = ln(n / num_groups) = ln(n * exp_bits / (2 * core_count * window_size))
    // window_size = ln(exp_bits * n / (2 * core_count)) - ln(window_size)
    //
    // Thus we need to solve the following equation:
    // window_size + ln(window_size) = ln(exp_bits * n / (2 * core_count))
    let lower_bound = (((exp_bits * n) as f64) / ((2 * core_count) as f64)).ln();
    for w in 0..MAX_WINDOW_SIZE {
        if (w as f64) + (w as f64).ln() > lower_bound {
            return w;
        }
    }

    MAX_WINDOW_SIZE
}

fn calc_best_chunk_size(max_window_size: usize, core_count: usize, exp_bits: usize) -> usize {
    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * 2 * core_count / exp_bits
    (((max_window_size as f64).exp() as f64)
        * (max_window_size as f64)
        * 2f64
        * (core_count as f64)
        / (exp_bits as f64))
        .ceil() as usize
}

fn calc_chunk_size<E>(mem: u64, core_count: usize) -> usize
where
    E: Engine,
{
    let aff_size = std::mem::size_of::<E::G1Affine>() + std::mem::size_of::<E::G2Affine>();
    let exp_size = exp_size::<E>();
    let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();
    ((((mem as f64) * (1f64 - MEMORY_PADDING)) as usize)
        - (2 * core_count * ((1 << MAX_WINDOW_SIZE) + 1) * proj_size))
        / (aff_size + exp_size)
}

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(d: opencl::Device, priority: bool) -> GPUResult<SingleMultiexpKernel<E>> {
        let src = sources::kernel::<E>(d.brand() == opencl::Brand::Nvidia);

        let exp_bits = exp_size::<E>() * 8;
        let core_count = utils::get_core_count(&d);
        let mem = d.memory();
        let max_n = calc_chunk_size::<E>(mem, core_count);
        let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        let n = std::cmp::min(max_n, best_n);

        Ok(SingleMultiexpKernel {
            program: opencl::Program::from_opencl(d, &src)?,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let exp_bits = exp_size::<E>() * 8;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let mut base_buffer = self.program.create_buffer::<G>(n)?;
        base_buffer.write_from(0, bases)?;
        let mut exp_buffer = self
            .program
            .create_buffer::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>(n)?;
        exp_buffer.write_from(0, exps)?;

        let bucket_buffer = self
            .program
            .create_buffer::<<G as CurveAffine>::Projective>(2 * self.core_count * bucket_len)?;
        let result_buffer = self
            .program
            .create_buffer::<<G as CurveAffine>::Projective>(2 * self.core_count)?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut global_work_size = num_windows * num_groups;
        global_work_size +=
            (LOCAL_WORK_SIZE - (global_work_size % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let kernel = self.program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            },
            global_work_size,
            None,
        );

        call_kernel!(
            kernel,
            &base_buffer,
            &bucket_buffer,
            &result_buffer,
            &exp_buffer,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        )?;

        let mut results = vec![<G as CurveAffine>::Projective::zero(); num_groups * num_windows];
        result_buffer.read_into(0, &mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

pub struct CudaCtx {
    context: rustacuda::context::Context,
    device: rustacuda::device::Device,
}

pub struct CudaCtxs {
    contexts: Vec<CudaCtx>,
}

#[derive(Clone)]
pub struct CudaUnownedCtx {
    context: rustacuda::context::UnownedContext,
    device: rustacuda::device::Device,
}

#[derive(Clone)]
pub struct CudaUnownedCtxs {
    contexts: Vec<CudaUnownedCtx>,
}

impl CudaCtxs {
    pub fn create() -> rustacuda::error::CudaResult<CudaCtxs> {
        if CUDA_GPUS.is_empty() {
            return Err(rustacuda::error::CudaError::NotInitialized);
        }

        let mut contexts = vec![];
        for device in &*CUDA_GPUS {
            let ctx = Context::create_and_push(
                ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
                *device,
            )?;
            rustacuda::context::ContextStack::pop()?;
            contexts.push(CudaCtx {
                context: ctx,
                device: *device,
            });
        }

        Ok(CudaCtxs { contexts })
    }
}

impl CudaUnownedCtxs {
    pub fn create(ctxs: &CudaCtxs) -> rustacuda::error::CudaResult<CudaUnownedCtxs> {
        let contexts = ctxs
            .contexts
            .iter()
            .map(|ctx| CudaUnownedCtx {
                context: ctx.context.get_unowned(),
                device: ctx.device.clone(),
            })
            .collect();

        Ok(CudaUnownedCtxs { contexts })
    }
}

pub struct SingleMultiexpKernelCuda<E>
where
    E: Engine,
{
    // This should be allocated per device and can be shared across multiple threads
    name: String,
    context: CudaUnownedCtx,
    module: rustacuda::module::Module,

    core_count: usize,
    n: usize,

    priority: bool,
    _phantom: std::marker::PhantomData<E::Fr>,
}

unsafe impl<E> Send for SingleMultiexpKernelCuda<E> where E: Engine {}

impl<E> SingleMultiexpKernelCuda<E>
where
    E: Engine,
{
    pub fn create(ctx: CudaUnownedCtx, priority: bool) -> GPUResult<SingleMultiexpKernelCuda<E>> {
        // TODO: how to handle the path??
        // (cd src/gpu/multiexp; nvcc -O6 -cubin -gencode arch=compute_70,code=sm_75 multiexp.cu -o multiexp.cubin)
        // (cd src/gpu/multiexp; nvcc -O6 -cubin -gencode arch=compute_70,code=sm_75 multiexp32.cu -o multiexp32.cubin)
        let src = CStr::from_bytes_with_nul(b"./src/gpu/multiexp/multiexp32.ptx\0").unwrap();

        let exp_bits = exp_size::<E>() * 8;
        let name = ctx.device.name()?;
        let core_count = utils::get_core_count_by_name(&name);

        info!("Running on {} with {} cores", &name, core_count);

        let mem = ctx.device.total_memory()? as u64;
        let max_n = calc_chunk_size::<E>(mem, core_count);
        let best_n = calc_best_chunk_size(MAX_WINDOW_SIZE, core_count, exp_bits);
        let n = std::cmp::min(max_n, best_n);

        rustacuda::context::CurrentContext::set_current(&ctx.context)?;

        Ok(SingleMultiexpKernelCuda {
            name,
            context: ctx,
            module: rustacuda::module::Module::load_from_file(src)?,
            core_count,
            n,
            priority,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let exp_bits = exp_size::<E>() * 8;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        rustacuda::context::CurrentContext::set_current(&self.context.context)?;

        let stream = rustacuda::stream::Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let module = &self.module;

        assert_eq!(n, bases.len(), "n and bases missmatch");

        let mut base_buffer_g = copy_to_device_buffer(&bases, &stream)?;
        let mut exp_buffer_g = copy_to_device_buffer(&exps, &stream)?;

        // Buckets are device side only
        let bucket_bytes = 2
            * self.core_count
            * bucket_len
            * std::mem::size_of::<<G as CurveAffine>::Projective>();
        let mut bucket_buffer_g =
            unsafe { rustacuda::memory::DeviceBuffer::<u8>::uninitialized(bucket_bytes) }?;
        let result_bytes =
            2 * self.core_count * std::mem::size_of::<<G as CurveAffine>::Projective>();
        let mut result_buffer_g =
            unsafe { rustacuda::memory::DeviceBuffer::<u8>::uninitialized(result_bytes) }?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let global_work_size = (num_windows * num_groups + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

        unsafe {
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                launch!(
                module.G1_bellman_multiexp<<<global_work_size as u32,
                                             LOCAL_WORK_SIZE as u32,
                                             0, stream>>>
                    (base_buffer_g.as_device_ptr(),
                     bucket_buffer_g.as_device_ptr(),
                     result_buffer_g.as_device_ptr(),
                     exp_buffer_g.as_device_ptr(),
                     n as u32,
                     num_groups as u32,
                     num_windows as u32,
                     window_size as u32
                    ))?;
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                launch!(
                module.G2_bellman_multiexp<<<global_work_size as u32,
                                             LOCAL_WORK_SIZE as u32,
                                             0, stream>>>
                    (base_buffer_g.as_device_ptr(),
                     bucket_buffer_g.as_device_ptr(),
                     result_buffer_g.as_device_ptr(),
                     exp_buffer_g.as_device_ptr(),
                     n as u32,
                     num_groups as u32,
                     num_windows as u32,
                     window_size as u32
                    ))?;
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            }
        };

        let mut results = vec![<G as CurveAffine>::Projective::zero(); 2 * self.core_count];
        unsafe {
            let bresults: &mut [u8] = std::slice::from_raw_parts_mut(
                results.as_mut_ptr() as *mut _ as *mut u8,
                result_bytes,
            );
            result_buffer_g.async_copy_to(bresults, &stream)?;
        }
        stream.synchronize()?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

fn copy_to_device_buffer<T>(
    source: &[T],
    stream: &rustacuda::stream::Stream,
) -> rustacuda::error::CudaResult<rustacuda::memory::DeviceBuffer<u8>> {
    let bytes_len = std::mem::size_of::<T>() * source.len();
    unsafe {
        let bytes: &[u8] =
            std::slice::from_raw_parts(source.as_ptr() as *const T as *const u8, bytes_len);

        let mut buffer = rustacuda::memory::DeviceBuffer::<u8>::uninitialized(bytes_len)?;
        buffer.async_copy_from(bytes, stream)?;
        Ok(buffer)
    }
}

// A struct that containts several multiexp kernels for different devices
pub struct MultiexpKernel<E>
where
    E: Engine,
{
    kernels: Vec<SingleMultiexpKernelCuda<E>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
}

impl<E> MultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(ctxs: CudaUnownedCtxs, priority: bool) -> GPUResult<MultiexpKernel<E>> {
        let lock = locks::GPULock::lock();

        let kernels: Vec<_> = ctxs
            .contexts
            .into_iter()
            .map(|ctx| {
                (
                    ctx.device.clone(),
                    SingleMultiexpKernelCuda::<E>::create(ctx, priority),
                )
            })
            .filter_map(|(device, res)| {
                if let Err(ref e) = res {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name().unwrap(),
                        e
                    );
                }
                res.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }
        info!(
            "Multiexp: {} working device(s) selected. (CPU utilization: {})",
            kernels.len(),
            get_cpu_utilization()
        );
        for (i, k) in kernels.iter().enumerate() {
            info!("Multiexp: Device {}: {} (Chunk-size: {})", i, k.name, k.n);
        }
        Ok(MultiexpKernel::<E> {
            kernels,
            _lock: lock,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
        <G as groupy::CurveAffine>::Engine: crate::bls::Engine,
    {
        let num_devices = self.kernels.len();
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + n)];
        let exps = &exps[..n];

        let cpu_n = ((n as f64) * get_cpu_utilization()) as usize;
        let n = n - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);

        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        crate::multicore::THREAD_POOL.install(|| {
            let mut acc = <G as CurveAffine>::Projective::zero();
            let results = if n > 0 {
                bases
                    .par_chunks(chunk_size)
                    .zip(exps.par_chunks(chunk_size))
                    .zip(self.kernels.par_iter_mut())
                    .map(|((bases, exps), kern)| -> Result<<G as CurveAffine>::Projective, GPUError> {
                        let mut acc = <G as CurveAffine>::Projective::zero();
                        for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
                            let result = kern.multiexp(bases, exps, bases.len())?;
                            acc.add_assign(&result);
                        }

                        Ok(acc)
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };

            let cpu_acc = cpu_multiexp(
                &pool,
                (Arc::new(cpu_bases.to_vec()), 0),
                FullDensity,
                Arc::new(cpu_exps.to_vec()),
                &mut None,
            );

            for r in results {
                acc.add_assign(&r?);
            }

            acc.add_assign(&cpu_acc.wait().unwrap());
            Ok(acc)
        })
    }
}
