//! This module contains an [`EvaluationDomain`] abstraction for performing
//! various kinds of polynomial arithmetic on top of the scalar field.
//!
//! In pairing-based SNARKs like [Groth16], we need to calculate a quotient
//! polynomial over a target polynomial with roots at distinct points associated
//! with each constraint of the constraint system. In order to be efficient, we
//! choose these roots to be the powers of a 2<sup>n</sup> root of unity in the
//! field. This allows us to perform polynomial operations in O(n) by performing
//! an O(n log n) FFT over such a domain.
//!
//! [`EvaluationDomain`]: crate::domain::EvaluationDomain
//! [Groth16]: https://eprint.iacr.org/2016/260

use blstrs::*;

use super::multicore::Worker;
use super::SynthesisError;

use crate::gpu;

use log::{info, warn};

pub struct EvaluationDomain {
    coeffs: Vec<Scalar>,
    exp: u32,
    omega: Scalar,
    omegainv: Scalar,
    geninv: Scalar,
    minv: Scalar,
}

impl AsRef<[Scalar]> for EvaluationDomain {
    fn as_ref(&self) -> &[Scalar] {
        &self.coeffs
    }
}

impl AsMut<[Scalar]> for EvaluationDomain {
    fn as_mut(&mut self) -> &mut [Scalar] {
        &mut self.coeffs
    }
}

impl EvaluationDomain {
    pub fn into_coeffs(self) -> Vec<Scalar> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<Scalar>) -> Result<EvaluationDomain, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= SCALAR_S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = Scalar::root_of_unity();
        for _ in exp..SCALAR_S {
            omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, Scalar::zero());

        Ok(EvaluationDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.invert().unwrap(),
            geninv: Scalar::multiplicative_generator().invert().unwrap(),
            minv: Scalar::from(m as u64).invert().unwrap(),
        })
    }

    pub fn fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel>,
    ) -> gpu::GPUResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omega, self.exp)?;
        Ok(())
    }

    pub fn ifft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel>,
    ) -> gpu::GPUResult<()> {
        best_fft(kern, &mut self.coeffs, worker, &self.omegainv, self.exp)?;

        worker.scope(self.coeffs.len(), |scope, chunk| {
            let minv = self.minv;

            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        *v *= &minv;
                    }
                });
            }
        });

        Ok(())
    }

    pub fn distribute_powers(&mut self, worker: &Worker, g: Scalar) {
        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (i, v) in self.coeffs.chunks_mut(chunk).enumerate() {
                scope.spawn(move |_| {
                    let mut u = g.pow(&[(i * chunk) as u64, 0, 0, 0]);
                    for v in v.iter_mut() {
                        *v *= &u;
                        u *= &g;
                    }
                });
            }
        });
    }

    pub fn coset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel>,
    ) -> gpu::GPUResult<()> {
        self.distribute_powers(worker, Scalar::multiplicative_generator());
        self.fft(worker, kern)?;
        Ok(())
    }

    pub fn icoset_fft(
        &mut self,
        worker: &Worker,
        kern: &mut Option<gpu::LockedFFTKernel>,
    ) -> gpu::GPUResult<()> {
        let geninv = self.geninv;
        self.ifft(worker, kern)?;
        self.distribute_powers(worker, geninv);
        Ok(())
    }

    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &Scalar) -> Scalar {
        let mut tmp = tau.pow(&[self.coeffs.len() as u64, 0, 0, 0]);
        tmp -= &Scalar::one();

        tmp
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self, worker: &Worker) {
        let i = self
            .z(&Scalar::multiplicative_generator())
            .invert()
            .unwrap();

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for v in self.coeffs.chunks_mut(chunk) {
                scope.spawn(move |_| {
                    for v in v {
                        *v *= &i;
                    }
                });
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, worker: &Worker, other: &EvaluationDomain) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        *a *= b;
                    }
                });
            }
        });
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, worker: &Worker, other: &EvaluationDomain) {
        assert_eq!(self.coeffs.len(), other.coeffs.len());

        worker.scope(self.coeffs.len(), |scope, chunk| {
            for (a, b) in self
                .coeffs
                .chunks_mut(chunk)
                .zip(other.coeffs.chunks(chunk))
            {
                scope.spawn(move |_| {
                    for (a, b) in a.iter_mut().zip(b.iter()) {
                        *a -= b;
                    }
                });
            }
        });
    }
}

fn best_fft(
    kern: &mut Option<gpu::LockedFFTKernel>,
    a: &mut [Scalar],
    worker: &Worker,
    omega: &Scalar,
    log_n: u32,
) -> gpu::GPUResult<()> {
    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut gpu::FFTKernel| gpu_fft(k, a, omega, log_n))
            .is_ok()
        {
            return Ok(());
        }
    }

    let log_cpus = worker.log_num_cpus();
    if log_n <= log_cpus {
        serial_fft(a, omega, log_n);
    } else {
        parallel_fft(a, worker, omega, log_n, log_cpus);
    }

    Ok(())
}

pub fn gpu_fft(
    kern: &mut gpu::FFTKernel,
    a: &mut [Scalar],
    omega: &Scalar,
    log_n: u32,
) -> gpu::GPUResult<()> {
    kern.radix_fft(a, omega, log_n)?;
    Ok(())
}

pub fn serial_fft(a: &mut [Scalar], omega: &Scalar, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow(&[u64::from(n / (2 * m)), 0, 0, 0]);

        let mut k = 0;
        while k < n {
            let mut w = Scalar::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t *= &w;
                let mut tmp = a[(k + j) as usize];
                tmp -= &t;
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize] += &t;
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft(a: &mut [Scalar], worker: &Worker, omega: &Scalar, log_n: u32, log_cpus: u32) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![Scalar::zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow(&[num_cpus as u64, 0, 0, 0]);

    worker.scope(0, |scope, _| {
        let a = &*a;

        for (j, tmp) in tmp.iter_mut().enumerate() {
            scope.spawn(move |_scope| {
                // Shuffle into a sub-FFT
                let omega_j = omega.pow(&[j as u64, 0, 0, 0]);
                let omega_step = omega.pow(&[(j as u64) << log_new_n, 0, 0, 0]);

                let mut elt = Scalar::one();
                for (i, tmp) in tmp.iter_mut().enumerate() {
                    for s in 0..num_cpus {
                        let idx = (i + (s << log_new_n)) % (1 << log_n);
                        let mut t = a[idx];
                        t *= &elt;
                        *tmp += &t;
                        elt *= &omega_step;
                    }
                    elt *= &omega_j;
                }

                // Perform sub-FFT
                serial_fft(tmp, &new_omega, log_new_n);
            });
        }
    });

    // TODO: does this hurt or help?
    worker.scope(a.len(), |scope, chunk| {
        let tmp = &tmp;

        for (idx, a) in a.chunks_mut(chunk).enumerate() {
            scope.spawn(move |_scope| {
                let mut idx = idx * chunk;
                let mask = (1 << log_cpus) - 1;
                for a in a {
                    *a = tmp[idx & mask][idx >> log_cpus];
                    idx += 1;
                }
            });
        }
    });
}

// Test multiplying various (low degree) polynomials together and
// comparing with naive evaluations.
#[cfg(feature = "pairing")]
#[test]
fn polynomial_arith() {
    use paired::bls12_381::Bls12;
    use rand_core::RngCore;

    fn test_mul<R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs_a in 0..70 {
            for coeffs_b in 0..70 {
                let mut a: Vec<_> = (0..coeffs_a).map(|_| Scalar::random(rng)).collect();
                let mut b: Vec<_> = (0..coeffs_b).map(|_| Scalar::random(rng)).collect();

                // naive evaluation
                let mut naive = vec![Scalar::zero(); coeffs_a + coeffs_b];
                for (i1, a) in a.iter().enumerate() {
                    for (i2, b) in b.iter().enumerate() {
                        let mut prod = *a;
                        prod *= &b;
                        naive[i1 + i2] += &prod;
                    }
                }

                a.resize(coeffs_a + coeffs_b, Scalar::zero());
                b.resize(coeffs_a + coeffs_b, Scalar::zero());

                let mut a = EvaluationDomain::from_coeffs(a).unwrap();
                let mut b = EvaluationDomain::from_coeffs(b).unwrap();

                a.fft(&worker, &mut None);
                b.fft(&worker, &mut None);
                a.mul_assign(&worker, &b);
                a.ifft(&worker, &mut None);

                for (naive, fft) in naive.iter().zip(a.coeffs.iter()) {
                    assert!(naive == fft);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_mul::<Bls12, _>(rng);
}

#[cfg(feature = "pairing")]
#[test]
fn fft_composition() {
    use paired::bls12_381::Bls12;
    use rand_core::RngCore;

    fn test_comp<R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for coeffs in 0..10 {
            let coeffs = 1 << coeffs;

            let mut v = vec![];
            for _ in 0..coeffs {
                v.push(Scalar::random(rng));
            }

            let mut domain = EvaluationDomain::from_coeffs(v.clone()).unwrap();
            domain.ifft(&worker, &mut None);
            domain.fft(&worker, &mut None);
            assert!(v == domain.coeffs);
            domain.fft(&worker, &mut None);
            domain.ifft(&worker, &mut None);
            assert!(v == domain.coeffs);
            domain.icoset_fft(&worker, &mut None);
            domain.coset_fft(&worker, &mut None);
            assert!(v == domain.coeffs);
            domain.coset_fft(&worker, &mut None);
            domain.icoset_fft(&worker, &mut None);
            assert!(v == domain.coeffs);
        }
    }

    let rng = &mut rand::thread_rng();

    test_comp::<Bls12, _>(rng);
}

#[cfg(feature = "pairing")]
#[test]
fn parallel_fft_consistency() {
    use paired::bls12_381::Bls12;
    use rand_core::RngCore;
    use std::cmp::min;

    fn test_consistency<R: RngCore>(rng: &mut R) {
        let worker = Worker::new();

        for _ in 0..5 {
            for log_d in 0..10 {
                let d = 1 << log_d;

                let v1 = (0..d).map(|_| Scalar::random(rng)).collect::<Vec<_>>();
                let mut v1 = EvaluationDomain::from_coeffs(v1).unwrap();
                let mut v2 = EvaluationDomain::from_coeffs(v1.coeffs.clone()).unwrap();

                for log_cpus in log_d..min(log_d + 1, 3) {
                    parallel_fft(&mut v1.coeffs, &worker, &v1.omega, log_d, log_cpus);
                    serial_fft(&mut v2.coeffs, &v2.omega, log_d);

                    assert!(v1.coeffs == v2.coeffs);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_consistency::<_>(rng);
}

pub fn create_fft_kernel(_log_d: usize, priority: bool) -> Option<gpu::FFTKernel> {
    match gpu::FFTKernel::create(priority) {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use crate::domain::{gpu_fft, parallel_fft, serial_fft, EvaluationDomain, Scalar};
    use crate::gpu;
    use crate::multicore::Worker;
    use ff::Field;

    #[test]
    pub fn gpu_fft_consistency() {
        let _ = env_logger::try_init();
        gpu::dump_device_list();

        use paired::bls12_381::{Bls12, Fr};
        use std::time::Instant;
        let rng = &mut rand::thread_rng();

        let worker = Worker::new();
        let log_cpus = worker.log_num_cpus();
        let mut kern = gpu::FFTKernel::create(false).expect("Cannot initialize kernel!");

        for log_d in 1..25 {
            let d = 1 << log_d;

            let elems = (0..d)
                .map(|_| Scalar::<Bls12>(Fr::random(rng)))
                .collect::<Vec<_>>();
            let mut v1 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();
            let mut v2 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();

            println!("Testing FFT for {} elements...", d);

            let mut now = Instant::now();
            gpu_fft(&mut kern, &mut v1.coeffs, &v1.omega, log_d).expect("GPU FFT failed!");
            let gpu_dur =
                now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_cpus {
                serial_fft(&mut v2.coeffs, &v2.omega, log_d);
            } else {
                parallel_fft(&mut v2.coeffs, &worker, &v2.omega, log_d, log_cpus);
            }
            let cpu_dur =
                now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_cpus, cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert!(v1.coeffs == v2.coeffs);
            println!("============================");
        }
    }
}
