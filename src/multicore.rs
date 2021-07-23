//! An interface for dealing with the kinds of parallel computations involved in
//! `bellperson`. It's currently just a thin wrapper around [`CpuPool`] and
//! [`rayon`] but may be extended in the future to allow for various
//! parallelism strategies.
//!
//! [`CpuPool`]: futures_cpupool::CpuPool

use std::env;

use lazy_static::lazy_static;
use yastl::Pool;

const MAX_VERIFIER_THREADS: usize = 6;

lazy_static! {
    static ref NUM_CPUS: usize = env::var("BELLMAN_NUM_CPUS")
        .ok()
        .and_then(|num| num.parse().ok())
        .unwrap_or_else(num_cpus::get);
    pub static ref THREAD_POOL: Pool = Pool::new(*NUM_CPUS);
    pub static ref VERIFIER_POOL: Pool = Pool::new(NUM_CPUS.max(MAX_VERIFIER_THREADS));
    pub static ref RAYON_THREAD_POOL: rayon::ThreadPool =
        rayon::ThreadPoolBuilder::new().build().unwrap();
}

#[derive(Clone, Default)]
pub struct Worker {}

impl Worker {
    pub fn new() -> Worker {
        Worker {}
    }

    pub fn log_num_cpus(&self) -> u32 {
        log2_floor(*NUM_CPUS)
    }

    pub fn scope<'a, F, R>(&self, elements: usize, f: F) -> R
    where
        F: FnOnce(&yastl::Scope<'a>, usize) -> R,
    {
        let chunk_size = if elements < *NUM_CPUS {
            1
        } else {
            elements / *NUM_CPUS
        };

        THREAD_POOL.scoped(|scope| f(scope, chunk_size))
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_log2_floor() {
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(4), 2);
        assert_eq!(log2_floor(5), 2);
        assert_eq!(log2_floor(6), 2);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
    }
}
