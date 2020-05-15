// -i, --inputs <inputs>    Sets number of public inputs
// -p, --proofs <proofs>    Sets number of proofs in a batch
// --nogpu                  Disables GPU

use ff::Field;
use rand::{thread_rng, Rng};
use std::sync::Arc;

use bellperson::groth16::{
    prepare_batch_verifying_key, verify_proofs_batch, Parameters, Proof, VerifyingKey,
};
use groupy::CurveProjective;
use paired::bls12_381::Bls12;
use paired::Engine;
use std::time::Instant;
use structopt::StructOpt;

fn random_points<C: CurveProjective, R: Rng>(count: usize, rng: &mut R) -> Vec<C::Affine> {
    // Number of distinct points is limited because generating random points is very time
    // consuming, so it's better to just repeat them.
    const DISTINT_POINTS: usize = 100;
    (0..DISTINT_POINTS)
        .map(|_| C::random(rng).into_affine())
        .collect::<Vec<_>>()
        .into_iter()
        .cycle()
        .take(count)
        .collect()
}

fn dummy_proofs<E: Engine, R: Rng>(count: usize, rng: &mut R) -> Vec<Proof<E>> {
    (0..count)
        .map(|_| Proof {
            a: E::G1::random(rng).into_affine(),
            b: E::G2::random(rng).into_affine(),
            c: E::G1::random(rng).into_affine(),
        })
        .collect()
}

fn dummy_inputs<E: Engine, R: Rng>(count: usize, rng: &mut R) -> Vec<<E as ff::ScalarEngine>::Fr> {
    (0..count)
        .map(|_| <E as ff::ScalarEngine>::Fr::random(rng))
        .collect()
}

fn dummy_vk<E: Engine, R: Rng>(count: usize, rng: &mut R) -> VerifyingKey<E> {
    VerifyingKey {
        alpha_g1: E::G1::random(rng).into_affine(),
        beta_g1: E::G1::random(rng).into_affine(),
        beta_g2: E::G2::random(rng).into_affine(),
        gamma_g2: E::G2::random(rng).into_affine(),
        delta_g1: E::G1::random(rng).into_affine(),
        delta_g2: E::G2::random(rng).into_affine(),
        ic: random_points::<E::G1, _>(count + 1, rng),
    }
}

fn dummy_params<E: Engine, R: Rng>(count: usize, rng: &mut R) -> Parameters<E> {
    let hlen = (1 << ((count as f64).log2().floor() as usize + 2)) - 1;
    Parameters {
        vk: dummy_vk(count, rng),
        h: Arc::new(random_points::<E::G1, _>(hlen, rng)),
        l: Arc::new(Vec::new()),
        a: Arc::new(random_points::<E::G1, _>(count + 1, rng)),
        b_g1: Arc::new(random_points::<E::G1, _>(count, rng)),
        b_g2: Arc::new(random_points::<E::G2, _>(count, rng)),
    }
}

#[derive(Debug, StructOpt, Clone, Copy)]
#[structopt(name = "Bellman Bench", about = "Benchmarking Bellman.")]
struct Opts {
    #[structopt(long = "proofs", default_value = "1")]
    proofs: usize,
    #[structopt(long = "public", default_value = "1")]
    public: usize,
    #[structopt(long = "private", default_value = "1000000")]
    private: usize,
    #[structopt(long = "samples", default_value = "10")]
    samples: usize,
    #[structopt(long = "gpu")]
    gpu: bool,
}

fn main() {
    let rng = &mut thread_rng();
    env_logger::init();

    let opts = Opts::from_args();
    if opts.gpu {
        std::env::set_var("BELLMAN_VERIFIER", "gpu");
    }

    let inputs = dummy_inputs::<Bls12, _>(opts.public, rng);
    let proofs = dummy_proofs::<Bls12, _>(opts.proofs, rng);
    let params = dummy_params::<Bls12, _>(opts.public, rng);
    let pvk = prepare_batch_verifying_key(&params.vk);
    println!(
        "{} proofs, each having {} public inputs...",
        opts.proofs, opts.public
    );

    let pref = proofs.iter().collect::<Vec<&_>>();
    println!("Verifying...");

    for _ in 0..10 {
        let now = Instant::now();
        verify_proofs_batch(&pvk, rng, &pref[..], &vec![inputs.clone(); opts.proofs]).unwrap();
        println!(
            "Verification finished in {}s and {}ms",
            now.elapsed().as_secs(),
            now.elapsed().subsec_nanos() / 1000000
        );
    }
}
