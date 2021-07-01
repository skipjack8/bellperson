/// The build script is needed to compile the CUDA kernel.

#[cfg(feature = "gpu")]
fn main() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    static CUDA_MULTIEXP_PATH: &str = "src/gpu/multiexp/multiexp32.cu";

    // The kernel only needs to be re-compiled if it changed.
    println!("cargo:rerun-if-changed={}", CUDA_MULTIEXP_PATH);

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR was not set.");
    let fatbin_path: PathBuf = [&out_dir, "multiexp32.fatbin"].iter().collect();

    // nvcc --optimize=6 --fatbin --gpu-architecture=sm_86 --generate-code=arch=compute_86,code=sm_86 --generate-code=arch=compute_80,code=sm_80 --generate-code=arch=compute_75,code=sm_75 --define-macro=BLSTRS --output-file multiexp32.fatbin src/gpu/multiexp/multiexp32.cu
    let status = Command::new("nvcc")
        .arg("--optimize=6")
        .arg("--fatbin")
        .arg("--gpu-architecture=sm_86")
        .arg("--generate-code=arch=compute_86,code=sm_86")
        .arg("--generate-code=arch=compute_80,code=sm_80")
        .arg("--generate-code=arch=compute_75,code=sm_75")
        .arg("--define-macro=BLSTRS")
        .arg("--output-file")
        .arg(&fatbin_path)
        .arg(CUDA_MULTIEXP_PATH)
        .status()
        .expect("Cannot run nvcc.");

    if status.success() {
        // The idea to put the path to the farbin into a compile-time env variable is from
        // https://github.com/LutzCle/fast-interconnects-demo/blob/b80ea8e04825167f486ab8ac1b5d67cf7dd51d2c/rust-demo/build.rs
        println!(
            "cargo:rustc-env=CUDA_MULTIEXP_FATBIN={}",
            fatbin_path.to_str().unwrap()
        );
    } else {
        panic!("nvcc failed.");
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {}
