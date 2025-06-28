
// Copyright 2025 Grigorios Piperagkas. All rights reserved.
// Use of this source code is governed by a BSD-3-clause
// license that can be found in the LICENSE file.
/*
//////////////////////////////////////////////////////////////////////////////

Main algorithm for Parallel Multidirectional Search proposed by VJ Torczon(1989).
implemented for evaluation of objective function in parallel on cuda enabled GPUs.

June 2025. 
/////////////////////////////////////////////////////////////////////////////

 */

// this build file has mostly been LLM generated for linking with .cu file.

use std::env;
use std::process::Command;


fn main() {
    // Get the output directory
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Compile the CUDA kernel
    let cuda_file = "src/objective_eval.cu";
    let output_file = format!("{}/objective_eval.o", out_dir);
    
    // Run nvcc to compile the CUDA kernel
    let output = Command::new("nvcc")
        .args(&[
            "-c",
            cuda_file,
            "-o",
            &output_file,
            "--compiler-options", "-fPIC"
        ])
        .output()
        .expect("Failed to execute nvcc. Make sure CUDA is installed and nvcc is in PATH.");
    
    if !output.status.success() {
        panic!("nvcc compilation failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Create a static library from the object file
    let lib_file = format!("{}/libobjective_eval.a", out_dir);
    let ar_output = Command::new("ar")
        .args(&["rcs", &lib_file, &output_file])
        .output()
        .expect("Failed to execute ar");
    
    if !ar_output.status.success() {
        panic!("ar failed: {}", String::from_utf8_lossy(&ar_output.stderr));
    }
    
    // Tell cargo to link the library
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=objective_eval");
    
    // Tell cargo to rerun this build script if the CUDA file changes
    println!("cargo:rerun-if-changed=src/objective_eval.cu");
}

