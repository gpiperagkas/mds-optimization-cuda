
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

pub mod mds_obj{
    use cuda::runtime::{cuda_alloc_device, cuda_free_device, cuda_memcpy, CudaMemcpyKind};
    use crate::Problem;

    unsafe extern "C" {
        fn objective(d_simplex: *const f32, d_fsimplex: *const f32, n: *const u32, ofsel: *const u32);
    }
    pub fn objeval( probparams: &Problem, simplex: &Vec<Vec<f32>>) -> Vec<f32> {
        // let solution: Vec<f32> = Vec::new();
        // let obf: f32 = 0.0;
        let ofsel: *const u32 = probparams.bench as *const u32;

        let n: usize = probparams.dim + 1;
        let dim: usize = probparams.dim;
        let nraw: *const u32 = (probparams.dim + 1) as *const u32;
        let simplex_elements: usize = n * dim;  // n points, each with dim dimensions
        let simplex_size = simplex_elements * std::mem::size_of::<f32>();
        let fsimplex_size = n * std::mem::size_of::<f32>();

        // Allocate host memory
        let h_simplex: Vec<f32> = simplex.iter().flat_map(|v| v.iter()).copied().collect();
        let mut h_fsimplex: Vec<f32> = vec![0.0; n];
        
        // Allocate device memory for simplex and evaluated simplex.
        let d_simplex = cuda_alloc_device(simplex_size).unwrap() as *mut f32;
        let d_fsimplex = cuda_alloc_device(fsimplex_size).unwrap() as *mut f32;

        // Copy simplex from host to device
        unsafe {
            match cuda_memcpy(d_simplex, h_simplex.as_ptr(), simplex_elements, CudaMemcpyKind::HostToDevice) {
                Ok(_) => println!("Memory copy to device successful"),
                Err(e) => {
                    println!("CUDA memcpy error: {:?}", e);
                    panic!("Failed to copy memory to device");
                }
            }
        }

        // Launch the kernel
        let threads_per_block = 1024;
        let _blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
        unsafe {
            objective(d_simplex, d_fsimplex, nraw, ofsel);
        }
        
        // Copy result from device to host
        unsafe {
            cuda_memcpy(h_fsimplex.as_mut_ptr(), d_fsimplex, n, CudaMemcpyKind::DeviceToHost).unwrap();
        }

        // Free device memory
        unsafe {
            cuda_free_device(d_simplex as *mut u8).unwrap();
            cuda_free_device(d_fsimplex as *mut u8).unwrap();
        }
        //return result vector
        h_fsimplex

    }
}