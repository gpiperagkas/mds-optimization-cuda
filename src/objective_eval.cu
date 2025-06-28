
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

#include <math.h>

__global__ void evaluate_simplex_sphere(float *a, float *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        b[index] = 0.0;
        for (int i=0;i<n-1;i++)
            b[index] = b[index] + a[index*(n-1)+i]*a[index*(n-1)+i];
    }
}

__global__ void evaluate_simplex_rosenbrock(float *a, float *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        b[index] = 0.0;
        for (int i=0;i<n-2;i++)
            b[index] = b[index] + 100*(a[index*(n-1)+i+1] - a[index*(n-1)+i]*a[index*(n-1)+i])*(a[index*(n-1)+i+1] - a[index*(n-1)+i]*a[index*(n-1)+i]) +
                    (a[index*(n-1)+i]-1)*(a[index*(n-1)+i]-1);
    }
}

__global__ void evaluate_simplex_rastrigin(float *a, float *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double pi=3.14159265358979;
     if (index < n) {
            b[index] = 0.0;
            for (int i=0;i<n-1;i++)
                b[index] = b[index] + a[index*(n-1)+i]*a[index*(n-1)+i] + 10 -10*cosf(2*pi*a[index*(n-1)+i]);
        }
}


extern "C" void objective(float *simplex, float *fsimplex, int n, int benchmark) {
    int threads_per_block = 1024;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    int solution[n-1];
    if (benchmark==0){
        evaluate_simplex_sphere<<<blocks_per_grid, threads_per_block>>>(simplex, fsimplex, n);
        for (int j=0;j<n-1;j++)
            solution[j]=0;
    }else if (benchmark==1){
        evaluate_simplex_rosenbrock<<<blocks_per_grid, threads_per_block>>>(simplex, fsimplex, n);
    }else if (benchmark==2){
        evaluate_simplex_rastrigin<<<blocks_per_grid, threads_per_block>>>(simplex, fsimplex, n);
    }

    cudaDeviceSynchronize();
}

