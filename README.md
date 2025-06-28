# mds-optimization-cuda
An implementation of multidirectional search in Rust with function evaluations in parallel on cuda enabled GPU.

The Multi-directional search optimization algorithm is based on the algorithm proposed by VJ
Torczon(1989), found in the reference: 
Torczon, V. J. (1989). Multidirectional search: A direct search algorithm for parallel machines. Rice
University. 

The evaluation of functions is processed in parallel, i.e. for N-dimensional problems, N+1 is the dimension of 
simplex which is passed on cuda kernel. This parallelism can help with increased dimensionality, improving 
computational efficiency. 

To-do: Add more objective functions for benchmarking. Propose functionality to aleviate the problem of getting stuck in local minima.

This software is intented for non-commercial purposes, and is licensed under a BSD 3-clause license.
