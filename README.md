# Sparsity Benchmarks

Benchmarking code for "PopSparse: Accelerated block sparse matrix multiplication on IPU" [paper](https://arxiv.org/abs/2303.16999).

_We recommend starting with the PyTorch demo notebook:_

[![Run on Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/graphcore-research/notebooks?container=graphcore%2Fpytorch-jupyter%3A3.2.0-ubuntu-20.04&machine=Free-IPU-POD4&file=%2Fsparsity_benchmarks%2FSpMM.ipynb)


## Contents

 - [GPU benchmarking](#gpu-benchmarking)
 - [IPU benchmarking](ipu/)
 - [Analysis](analysis/)

### GPU benchmarking

To produce GPU timing numbers, make the following modifications to third-party benchmarks:

 - Using CUDA 11.6.2 on a DGX A100.
 - (BSR) Clone [ceruleangu/Block-Sparse-Benchmark](https://github.com/ceruleangu/Block-Sparse-Benchmark).
   - Replace `num_r_block -> (num_r_block - 1)`, `num_c_block -> (num_c_block - 1)` in `generate_candidate_blocks()`.
 - (Dense) Clone [hbrunie/cublas_benchmarks](https://github.com/hbrunie/cublas_benchmarks).
 - (Optional) We recommend wrapping all CUDA calls in `CHECK_CUDA` macros to identify errors.
 - Add `cudaDeviceSynchronize()` at the beginning of `GpuTimer::Start()`.
 - Start `GpuTimer` after the first 5 runs.
 - Stop after 20 timed runs, recording the total time using `cudaEventElapsedTime()`.


## References & license

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under a MIT license (see [LICENSE](LICENSE)).

Our dependencies are:

| Component | About | License |
| --- | --- | --- |
| cxxopts | CLI option parsing library (https://github.com/jarro2783/cxxopts) | MIT |
| matplotlib | Plotting library | BSD |
| numpy | Scientific computing with Python | BSD 3-Clause |
