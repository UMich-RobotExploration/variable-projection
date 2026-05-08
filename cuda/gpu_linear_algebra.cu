/**
 * @file gpu_linear_algebra.cu
 * @brief Minimal CUDA kernel code — only custom kernels that require nvcc.
 *
 * cuBLAS and cuSPARSE API calls (which are plain C API) are implemented in
 * src/gpu/GpuLinearAlgebra.cpp compiled by g++, not nvcc.
 *
 * This file only contains kernels that must be compiled by nvcc.
 */

#ifdef VARPRO_HAVE_CUDA

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Currently no custom kernels needed in gpu_linear_algebra.cu.
// All cuBLAS/cuSPARSE calls are in src/gpu/GpuLinearAlgebra.cpp.
// ---------------------------------------------------------------------------

// Placeholder to ensure this translation unit is non-empty.
namespace VarProGPU {
__global__ void noop_kernel() {}
}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
