/**
 * @file CudaErrorCheck.h
 * @brief CUDA / cuBLAS / cuSPARSE error-checking macros with file+line info.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <stdexcept>
#include <string>

namespace VarProGPU {

inline void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                             std::to_string(line) + " — " +
                             cudaGetErrorString(err));
  }
}

inline void cublasCheck(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error at ") + file + ":" +
                             std::to_string(line) + " (status=" +
                             std::to_string(static_cast<int>(status)) + ")");
  }
}

inline void cusparseCheck(cusparseStatus_t status, const char* file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("cuSPARSE error at ") + file + ":" +
        std::to_string(line) + " — " +
        cusparseGetErrorString(status));
  }
}

}  // namespace VarProGPU

#define CUDA_CHECK(expr) ::VarProGPU::cudaCheck((expr), __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::VarProGPU::cublasCheck((expr), __FILE__, __LINE__)
#define CUSPARSE_CHECK(expr) \
  ::VarProGPU::cusparseCheck((expr), __FILE__, __LINE__)

#else  // !VARPRO_HAVE_CUDA

// Stubs so headers compile without CUDA
#define CUDA_CHECK(expr) (void)(expr)
#define CUBLAS_CHECK(expr) (void)(expr)
#define CUSPARSE_CHECK(expr) (void)(expr)

#endif  // VARPRO_HAVE_CUDA
