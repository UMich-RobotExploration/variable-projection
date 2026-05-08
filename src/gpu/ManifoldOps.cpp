/**
 * @file ManifoldOps.cpp
 * @brief Host-side wrappers for manifold CUDA kernels.
 *
 * Compiled by g++; calls CUDA device kernels via forward-declared C symbols
 * (defined in manifold_kernels.cu).
 */

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/ManifoldKernels.h>
#include <VarProGPU/DeviceBuffer.h>

#include <cuda_runtime.h>

// Forward declarations of the nvcc-compiled kernel launchers
namespace VarProGPU {

extern void obliqueProjectTangentImpl(
    const double* Y, double* V, int n, int r, int lda, cudaStream_t stream);

extern void obliqueRetractImpl(
    const double* Y, const double* V, double* out,
    int n, int r, cudaStream_t stream);

extern void stiefelProjectTangentImpl(
    const double* Y, double* V, int n, int K, int R, int lda, cudaStream_t stream);

extern void checkFiniteImpl(
    const double* data, int n, int* flag_dev, cudaStream_t stream);

extern void stiefelCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int K, int R, int lda, cudaStream_t stream);

extern void scaledStiefelCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int K, int R, int lda, cudaStream_t stream);

extern void scaledStiefelProjectTangentImpl(
    const double* Y, double* V, int n, int K, int R, int lda, cudaStream_t stream);

extern void obliqueCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int R, int lda, cudaStream_t stream);

extern void blockDiagMultiplyImpl(
    const double* B, const double* r, double* z,
    int n, int K, int R, int lda, cudaStream_t stream);

extern void permuteRowsImpl(const double* in, double* out, const int* perm,
                            int n, int cols, int lda_in, int lda_out,
                            cudaStream_t stream);

extern void invPermuteRowsImpl(const double* in, double* out, const int* perm,
                               int n, int cols, int lda_in, int lda_out,
                               cudaStream_t stream);

}  // namespace VarProGPU

// ---------------------------------------------------------------------------
// Public API (declared in ManifoldKernels.h)
// ---------------------------------------------------------------------------

namespace VarProGPU {

void obliqueProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int r, cudaStream_t stream, int lda) {
  obliqueProjectTangentImpl(Y_dev, V_dev, n, r, lda, stream);
  CUDA_CHECK(cudaGetLastError());
}

void obliqueRetract(
    const double* Y_dev, const double* V_dev,
    double* out_dev,
    int n, int r, cudaStream_t stream) {
  obliqueRetractImpl(Y_dev, V_dev, out_dev, n, r, stream);
  CUDA_CHECK(cudaGetLastError());
}

void stiefelProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int K, int R, cudaStream_t stream, int lda) {
  if (K >= 2 && K <= 4) {
    stiefelProjectTangentImpl(Y_dev, V_dev, n, K, R, lda, stream);
    CUDA_CHECK(cudaGetLastError());
  }
  // K out of range: no-op (CPU fallback used by caller)
}

void stiefelCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_poses, int K, int R, int lda, cudaStream_t stream) {
  if (K >= 2 && K <= 4) {
    stiefelCurvatureCorrectionImpl(Y, egrad, eta, Hp, n_poses, K, R, lda, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

void scaledStiefelCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_poses, int K, int R, int lda, cudaStream_t stream) {
  if (K >= 2 && K <= 4) {
    scaledStiefelCurvatureCorrectionImpl(Y, egrad, eta, Hp, n_poses, K, R, lda, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

void scaledStiefelProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int K, int R, cudaStream_t stream, int lda) {
  if (K >= 2 && K <= 4) {
    scaledStiefelProjectTangentImpl(Y_dev, V_dev, n, K, R, lda, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

void obliqueCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_range, int R, int lda, cudaStream_t stream) {
  obliqueCurvatureCorrectionImpl(Y, egrad, eta, Hp, n_range, R, lda, stream);
  CUDA_CHECK(cudaGetLastError());
}

void blockDiagMultiply(
    const double* B_dev, const double* r_dev, double* z_dev,
    int n, int K, int R, int lda, cudaStream_t stream) {
  if (K >= 2 && K <= 4) {
    blockDiagMultiplyImpl(B_dev, r_dev, z_dev, n, K, R, lda, stream);
    CUDA_CHECK(cudaGetLastError());
  }
}

void permuteRows(const double* in, double* out, const int* perm,
                 int n, int cols, int lda_in, int lda_out,
                 cudaStream_t stream) {
  permuteRowsImpl(in, out, perm, n, cols, lda_in, lda_out, stream);
  CUDA_CHECK(cudaGetLastError());
}

void invPermuteRows(const double* in, double* out, const int* perm,
                    int n, int cols, int lda_in, int lda_out,
                    cudaStream_t stream) {
  invPermuteRowsImpl(in, out, perm, n, cols, lda_in, lda_out, stream);
  CUDA_CHECK(cudaGetLastError());
}

bool hasNanOrInf(const double* dev_ptr, int n, cudaStream_t stream) {
  DeviceBuffer<int> flag(1);
  flag.zero();
  checkFiniteImpl(dev_ptr, n, flag.get(), stream);
  CUDA_CHECK(cudaGetLastError());
  int h_flag = 0;
  CUDA_CHECK(cudaMemcpyAsync(&h_flag, flag.get(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h_flag != 0;
}

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
