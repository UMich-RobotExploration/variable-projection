/**
 * @file ManifoldKernels.h
 * @brief Device-side declarations for manifold operations.
 *
 * These functions wrap the CUDA kernels defined in manifold_kernels.cu.
 * All pointers are device pointers.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/DeviceBuffer.h>
#include <cuda_runtime.h>

namespace VarProGPU {

// ---------------------------------------------------------------------------
// Oblique manifold  (product of unit spheres S^{r-1})
// Y ∈ R^{n×r}: each row is a unit vector
// ---------------------------------------------------------------------------

/// In-place tangent projection: V_i ← V_i − (y_i · v_i) y_i  for each row i
/// lda: column stride of Y and V; 0 means use natural stride n.
void obliqueProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int r, cudaStream_t stream, int lda = 0);

/// Retraction: out_i = (y_i + v_i) / ||y_i + v_i||
void obliqueRetract(
    const double* Y_dev, const double* V_dev,
    double* out_dev,
    int n, int r, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Stiefel manifold  (product of orthonormal frames St(K,R)^n)
// Y ∈ R^{(n*K)×R}: block i is the K×R frame for pose i
// ---------------------------------------------------------------------------

/// In-place tangent projection per block: V_i ← V_i − sym(Y_i V_i^T) Y_i
/// lda: column stride of the full matrix; 0 means use natural stride n*K.
void stiefelProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int K, int R, cudaStream_t stream, int lda = 0);

// ---------------------------------------------------------------------------
// Riemannian Hessian curvature corrections (applied BEFORE tangent projection)
// ---------------------------------------------------------------------------

/// Stiefel: Hp_i -= sym(Y_i * egrad_i^T) * eta_i  for each pose i
void stiefelCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_poses, int K, int R, int lda, cudaStream_t stream);

/// Scaled Stiefel: Hp_i -= (aniso_sym(Y_i * egrad_i^T) / s_i^2) * eta_i
/// where aniso_sym(M) = sym(M) − tr(sym(M))/K · I_K and s_i^2 = ||Y_i||_F^2 / K.
void scaledStiefelCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_poses, int K, int R, int lda, cudaStream_t stream);

/// Scaled Stiefel tangent projection:
///   V_i ← V_i − (aniso_sym(Y_i V_i^T) / s_i^2) Y_i  for each pose i
void scaledStiefelProjectTangent(
    const double* Y_dev, double* V_dev,
    int n, int K, int R, cudaStream_t stream, int lda = 0);

/// Oblique: Hp_i -= eta_i * (egrad_i · Y_i)  for each row i
void obliqueCurvatureCorrection(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n_range, int R, int lda, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Block-diagonal multiply for preconditioner
// z_i[:, c] = B_i * r_i[:, c]  for each pose i and column c
// B: flat array of n × (K*K) doubles (one K×K block per pose, row-major)
// lda: column stride of r and z; 0 means use natural stride n*K.
// ---------------------------------------------------------------------------

void blockDiagMultiply(
    const double* B_dev, const double* r_dev, double* z_dev,
    int n, int K, int R, int lda, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Row permutation for GPU Cholesky solve
// ---------------------------------------------------------------------------

/// out[perm[i], c] = in[i, c] for all rows i and columns c
void permuteRows(const double* in, double* out, const int* perm,
                 int n, int cols, int lda_in, int lda_out,
                 cudaStream_t stream);

/// out[i, c] = in[perm[i], c] for all rows i and columns c
void invPermuteRows(const double* in, double* out, const int* perm,
                    int n, int cols, int lda_in, int lda_out,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Debug / validation
// ---------------------------------------------------------------------------

/// Returns true if any element of the device array is NaN or infinite
bool hasNanOrInf(const double* dev_ptr, int n, cudaStream_t stream);

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
