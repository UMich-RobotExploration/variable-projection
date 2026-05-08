/**
 * @file manifold_kernels.cu
 * @brief Custom CUDA kernels for manifold operations.
 *
 * Keep this file minimal: only CUDA kernel code, no complex C++ headers.
 * The host-side wrappers live in src/gpu/ManifoldOps.cpp.
 */

#ifdef VARPRO_HAVE_CUDA

#include <cuda_runtime.h>
#include <math.h>

// Expose the host-callable launchers.  Declarations only — no complex headers.
namespace VarProGPU {

// ---------------------------------------------------------------------------
// Oblique tangent projection
// For each row i of Y (unit vector in R^r):
//   V_i ← V_i − (y_i · v_i) y_i
// Y, V: n×r column-major device arrays; lda = column stride (0 → use n)
// ---------------------------------------------------------------------------

__global__ void oblique_tangent_project_kernel(
    const double* __restrict__ Y,
    double* __restrict__ V,
    int n, int r, int lda) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double dot = 0.0;
  for (int c = 0; c < r; ++c)
    dot += Y[i + (long long)c * lda] * V[i + (long long)c * lda];

  for (int c = 0; c < r; ++c)
    V[i + (long long)c * lda] -= dot * Y[i + (long long)c * lda];
}

// lda: column stride; 0 means use natural stride n.
void obliqueProjectTangentImpl(
    const double* Y, double* V, int n, int r, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n;
  int block = 256;
  int grid  = (n + block - 1) / block;
  oblique_tangent_project_kernel<<<grid, block, 0, stream>>>(Y, V, n, r, lda);
}

// ---------------------------------------------------------------------------
// Oblique retraction
// out_i = (y_i + v_i) / ||y_i + v_i||
// ---------------------------------------------------------------------------

__global__ void oblique_retract_kernel(
    const double* __restrict__ Y,
    const double* __restrict__ V,
    double* __restrict__ out,
    int n, int r) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double norm_sq = 0.0;
  for (int c = 0; c < r; ++c) {
    double val = Y[i + (long long)c * n] + V[i + (long long)c * n];
    norm_sq += val * val;
  }
  double inv_norm = rsqrt(norm_sq);
  for (int c = 0; c < r; ++c)
    out[i + (long long)c * n] =
        (Y[i + (long long)c * n] + V[i + (long long)c * n]) * inv_norm;
}

void obliqueRetractImpl(
    const double* Y, const double* V, double* out,
    int n, int r, cudaStream_t stream) {
  int block = 256;
  int grid  = (n + block - 1) / block;
  oblique_retract_kernel<<<grid, block, 0, stream>>>(Y, V, out, n, r);
}

// ---------------------------------------------------------------------------
// Stiefel tangent projection (per K×R block)
// For block i (rows [i*K .. i*K+K-1]):
//   M = Y_i * V_i^T          (K×K matrix, uses ALL R columns)
//   S = sym(M) = (M+M^T)/2   (K×K symmetric)
//   V_i ← V_i − S * Y_i
//
// This implements the correct Stiefel tangent projection matching the CPU:
//   proj_Y(V) = V - sym(Y V^T) Y
// where Y_i ∈ R^{K×R} satisfies Y_i Y_i^T = I_K.
//
// One thread per column, one warp per pose.  R ≤ 32 required.
// lda = column stride of the full matrix (0 → use natural stride n*K).
// ---------------------------------------------------------------------------

template<int K>
__global__ void stiefel_tangent_project_kernel(
    const double* __restrict__ Y,
    double* __restrict__ V,
    int n, int R, int lda) {

  int pose = blockIdx.x;
  if (pose >= n) return;

  int base = pose * K;
  int c = threadIdx.x;

  // Load this thread's Y and V columns (K values each)
  double y[K], v[K];
  if (c < R) {
    for (int k = 0; k < K; ++k) {
      y[k] = Y[(base + k) + (long long)c * lda];
      v[k] = V[(base + k) + (long long)c * lda];
    }
  } else {
    for (int k = 0; k < K; ++k) { y[k] = 0.0; v[k] = 0.0; }
  }

  // Compute M = Y_i * V_i^T via warp shuffle reduction.
  // Each thread contributes rank-1 outer product y * v^T.
  double m[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      m[j * K + k] = y[j] * v[k];

  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset /= 2)
    for (int i = 0; i < K * K; ++i)
      m[i] += __shfl_down_sync(mask, m[i], offset);

  // Broadcast M from thread 0 to all threads
  for (int i = 0; i < K * K; ++i)
    m[i] = __shfl_sync(mask, m[i], 0);

  // Symmetrize: S = (M + M^T) / 2
  double s[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      s[j * K + k] = 0.5 * (m[j * K + k] + m[k * K + j]);

  // Apply correction: V_i[:, c] -= S * Y_i[:, c]
  if (c < R) {
    for (int k = 0; k < K; ++k) {
      double correction = 0.0;
      for (int j = 0; j < K; ++j)
        correction += s[k * K + j] * y[j];
      V[(base + k) + (long long)c * lda] -= correction;
    }
  }
}

// lda: column stride; 0 means use natural stride n*K.
void stiefelProjectTangentImpl(
    const double* Y, double* V, int n, int K, int R, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n * K;
  // Launch with R threads per block (one per column), n blocks (one per pose)
  int threads = min(R, 32);
  if (K == 2) {
    stiefel_tangent_project_kernel<2><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
  } else if (K == 3) {
    stiefel_tangent_project_kernel<3><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
  } else if (K == 4) {
    stiefel_tangent_project_kernel<4><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
  }
  // K=1 or K>4 handled by CPU fallback
}

// ---------------------------------------------------------------------------
// Stiefel curvature correction for Riemannian Hessian
// For each pose i:  Hp_i -= sym(Y_i * egrad_i^T) * eta_i
// Y, egrad, eta, Hp all have lda as column stride; Stiefel block occupies
// rows [i*K .. i*K+K-1].  One thread per column, one warp per pose.
// ---------------------------------------------------------------------------

template<int K>
__global__ void stiefel_curvature_kernel(
    const double* __restrict__ Y,
    const double* __restrict__ egrad,
    const double* __restrict__ eta,
    double* __restrict__ Hp,
    int n, int R, int lda) {

  int pose = blockIdx.x;
  if (pose >= n) return;
  int c = threadIdx.x;
  int base = pose * K;

  // Load Y_i[:,c] and egrad_i[:,c]
  double y[K], eg[K];
  if (c < R) {
    for (int k = 0; k < K; ++k) {
      y[k]  = Y[(base + k) + (long long)c * lda];
      eg[k] = egrad[(base + k) + (long long)c * lda];
    }
  } else {
    for (int k = 0; k < K; ++k) { y[k] = 0.0; eg[k] = 0.0; }
  }

  // P = Y_i * egrad_i^T (K×K) via warp reduction of outer products
  double p[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      p[j * K + k] = y[j] * eg[k];

  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset /= 2)
    for (int i = 0; i < K * K; ++i)
      p[i] += __shfl_down_sync(mask, p[i], offset);
  for (int i = 0; i < K * K; ++i)
    p[i] = __shfl_sync(mask, p[i], 0);

  // S = sym(P)
  double s[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      s[j * K + k] = 0.5 * (p[j * K + k] + p[k * K + j]);

  // Hp_i[:,c] -= S * eta_i[:,c]
  if (c < R) {
    double et[K];
    for (int k = 0; k < K; ++k)
      et[k] = eta[(base + k) + (long long)c * lda];
    for (int k = 0; k < K; ++k) {
      double corr = 0.0;
      for (int j = 0; j < K; ++j)
        corr += s[k * K + j] * et[j];
      Hp[(base + k) + (long long)c * lda] -= corr;
    }
  }
}

void stiefelCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int K, int R, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n * K;
  int threads = min(R, 32);
  if (K == 2) stiefel_curvature_kernel<2><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
  else if (K == 3) stiefel_curvature_kernel<3><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
  else if (K == 4) stiefel_curvature_kernel<4><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
}

// ---------------------------------------------------------------------------
// Scaled Stiefel curvature correction for Riemannian Hessian
// CPU analogue: ScaledStiefelProduct::SymBlockDiagProduct_aniso.
// For each pose i (Y_i ∈ R^{K×R}, with Y_i = s_i R_i, R_i R_i^T = I_K):
//   P     = Y_i * egrad_i^T                        (K×K)
//   S     = sym(P) − (tr(sym(P))/K) I_K            (aniso symmetric)
//   s_i^2 = ||Y_i||_F^2 / K
//   Hp_i -= (S / s_i^2) * eta_i
// One thread per column, one warp per pose; R ≤ 32.
// ---------------------------------------------------------------------------

template<int K>
__global__ void scaled_stiefel_curvature_kernel(
    const double* __restrict__ Y,
    const double* __restrict__ egrad,
    const double* __restrict__ eta,
    double* __restrict__ Hp,
    int n, int R, int lda) {

  int pose = blockIdx.x;
  if (pose >= n) return;
  int c = threadIdx.x;
  int base = pose * K;

  double y[K], eg[K];
  if (c < R) {
    for (int k = 0; k < K; ++k) {
      y[k]  = Y[(base + k) + (long long)c * lda];
      eg[k] = egrad[(base + k) + (long long)c * lda];
    }
  } else {
    for (int k = 0; k < K; ++k) { y[k] = 0.0; eg[k] = 0.0; }
  }

  // Compute P = Y_i * egrad_i^T (K×K) and y_sq = ||Y_i||_F^2 (scalar) jointly.
  double p[K * K];
  double y_sq = 0.0;
  for (int j = 0; j < K; ++j) {
    y_sq += y[j] * y[j];
    for (int k = 0; k < K; ++k)
      p[j * K + k] = y[j] * eg[k];
  }

  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset /= 2) {
    for (int i = 0; i < K * K; ++i)
      p[i] += __shfl_down_sync(mask, p[i], offset);
    y_sq += __shfl_down_sync(mask, y_sq, offset);
  }
  for (int i = 0; i < K * K; ++i)
    p[i] = __shfl_sync(mask, p[i], 0);
  y_sq = __shfl_sync(mask, y_sq, 0);

  // S = sym(P) − tr(sym(P))/K · I_K, scaled by 1/s_i^2 = K / y_sq.
  double s[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      s[j * K + k] = 0.5 * (p[j * K + k] + p[k * K + j]);
  double trace_sym = 0.0;
  for (int j = 0; j < K; ++j) trace_sym += s[j * K + j];
  for (int j = 0; j < K; ++j)
    s[j * K + j] -= trace_sym / static_cast<double>(K);
  double inv_s_sq = (y_sq > 0.0) ? (static_cast<double>(K) / y_sq) : 0.0;
  for (int i = 0; i < K * K; ++i)
    s[i] *= inv_s_sq;

  if (c < R) {
    double et[K];
    for (int k = 0; k < K; ++k)
      et[k] = eta[(base + k) + (long long)c * lda];
    for (int k = 0; k < K; ++k) {
      double corr = 0.0;
      for (int j = 0; j < K; ++j)
        corr += s[k * K + j] * et[j];
      Hp[(base + k) + (long long)c * lda] -= corr;
    }
  }
}

void scaledStiefelCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int K, int R, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n * K;
  int threads = min(R, 32);
  if (K == 2) scaled_stiefel_curvature_kernel<2><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
  else if (K == 3) scaled_stiefel_curvature_kernel<3><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
  else if (K == 4) scaled_stiefel_curvature_kernel<4><<<n, threads, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
}

// ---------------------------------------------------------------------------
// Scaled Stiefel tangent projection
// CPU analogue: ScaledStiefelProduct::projectToTangentSpace.
// For each pose i:
//   M     = Y_i * V_i^T                         (K×K)
//   S     = sym(M) − (tr(sym(M))/K) I_K          (aniso symmetric)
//   s_i^2 = ||Y_i||_F^2 / K
//   V_i  -= (S / s_i^2) * Y_i
// ---------------------------------------------------------------------------

template<int K>
__global__ void scaled_stiefel_tangent_project_kernel(
    const double* __restrict__ Y,
    double* __restrict__ V,
    int n, int R, int lda) {

  int pose = blockIdx.x;
  if (pose >= n) return;
  int base = pose * K;
  int c = threadIdx.x;

  double y[K], v[K];
  if (c < R) {
    for (int k = 0; k < K; ++k) {
      y[k] = Y[(base + k) + (long long)c * lda];
      v[k] = V[(base + k) + (long long)c * lda];
    }
  } else {
    for (int k = 0; k < K; ++k) { y[k] = 0.0; v[k] = 0.0; }
  }

  double m[K * K];
  double y_sq = 0.0;
  for (int j = 0; j < K; ++j) {
    y_sq += y[j] * y[j];
    for (int k = 0; k < K; ++k)
      m[j * K + k] = y[j] * v[k];
  }

  unsigned mask = __activemask();
  for (int offset = 16; offset > 0; offset /= 2) {
    for (int i = 0; i < K * K; ++i)
      m[i] += __shfl_down_sync(mask, m[i], offset);
    y_sq += __shfl_down_sync(mask, y_sq, offset);
  }
  for (int i = 0; i < K * K; ++i)
    m[i] = __shfl_sync(mask, m[i], 0);
  y_sq = __shfl_sync(mask, y_sq, 0);

  double s[K * K];
  for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
      s[j * K + k] = 0.5 * (m[j * K + k] + m[k * K + j]);
  double trace_sym = 0.0;
  for (int j = 0; j < K; ++j) trace_sym += s[j * K + j];
  for (int j = 0; j < K; ++j)
    s[j * K + j] -= trace_sym / static_cast<double>(K);
  double inv_s_sq = (y_sq > 0.0) ? (static_cast<double>(K) / y_sq) : 0.0;
  for (int i = 0; i < K * K; ++i)
    s[i] *= inv_s_sq;

  if (c < R) {
    for (int k = 0; k < K; ++k) {
      double corr = 0.0;
      for (int j = 0; j < K; ++j)
        corr += s[k * K + j] * y[j];
      V[(base + k) + (long long)c * lda] -= corr;
    }
  }
}

void scaledStiefelProjectTangentImpl(
    const double* Y, double* V, int n, int K, int R, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n * K;
  int threads = min(R, 32);
  if (K == 2) scaled_stiefel_tangent_project_kernel<2><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
  else if (K == 3) scaled_stiefel_tangent_project_kernel<3><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
  else if (K == 4) scaled_stiefel_tangent_project_kernel<4><<<n, threads, 0, stream>>>(Y, V, n, R, lda);
}

// ---------------------------------------------------------------------------
// Oblique curvature correction for Riemannian Hessian
// For each row i in the Oblique block:
//   d_i = sum_c egrad[i,c] * Y[i,c]   (dot product)
//   Hp[i,c] -= eta[i,c] * d_i
// ---------------------------------------------------------------------------

__global__ void oblique_curvature_kernel(
    const double* __restrict__ Y,
    const double* __restrict__ egrad,
    const double* __restrict__ eta,
    double* __restrict__ Hp,
    int n, int R, int lda) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double d = 0.0;
  for (int c = 0; c < R; ++c)
    d += egrad[i + (long long)c * lda] * Y[i + (long long)c * lda];

  for (int c = 0; c < R; ++c)
    Hp[i + (long long)c * lda] -= eta[i + (long long)c * lda] * d;
}

void obliqueCurvatureCorrectionImpl(
    const double* Y, const double* egrad, const double* eta, double* Hp,
    int n, int R, int lda, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  oblique_curvature_kernel<<<grid, block, 0, stream>>>(Y, egrad, eta, Hp, n, R, lda);
}

// ---------------------------------------------------------------------------
// Block-diagonal multiply for preconditioner
// For each pose i, apply K×K block: z_i[:, c] = B_i * r_i[:, c]
// B is stored as n_poses contiguous K×K blocks (row-major within each block),
// i.e. B[i*K*K + j*K + k] = B_i[j][k].
// lda = column stride of r and z (0 → n_poses*K).
// ---------------------------------------------------------------------------

template<int K>
__global__ void block_diag_multiply_kernel(
    const double* __restrict__ B,   // n × (K*K) block inverses
    const double* __restrict__ r,
    double* __restrict__ z,
    int n, int R, int lda) {

  int pose = blockIdx.x;
  if (pose >= n) return;
  int c = threadIdx.x;
  if (c >= R) return;

  int base = pose * K;
  const double* Bi = B + pose * K * K;

  double rv[K];
  for (int k = 0; k < K; ++k)
    rv[k] = r[(base + k) + (long long)c * lda];

  for (int j = 0; j < K; ++j) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k)
      acc += Bi[j * K + k] * rv[k];
    z[(base + j) + (long long)c * lda] = acc;
  }
}

void blockDiagMultiplyImpl(
    const double* B, const double* r, double* z,
    int n, int K, int R, int lda, cudaStream_t stream) {
  if (lda == 0) lda = n * K;
  int threads = min(R, 32);
  if (K == 2) {
    block_diag_multiply_kernel<2><<<n, threads, 0, stream>>>(B, r, z, n, R, lda);
  } else if (K == 3) {
    block_diag_multiply_kernel<3><<<n, threads, 0, stream>>>(B, r, z, n, R, lda);
  } else if (K == 4) {
    block_diag_multiply_kernel<4><<<n, threads, 0, stream>>>(B, r, z, n, R, lda);
  }
}

// ---------------------------------------------------------------------------
// Row permutation for Cholesky solve: out[perm[i], c] = in[i, c]
// and inverse: out[i, c] = in[perm[i], c]
// ---------------------------------------------------------------------------

__global__ void permute_rows_kernel(
    const double* __restrict__ in,
    double* __restrict__ out,
    const int* __restrict__ perm,
    int n, int cols, int lda_in, int lda_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int dst = perm[i];
  for (int c = 0; c < cols; ++c)
    out[dst + (long long)c * lda_out] = in[i + (long long)c * lda_in];
}

__global__ void inv_permute_rows_kernel(
    const double* __restrict__ in,
    double* __restrict__ out,
    const int* __restrict__ perm,
    int n, int cols, int lda_in, int lda_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int src = perm[i];
  for (int c = 0; c < cols; ++c)
    out[i + (long long)c * lda_out] = in[src + (long long)c * lda_in];
}

void permuteRowsImpl(const double* in, double* out, const int* perm,
                     int n, int cols, int lda_in, int lda_out,
                     cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  permute_rows_kernel<<<grid, block, 0, stream>>>(in, out, perm, n, cols, lda_in, lda_out);
}

void invPermuteRowsImpl(const double* in, double* out, const int* perm,
                        int n, int cols, int lda_in, int lda_out,
                        cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  inv_permute_rows_kernel<<<grid, block, 0, stream>>>(in, out, perm, n, cols, lda_in, lda_out);
}

// ---------------------------------------------------------------------------
// NaN/Inf detection
// ---------------------------------------------------------------------------

__global__ void check_finite_kernel(
    const double* __restrict__ data, int n, int* flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(data[i]))
    atomicExch(flag, 1);
}

void checkFiniteImpl(const double* data, int n, int* flag_dev,
                      cudaStream_t stream) {
  int block = 256;
  int grid  = (n + block - 1) / block;
  check_finite_kernel<<<grid, block, 0, stream>>>(data, n, flag_dev);
}

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
