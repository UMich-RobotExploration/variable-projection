/**
 * @file GpuSparseSolver.h
 * @brief Device-resident sparse SPD direct solver (cuDSS) for M z = b.
 *
 * The marginalized formulations need M^{-1} applied once per Hessian-vector
 * product, where M = C^T Omega C is the (gauge-pinned) translation block.
 * Historically that solve ran on the host via CHOLMOD, which forced a
 * synchronize + download + host solve + upload on *every* inner tCG iteration
 * -- up to 80 per outer iteration. That single round-trip is what capped the
 * GPU speedup at ~1.2-1.4x, while Formulation::Dense (which folds M^{-1} into
 * Q_sc at formation and therefore never round-trips) reached 3.6-16.6x on the
 * same problems.
 *
 * This wrapper keeps the whole solve on device:
 *   - analysis + numeric factorization run once at construction
 *   - solve() runs CUDSS_PHASE_SOLVE on the context's stream, no host transfer
 *
 * M is symmetric positive definite (a graph Laplacian with one translation
 * pinned), so we declare CUDSS_MTYPE_SPD and hand cuDSS the full CSR.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarPro/Types.h>

#include <memory>
#include <string>

namespace VarProGPU {

/// True when the build has cuDSS available (see VARPRO_HAVE_CUDSS).
bool gpuSparseSolverAvailable();

/**
 * @brief cuDSS-backed Cholesky solve, resident on the device.
 *
 * Construction is expensive (reordering + symbolic + numeric factorization);
 * solve() is cheap and is the only thing on the hot path. The factorization is
 * reused across all solves, exactly as the CHOLMOD factor was.
 */
class GpuSparseSolver {
 public:
  /**
   * @param A    symmetric positive definite matrix, in Eigen row-major CSR.
   *             Only the values/structure are read; no reference is retained.
   * @param ctx  GPU context; the solver runs on ctx.stream.
   */
  GpuSparseSolver(const VarPro::SparseMatrix& A, GpuContext& ctx);
  ~GpuSparseSolver();

  GpuSparseSolver(const GpuSparseSolver&) = delete;
  GpuSparseSolver& operator=(const GpuSparseSolver&) = delete;

  /**
   * @brief X <- A^{-1} B, entirely on device.
   *
   * B and X are n x nrhs column-major device matrices. X is resized if needed.
   * Does not synchronize -- the work is enqueued on the context stream.
   */
  void solve(const GpuDenseMatrix& B, GpuDenseMatrix& X) const;

  /**
   * @brief Refactor in place after the values of A change but its sparsity
   * pattern does not -- the IRLS case, where reweighting rewrites M every
   * outer iteration. Skips reordering and symbolic analysis.
   */
  void refactorize(const VarPro::SparseMatrix& A);

  int rows() const { return n_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int n_{0};
};

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
