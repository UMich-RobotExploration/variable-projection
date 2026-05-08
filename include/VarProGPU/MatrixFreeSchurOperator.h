/**
 * @file MatrixFreeSchurOperator.h
 * @brief Matrix-free Schur complement operator matching the paper's formulation.
 *
 * The paper (Section 3.2) defines the reduced cost as:
 *
 *   f(X_c) = 0.5 * tr( X_c^T  Q_sc  X_c )
 *
 * where the Schur complement operator is applied matrix-free:
 *
 *   Q_sc * X_c = Q_c * X_c - B * (M^{-1} B^T X_c)
 *
 * with precomputed M = C^T Ω C  (Cholesky: M = L L^T) and B = A_c^T Ω C.
 *
 * Connection to existing code (src/Problem.cpp, Formulation::Implicit):
 *   Q_c   ↔  Qmain_
 *   B^T   ↔  TransOffDiagRed_  (off-diagonal block, last column removed)
 *   L     ↔  LtransCholRed_
 *
 * This header provides:
 *   - VarProPrecomputeResult: the one-time preprocessing output
 *   - CpuSchurOperator:       reference (CPU-only) implementation
 *   - GpuSchurOperator:       GPU-accelerated implementation (when CUDA available)
 *   - ExplicitSchurReference: dense explicit Schur for small-problem unit tests
 */

#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <memory>
#include <stdexcept>
#include <string>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/ManifoldKernels.h>
#endif

namespace VarProGPU {

// ---------------------------------------------------------------------------
// VarProPrecomputeResult
// One-time preprocessing output.  All fields are immutable after construction.
// ---------------------------------------------------------------------------

struct VarProPrecomputeResult {
  // CPU sparse matrices (row-major CSR)
  VarPro::SparseMatrix Qmain;           ///< Upper-left block Q_c  (p × p)
  VarPro::SparseMatrix TransOffDiagRed; ///< B^T block (p × m), last col removed
  VarPro::CholFactorPtr LtransCholRed;  ///< Cholesky of M = C^T Ω C  (m × m)

  // Dimensions
  int p{0};  ///< rows of X_c (dim*n_poses + n_ranges)
  int m{0};  ///< cols of TransOffDiagRed (n_translations − 1)
  int r{0};  ///< relaxation rank

  // Manifold structure (needed for GPU tangent projections)
  int n_rot{0};    ///< number of rotation rows = dim * n_poses  (Stiefel block)
  int n_range{0};  ///< number of range rows = n_range measurements (Oblique block)
  int K{0};        ///< rotation dimension (dim_), e.g. 2 or 3
  int n_poses{0};  ///< number of pose variables


  // Validation
  void check() const {
    if (p <= 0 || m <= 0 || r <= 0)
      throw std::runtime_error("VarProPrecomputeResult: invalid dimensions");
    if (!LtransCholRed)
      throw std::runtime_error(
          "VarProPrecomputeResult: Cholesky factor not computed");
    if (Qmain.rows() != p || Qmain.cols() != p)
      throw std::runtime_error("VarProPrecomputeResult: Qmain shape mismatch");
    if (TransOffDiagRed.rows() != p || TransOffDiagRed.cols() != m)
      throw std::runtime_error(
          "VarProPrecomputeResult: TransOffDiagRed shape mismatch");
  }
};

// ---------------------------------------------------------------------------
// Build a VarProPrecomputeResult from an already-constructed Problem
// The problem must have been initialized with Formulation::Implicit and
// updateProblemData() must have been called.
// ---------------------------------------------------------------------------

inline VarProPrecomputeResult buildPrecomputeResult(VarPro::Problem& prob) {
  if (prob.getFormulation() != VarPro::Formulation::Implicit)
    throw std::runtime_error(
        "buildPrecomputeResult: problem must use Formulation::Implicit");

  VarProPrecomputeResult res;
  res.Qmain           = prob.Qmain_;
  res.TransOffDiagRed = prob.TransOffDiagRed_;
  res.LtransCholRed   = prob.LtransCholRed_;
  res.p               = prob.rotAndRangeMatrixSize();
  res.m               = prob.numTranslationalStates() - 1;
  res.r               = static_cast<int>(prob.getRelaxationRank());
  res.n_rot           = prob.numPosesDim();
  res.n_range         = prob.numRangeMeasurements();
  res.K               = prob.dim();
  res.n_poses         = prob.numPoses();

  // GPU triangular solve: disabled for now.
  // SimplicialLLT and CHOLMOD produce numerically different factors on these
  // matrices. Moving to GPU requires extracting L from the CHOLMOD factor
  // directly (via cholmod_factor_to_sparse C API), which is future work.
  // The CPU CHOLMOD solve in applyDevice is the only remaining D2H/H2D.

  res.check();
  return res;
}

// ---------------------------------------------------------------------------
// CpuSchurOperator
// CPU reference implementation — always available, used for tests.
// ---------------------------------------------------------------------------

class CpuSchurOperator {
 public:
  explicit CpuSchurOperator(const VarProPrecomputeResult& pre) : pre_(pre) {
    pre_.check();
  }

  /**
   * @brief  Q_sc * X_c  (matrix-free Schur product)
   *
   * Implements: Q_sc X = Qmain X − TransOffDiagRed · L^{-1}(L^{-T}(TransOffDiagRed^T X))
   *
   * This is the paper's eq. for the reduced Hessian-vector product (Section 3.2).
   * Equivalent to Problem::dataMatrixProduct() in Formulation::Implicit.
   */
  VarPro::Matrix apply(const VarPro::Matrix& X) const {
    // P1 = TransOffDiagRed^T * X   (m × r)
    VarPro::Matrix P1 = pre_.TransOffDiagRed.transpose() * X;
    // P2 = L^{-1}(L^{-T} P1)       (m × r)
    VarPro::Matrix P2 = pre_.LtransCholRed->solve(P1);
    // QX = Qmain * X                (p × r)
    VarPro::Matrix QX = pre_.Qmain * X;
    // QX -= TransOffDiagRed * P2
    QX -= pre_.TransOffDiagRed * P2;
    return QX;
  }

  VarPro::Scalar cost(const VarPro::Matrix& X) const {
    return 0.5 * (X.transpose() * apply(X)).trace();
  }

  VarPro::Matrix gradient(const VarPro::Matrix& X) const {
    return apply(X);
  }

  /**
   * @brief Hessian-vector product: Hess[f](X)[eta] = Q_sc * eta
   * (The Hessian of the quadratic f(X) = 0.5*tr(X^T Q_sc X) is exactly Q_sc.)
   */
  VarPro::Matrix hessianVectorProduct(const VarPro::Matrix& /*X*/,
                                       const VarPro::Matrix& eta) const {
    return apply(eta);
  }

  const VarProPrecomputeResult& precompute() const { return pre_; }

 private:
  const VarProPrecomputeResult& pre_;
};

// ---------------------------------------------------------------------------
// ExplicitSchurReference
// Forms Q_sc explicitly as a dense matrix.  Only used on SMALL problems for
// unit-testing the matrix-free operator.
// ---------------------------------------------------------------------------

class ExplicitSchurReference {
 public:
  explicit ExplicitSchurReference(const VarProPrecomputeResult& pre) {
    // Build dense Q_sc = Qmain - TransOffDiagRed * M^{-1} * TransOffDiagRed^T
    // where M^{-1} is the Cholesky solve.
    int p = pre.p, m = pre.m;

    // Solve for M^{-1} * TransOffDiagRed^T by treating it column by column
    // (equivalently, solve M * X = TransOffDiagRed^T using the stored factor)
    VarPro::Matrix rhs = VarPro::Matrix(pre.TransOffDiagRed.transpose());
    VarPro::Matrix sol = pre.LtransCholRed->solve(rhs);  // m × p

    Qsc_ = VarPro::Matrix(pre.Qmain) -
           VarPro::Matrix(pre.TransOffDiagRed) * sol;
  }

  VarPro::Matrix apply(const VarPro::Matrix& X) const { return Qsc_ * X; }

  const VarPro::Matrix& matrix() const { return Qsc_; }

 private:
  VarPro::Matrix Qsc_;
};

// ---------------------------------------------------------------------------
// GpuSchurOperator  (only when CUDA is available)
// ---------------------------------------------------------------------------

#ifdef VARPRO_HAVE_CUDA

class GpuSchurOperator {
 public:
  /**
   * @brief Construct and upload sparse matrices to device.
   *
   * Uploads Qmain, TransOffDiagRed, and the Cholesky factor L (as CSC)
   * plus the fill-reducing permutation.  After construction, applyDevice()
   * runs entirely on the GPU (cuSPARSE SpMM + SpSM, no host round-trips).
   */
  explicit GpuSchurOperator(const VarProPrecomputeResult& pre,
                             GpuContext& ctx)
      : pre_(pre), ctx_(ctx) {
    pre_.check();

    // Upload Qmain to device
    uploadEigenSparse(Qmain_dev_, pre_.Qmain);

    // Upload TransOffDiagRed to device
    uploadEigenSparse(TransOffDiagRed_dev_, pre_.TransOffDiagRed);

    // Upload TransOffDiagRed^T to device (explicit transpose for SpMM efficiency)
    VarPro::SparseMatrix Bt_rm = pre_.TransOffDiagRed.transpose();
    uploadEigenSparse(TransOffDiagRedT_dev_, Bt_rm);

    p_ = pre_.p;
    m_ = pre_.m;
    r_ = pre_.r;
  }

  ~GpuSchurOperator() = default;

  // Non-copyable
  GpuSchurOperator(const GpuSchurOperator&) = delete;
  GpuSchurOperator& operator=(const GpuSchurOperator&) = delete;

  VarPro::Matrix apply(const VarPro::Matrix& X) const {
    X_dev_.upload(X);
    applyDevice(X_dev_, Y_dev_);
    ctx_.synchronize();
    return Y_dev_.download();
  }

  /**
   * @brief Apply Q_sc * X entirely on device (no host round-trips).
   *
   *   P1 = B^T * X            (cuSPARSE SpMM)
   *   P2 = (P^T L^{-T} L^{-1} P) P1   (cuSPARSE SpSM + permute kernels)
   *   QX = Q_c * X            (cuSPARSE SpMM)
   *   Y  = QX − B * P2       (cuSPARSE SpMM + cuBLAS daxpy)
   */
  void applyDevice(const GpuDenseMatrix& X, GpuDenseMatrix& Y) const {
    int cols = X.cols;

    // Ensure scratch buffers
    if (P1_dev_.rows != m_ || P1_dev_.cols != cols)
      P1_dev_.resize(m_, cols);
    if (P2_dev_.rows != m_ || P2_dev_.cols != cols)
      P2_dev_.resize(m_, cols);
    if (Y.rows != p_ || Y.cols != cols)
      Y.resize(p_, cols);

    // P1 = TransOffDiagRed^T * X    (m × cols)
    spmmCSR(ctx_, TransOffDiagRedT_dev_, X, P1_dev_);

    // CPU triangular solve: P2 = (L L^T)^{-1} P1 via CHOLMOD
    ctx_.synchronize();
    VarPro::Matrix P1_host = P1_dev_.download();
    VarPro::Matrix P2_host = pre_.LtransCholRed->solve(P1_host);
    P2_dev_.upload(P2_host);

    // QX = Qmain * X                (p × cols), stored in Y
    spmmCSR(ctx_, Qmain_dev_, X, Y);

    // P3 = TransOffDiagRed * P2
    if (P3_dev_.rows != p_ || P3_dev_.cols != cols)
      P3_dev_.resize(p_, cols);
    spmmCSR(ctx_, TransOffDiagRed_dev_, P2_dev_, P3_dev_);

    // Y = Y - P3
    daxpy(ctx_, -1.0, P3_dev_, Y);
  }

  VarPro::Scalar cost(const VarPro::Matrix& X) const {
    VarPro::Matrix QscX = apply(X);
    return 0.5 * (X.transpose() * QscX).trace();
  }

  VarPro::Matrix gradient(const VarPro::Matrix& X) const {
    return apply(X);
  }

  VarPro::Matrix hessianVectorProduct(const VarPro::Matrix& /*X*/,
                                       const VarPro::Matrix& eta) const {
    return apply(eta);
  }

  GpuContext& context() const { return ctx_; }
  const VarProPrecomputeResult& precompute() const { return pre_; }

 private:
  const VarProPrecomputeResult& pre_;
  GpuContext& ctx_;

  // Device-side sparse matrices (uploaded once at construction)
  GpuCsrMatrix Qmain_dev_;
  GpuCsrMatrix TransOffDiagRed_dev_;
  GpuCsrMatrix TransOffDiagRedT_dev_;

  // Scratch buffers (mutable — resized lazily)
  mutable GpuDenseMatrix X_dev_;    // p × r  (for host→device apply())
  mutable GpuDenseMatrix Y_dev_;    // p × r
  mutable GpuDenseMatrix P1_dev_;   // m × r
  mutable GpuDenseMatrix P2_dev_;   // m × r
  mutable GpuDenseMatrix P3_dev_;   // p × r

  int p_, m_, r_;
};

// ---------------------------------------------------------------------------
// GpuExplicitOperator — for Explicit and ExplicitVarPro formulations
// The data matrix product is a single SpMM:  Y = data_matrix * X
// No triangular solve needed.
// ---------------------------------------------------------------------------

class GpuExplicitOperator {
 public:
  explicit GpuExplicitOperator(VarPro::Problem& prob, GpuContext& ctx)
      : prob_(prob), ctx_(ctx) {
    uploadEigenSparse(Q_dev_, prob.data_matrix_);
    n_ = prob.getDataMatrixSize();
    r_ = static_cast<int>(prob.getRelaxationRank());
  }

  void applyDevice(const GpuDenseMatrix& X, GpuDenseMatrix& Y) const {
    if (Y.rows != n_ || Y.cols != X.cols)
      Y.resize(n_, X.cols);
    spmmCSR(ctx_, Q_dev_, X, Y);
  }

  VarPro::Matrix apply(const VarPro::Matrix& X) const {
    X_dev_.upload(X);
    applyDevice(X_dev_, Y_dev_);
    ctx_.synchronize();
    return Y_dev_.download();
  }

  GpuContext& context() const { return ctx_; }
  VarPro::Problem& problem() const { return prob_; }
  int rows() const { return n_; }

 private:
  VarPro::Problem& prob_;
  GpuContext& ctx_;
  GpuCsrMatrix Q_dev_;
  mutable GpuDenseMatrix X_dev_;
  mutable GpuDenseMatrix Y_dev_;
  int n_, r_;
};

#endif  // VARPRO_HAVE_CUDA

}  // namespace VarProGPU
