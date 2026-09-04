/**
 * @file GpuSparseSolver.cpp
 * @brief cuDSS implementation of the device-resident sparse SPD solve.
 *
 * Compiled by g++ (cuDSS is a C API, like cuBLAS/cuSPARSE), matching the rest
 * of src/gpu/.
 */

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuSparseSolver.h>

#include <stdexcept>
#include <string>
#include <vector>

#ifdef VARPRO_HAVE_CUDSS
#include <cudss.h>
#endif

namespace VarProGPU {

#ifndef VARPRO_HAVE_CUDSS

// ---------------------------------------------------------------------------
// Build without cuDSS: the type still exists so callers compile, but any
// attempt to construct one is a hard error -- callers must check
// gpuSparseSolverAvailable() and fall back to the host CHOLMOD path.
// ---------------------------------------------------------------------------

bool gpuSparseSolverAvailable() { return false; }

struct GpuSparseSolver::Impl {};

GpuSparseSolver::GpuSparseSolver(const VarPro::SparseMatrix&, GpuContext&) {
  throw std::runtime_error(
      "GpuSparseSolver: built without cuDSS. Reconfigure with cuDSS available, "
      "or use the host CHOLMOD fallback.");
}
GpuSparseSolver::~GpuSparseSolver() = default;
void GpuSparseSolver::solve(const GpuDenseMatrix&, GpuDenseMatrix&) const {}
void GpuSparseSolver::refactorize(const VarPro::SparseMatrix&) {}

#else

bool gpuSparseSolverAvailable() { return true; }

#define CUDSS_CHECK(call)                                                    \
  do {                                                                       \
    cudssStatus_t st_ = (call);                                              \
    if (st_ != CUDSS_STATUS_SUCCESS) {                                       \
      throw std::runtime_error("cuDSS error " + std::to_string((int)st_) +   \
                               " at " __FILE__ ":" + std::to_string(__LINE__)); \
    }                                                                        \
  } while (0)

struct GpuSparseSolver::Impl {
  GpuContext* ctx{nullptr};
  cudssHandle_t handle{nullptr};
  cudssConfig_t config{nullptr};
  cudssData_t data{nullptr};
  cudssMatrix_t A{nullptr};

  // Device-side CSR storage for A (owned).
  DeviceBuffer<int> row_ptr;
  DeviceBuffer<int> col_idx;
  DeviceBuffer<double> values;

  // Reusable device wrappers for the rhs/solution; recreated when nrhs changes.
  mutable cudssMatrix_t b_mat{nullptr};
  mutable cudssMatrix_t x_mat{nullptr};
  mutable int cached_nrhs{-1};
  mutable int cached_n{-1};

  int n{0};
  int nnz{0};

  ~Impl() {
    if (b_mat) cudssMatrixDestroy(b_mat);
    if (x_mat) cudssMatrixDestroy(x_mat);
    if (A) cudssMatrixDestroy(A);
    if (data) cudssDataDestroy(handle, data);
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
  }

  // Eigen row-major CSR of a symmetric matrix is the same data as column-major
  // CSC, so handing cuDSS the full pattern is unambiguous either way.
  void uploadPattern(const VarPro::SparseMatrix& Asp) {
    VarPro::SparseMatrix Ac = Asp;
    Ac.makeCompressed();
    n = static_cast<int>(Ac.rows());
    nnz = static_cast<int>(Ac.nonZeros());

    std::vector<int> h_rp(n + 1), h_ci(nnz);
    const auto* outer = Ac.outerIndexPtr();
    const auto* inner = Ac.innerIndexPtr();
    for (int i = 0; i <= n; ++i) h_rp[i] = static_cast<int>(outer[i]);
    for (int i = 0; i < nnz; ++i) h_ci[i] = static_cast<int>(inner[i]);

    row_ptr.allocate(n + 1);
    col_idx.allocate(nnz);
    values.allocate(nnz);
    row_ptr.upload(h_rp.data(), n + 1);
    col_idx.upload(h_ci.data(), nnz);
    values.upload(Ac.valuePtr(), nnz);
  }

  void uploadValues(const VarPro::SparseMatrix& Asp) {
    VarPro::SparseMatrix Ac = Asp;
    Ac.makeCompressed();
    if (static_cast<int>(Ac.nonZeros()) != nnz)
      throw std::runtime_error(
          "GpuSparseSolver::refactorize: sparsity pattern changed");
    values.upload(Ac.valuePtr(), nnz);
  }

  void ensureRhsWrappers(int nrhs, const double* b, double* x, int ld) const {
    if (b_mat && x_mat && cached_nrhs == nrhs && cached_n == ld) {
      // Values pointers are baked into the wrapper, so they must still match.
      // We recreate unconditionally below if they do not; cheap either way.
    }
    if (b_mat) { cudssMatrixDestroy(b_mat); b_mat = nullptr; }
    if (x_mat) { cudssMatrixDestroy(x_mat); x_mat = nullptr; }
    CUDSS_CHECK(cudssMatrixCreateDn(&b_mat, ld, nrhs, ld, const_cast<double*>(b),
                                    CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&x_mat, ld, nrhs, ld, x, CUDSS_R_64F,
                                    CUDSS_LAYOUT_COL_MAJOR));
    cached_nrhs = nrhs;
    cached_n = ld;
  }
};

GpuSparseSolver::GpuSparseSolver(const VarPro::SparseMatrix& A, GpuContext& ctx)
    : impl_(new Impl()) {
  impl_->ctx = &ctx;
  CUDSS_CHECK(cudssCreate(&impl_->handle));
  CUDSS_CHECK(cudssSetStream(impl_->handle, ctx.stream.get()));
  CUDSS_CHECK(cudssConfigCreate(&impl_->config));
  CUDSS_CHECK(cudssDataCreate(impl_->handle, &impl_->data));

  impl_->uploadPattern(A);
  n_ = impl_->n;

  CUDSS_CHECK(cudssMatrixCreateCsr(
      &impl_->A, impl_->n, impl_->n, impl_->nnz, impl_->row_ptr.get(),
      /*rowEnd=*/nullptr, impl_->col_idx.get(), impl_->values.get(),
      CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL,
      CUDSS_BASE_ZERO));

  // Analysis + numeric factorization once; solves reuse both.
  CUDSS_CHECK(cudssExecute(impl_->handle, CUDSS_PHASE_ANALYSIS, impl_->config,
                           impl_->data, impl_->A, nullptr, nullptr));
  CUDSS_CHECK(cudssExecute(impl_->handle, CUDSS_PHASE_FACTORIZATION,
                           impl_->config, impl_->data, impl_->A, nullptr,
                           nullptr));
  ctx.synchronize();
}

GpuSparseSolver::~GpuSparseSolver() = default;

void GpuSparseSolver::solve(const GpuDenseMatrix& B, GpuDenseMatrix& X) const {
  if (B.rows != n_)
    throw std::runtime_error("GpuSparseSolver::solve: rhs row count mismatch");
  if (X.rows != n_ || X.cols != B.cols) X.resize(n_, B.cols);

  impl_->ensureRhsWrappers(B.cols, B.data.get(), X.data.get(), n_);
  CUDSS_CHECK(cudssExecute(impl_->handle, CUDSS_PHASE_SOLVE, impl_->config,
                           impl_->data, impl_->A, impl_->x_mat, impl_->b_mat));
}

void GpuSparseSolver::refactorize(const VarPro::SparseMatrix& A) {
  impl_->uploadValues(A);
  // Pattern is unchanged, so reuse the existing reordering/symbolic result.
  CUDSS_CHECK(cudssExecute(impl_->handle, CUDSS_PHASE_REFACTORIZATION,
                           impl_->config, impl_->data, impl_->A, nullptr,
                           nullptr));
  impl_->ctx->synchronize();
}

#undef CUDSS_CHECK

#endif  // VARPRO_HAVE_CUDSS

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
