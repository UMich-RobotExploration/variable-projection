/**
 * @file GpuDenseSchur.cpp
 * @brief Device-side formation of Q_sc = Qmain - B M^{-1} B^T.
 *
 * Compiled by g++ like the rest of src/gpu/ -- cuSPARSE and cuDSS are C APIs.
 */

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuDenseSchur.h>
#include <VarProGPU/GpuSparseSolver.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace VarProGPU {
namespace {

// Descriptor over memory we do not own (a column block of Q_sc, a slice of a
// CSR array). cuSPARSE descriptors are cheap to create; we make one per block
// rather than trying to mutate them in place.
struct DnMatView {
  cusparseDnMatDescr_t d{nullptr};
  DnMatView(int rows, int cols, int ld, double* ptr, cusparseOrder_t order) {
    CUSPARSE_CHECK(cusparseCreateDnMat(&d, rows, cols, ld, ptr, CUDA_R_64F,
                                       order));
  }
  ~DnMatView() { if (d) cusparseDestroyDnMat(d); }
  DnMatView(const DnMatView&) = delete;
  DnMatView& operator=(const DnMatView&) = delete;
};

struct SpMatView {
  cusparseSpMatDescr_t d{nullptr};
  SpMatView(int rows, int cols, int nnz, int* row_ptr, int* col_idx,
            double* values) {
    CUSPARSE_CHECK(cusparseCreateCsr(&d, rows, cols, nnz, row_ptr, col_idx,
                                     values, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  }
  ~SpMatView() { if (d) cusparseDestroySpMat(d); }
  SpMatView(const SpMatView&) = delete;
  SpMatView& operator=(const SpMatView&) = delete;
};

void sparseToDense(GpuContext& ctx, cusparseSpMatDescr_t A,
                   cusparseDnMatDescr_t out, SpmmWorkspace& ws) {
  std::size_t buf_size = 0;
  CUSPARSE_CHECK(cusparseSparseToDense_bufferSize(
      ctx.cusparse.get(), A, out, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
      &buf_size));
  ws.reserve(buf_size + 1);
  CUSPARSE_CHECK(cusparseSparseToDense(ctx.cusparse.get(), A, out,
                                       CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                       ws.buf.get()));
}

// Y = alpha * A * X + beta * Y, on raw descriptors.
void spmmView(GpuContext& ctx, cusparseSpMatDescr_t A, cusparseDnMatDescr_t X,
              cusparseDnMatDescr_t Y, double alpha, double beta,
              SpmmWorkspace& ws) {
  std::size_t buf_size = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      ctx.cusparse.get(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X, &beta, Y, CUDA_R_64F,
      CUSPARSE_SPMM_CSR_ALG2, &buf_size));
  ws.reserve(buf_size + 1);
  CUSPARSE_CHECK(cusparseSpMM(ctx.cusparse.get(),
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X,
                              &beta, Y, CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2,
                              ws.buf.get()));
}

// Set VARPRO_DENSE_SCHUR_DEBUG=1 for a phase breakdown of the formation.
bool debugEnabled() {
  const char* e = std::getenv("VARPRO_DENSE_SCHUR_DEBUG");
  return e && e[0] != '0';
}

using Clock = std::chrono::high_resolution_clock;
double since(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now() - t).count();
}

}  // namespace

void formDenseSchurOnDevice(const VarPro::SparseMatrix& Qmain,
                            const VarPro::SparseMatrix& B,
                            GpuContext& ctx,
                            GpuSparseSolver& Msolver,
                            DeviceBuffer<double>& Qsc,
                            int col_block) {
  const int p = static_cast<int>(Qmain.rows());
  const int m = Msolver.rows();

  if (Qmain.cols() != p)
    throw std::runtime_error("formDenseSchurOnDevice: Qmain must be square");
  if (B.rows() != p || B.cols() != m)
    throw std::runtime_error("formDenseSchurOnDevice: B must be p x m");

  const std::size_t pp = static_cast<std::size_t>(p) * p;
  Qsc.reallocate(pp);

  SpmmWorkspace ws;
  const bool debug = debugEnabled();
  double t_qmain = 0.0, t_factor = 0.0, t_s2d = 0.0, t_solve = 0.0,
         t_spmm = 0.0;
  auto t_phase = Clock::now();

  // ---- Q_sc <- dense(Qmain) --------------------------------------------
  // cusparseSparseToDense writes every entry of the destination, but we zero
  // first so that a future partially-structured Qmain cannot leave garbage.
  Qsc.zero();
  {
    GpuCsrMatrix Qmain_dev;
    uploadEigenSparse(Qmain_dev, Qmain);
    DnMatView Qsc_view(p, p, p, Qsc.get(), CUSPARSE_ORDER_COL);
    sparseToDense(ctx, Qmain_dev.descr, Qsc_view.d, ws);
    ctx.synchronize();  // Qmain_dev dies here; the conversion must be done
  }
  t_qmain = since(t_phase);

  // ---- B on device, plus the host row offsets used to slice it ----------
  GpuCsrMatrix B_dev;
  uploadEigenSparse(B_dev, B);

  VarPro::SparseMatrix Bc = B;
  Bc.makeCompressed();
  const auto* outer = Bc.outerIndexPtr();  // row offsets: VarPro::SparseMatrix
                                           // is RowMajor, so a block of rows is
                                           // a contiguous CSR slice.

  // ---- M^{-1} is supplied by the caller (already factorized) -------------
  t_factor = 0.0;

  const int nc_max = std::min(col_block > 0 ? col_block : 1024, p);
  GpuDenseMatrix rhs(m, nc_max);
  GpuDenseMatrix sol(m, nc_max);
  DeviceBuffer<int> row_ptr_dev;
  std::vector<int> row_ptr_host(nc_max + 1);

  for (int c0 = 0; c0 < p; c0 += nc_max) {
    const int nc = std::min(nc_max, p - c0);
    const int off = static_cast<int>(outer[c0]);
    const int nnz_blk = static_cast<int>(outer[c0 + nc]) - off;

    if (debug) t_phase = Clock::now();
    if (rhs.cols != nc) rhs.resize(m, nc);
    rhs.zero();

    if (nnz_blk > 0) {
      // Rows [c0, c0+nc) of B, rebased so the slice is a valid CSR.
      for (int i = 0; i <= nc; ++i)
        row_ptr_host[i] = static_cast<int>(outer[c0 + i]) - off;
      row_ptr_dev.upload(row_ptr_host.data(), nc + 1);

      SpMatView B_slice(nc, m, nnz_blk, row_ptr_dev.get(),
                        B_dev.col_indices.get() + off,
                        B_dev.values.get() + off);

      // rhs = B(c0:c0+nc, :)^T, i.e. an m x nc column-major block. A row-major
      // nc x m dense matrix with ld = m is bit-identical to that, so the
      // transpose is free: we just describe the same memory the other way up.
      DnMatView rhs_view(nc, m, m, rhs.data.get(), CUSPARSE_ORDER_ROW);
      sparseToDense(ctx, B_slice.d, rhs_view.d, ws);
    }
    if (debug) { ctx.synchronize(); t_s2d += since(t_phase); t_phase = Clock::now(); }

    // sol = M^{-1} rhs
    Msolver.solve(rhs, sol);
    if (debug) { ctx.synchronize(); t_solve += since(t_phase); t_phase = Clock::now(); }

    // Q_sc[:, c0:c0+nc] -= B * sol, accumulated straight into the result.
    DnMatView out_view(p, nc, p, Qsc.get() + static_cast<std::size_t>(c0) * p,
                       CUSPARSE_ORDER_COL);
    spmmView(ctx, B_dev.descr, sol.descr, out_view.d, -1.0, 1.0, ws);
    if (debug) { ctx.synchronize(); t_spmm += since(t_phase); }

    // cuDSS and cuSPARSE share the stream, but row_ptr_dev is reused (and
    // reallocated on the short final block) from the host next iteration.
    ctx.synchronize();
  }

  if (debug) {
    std::cout << "[dense-schur] p=" << p << " m=" << m << " blocks="
              << (p + nc_max - 1) / nc_max << " (nc=" << nc_max << ")\n"
              << "[dense-schur]   dense(Qmain)   " << t_qmain << " s\n"
              << "[dense-schur]   factor(M)      " << t_factor << " s\n"
              << "[dense-schur]   sparse->dense  " << t_s2d << " s\n"
              << "[dense-schur]   M^-1 solves    " << t_solve << " s\n"
              << "[dense-schur]   SpMM update    " << t_spmm << " s"
              << std::endl;
  }
}

// Convenience overload: build a solver for M, then delegate. Callers that
// re-form Q_sc repeatedly (IRLS) should own the solver and use the other
// overload so the cuDSS symbolic factorization is computed once.
void formDenseSchurOnDevice(const VarPro::SparseMatrix& Qmain,
                            const VarPro::SparseMatrix& B,
                            const VarPro::SparseMatrix& M,
                            GpuContext& ctx,
                            DeviceBuffer<double>& Qsc,
                            int col_block) {
  if (M.rows() != M.cols())
    throw std::runtime_error("formDenseSchurOnDevice: M must be square");
  if (!gpuSparseSolverAvailable())
    throw std::runtime_error(
        "formDenseSchurOnDevice: needs cuDSS for M^{-1}; rebuild with cuDSS "
        "or form the Schur complement on the host.");
  GpuSparseSolver Msolver(M, ctx);
  formDenseSchurOnDevice(Qmain, B, ctx, Msolver, Qsc, col_block);
}

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
