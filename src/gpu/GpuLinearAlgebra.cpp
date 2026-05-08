/**
 * @file GpuLinearAlgebra.cpp
 * @brief cuBLAS / cuSPARSE wrapper implementations compiled by g++.
 *
 * cuBLAS and cuSPARSE are C APIs that don't require nvcc.  Compiling them
 * with g++ avoids the C++17 template limitations of nvcc 11.x and allows
 * full use of Eigen types in the same translation unit.
 */

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuLinearAlgebra.h>

#include <stdexcept>

namespace VarProGPU {

// ---------------------------------------------------------------------------
// Thread-local SpMM workspace (grows as needed, never shrinks)
// ---------------------------------------------------------------------------

static thread_local SpmmWorkspace g_spmm_ws;

// ---------------------------------------------------------------------------
// Internal SpMM dispatch
// ---------------------------------------------------------------------------

void spmmImpl(GpuContext& ctx,
              cusparseOperation_t opA,
              const GpuCsrMatrix& A,
              const GpuDenseMatrix& X,
              GpuDenseMatrix& Y,
              double alpha,
              double beta,
              SpmmWorkspace& ws) {

  int out_rows = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;
  int out_cols = X.cols;

  if (Y.rows != out_rows || Y.cols != out_cols) {
    Y.resize(out_rows, out_cols);
  }

  std::size_t buf_size = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      ctx.cusparse.get(),
      opA,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr,
      const_cast<cusparseDnMatDescr_t>(X.descr),
      &beta,
      Y.descr,
      CUDA_R_64F,
      CUSPARSE_SPMM_CSR_ALG2,
      &buf_size));

  ws.reserve(buf_size + 1);

  CUSPARSE_CHECK(cusparseSpMM(
      ctx.cusparse.get(),
      opA,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr,
      const_cast<cusparseDnMatDescr_t>(X.descr),
      &beta,
      Y.descr,
      CUDA_R_64F,
      CUSPARSE_SPMM_CSR_ALG2,
      ws.buf.get()));
}

// ---------------------------------------------------------------------------
// Public SpMM functions
// ---------------------------------------------------------------------------

void spmmCSR(GpuContext& ctx,
             const GpuCsrMatrix& A,
             const GpuDenseMatrix& X,
             GpuDenseMatrix& Y,
             double alpha, double beta) {
  spmmImpl(ctx, CUSPARSE_OPERATION_NON_TRANSPOSE, A, X, Y, alpha, beta,
           g_spmm_ws);
}

void spmmCSRT(GpuContext& ctx,
              const GpuCsrMatrix& A,
              const GpuDenseMatrix& X,
              GpuDenseMatrix& Y,
              double alpha, double beta) {
  spmmImpl(ctx, CUSPARSE_OPERATION_TRANSPOSE, A, X, Y, alpha, beta,
           g_spmm_ws);
}

// ---------------------------------------------------------------------------
// cuBLAS vector operations
// ---------------------------------------------------------------------------

void daxpy(GpuContext& ctx, double alpha,
           const GpuDenseMatrix& X, GpuDenseMatrix& Y) {
  int n = X.rows * X.cols;
  CUBLAS_CHECK(cublasDaxpy(ctx.cublas.get(), n,
                           &alpha, X.data.get(), 1, Y.data.get(), 1));
}

void dscal(GpuContext& ctx, double alpha, GpuDenseMatrix& X) {
  int n = X.rows * X.cols;
  CUBLAS_CHECK(cublasDscal(ctx.cublas.get(), n, &alpha, X.data.get(), 1));
}

double ddot(GpuContext& ctx,
            const GpuDenseMatrix& X,
            const GpuDenseMatrix& Y) {
  int n = X.rows * X.cols;
  double result = 0.0;
  CUBLAS_CHECK(cublasDdot(ctx.cublas.get(), n,
                          X.data.get(), 1, Y.data.get(), 1, &result));
  return result;
}

double dnrm2(GpuContext& ctx, const GpuDenseMatrix& X) {
  int n = X.rows * X.cols;
  double result = 0.0;
  CUBLAS_CHECK(cublasDnrm2(ctx.cublas.get(), n, X.data.get(), 1, &result));
  return result;
}

void dcopy(GpuContext& ctx,
           const GpuDenseMatrix& X,
           GpuDenseMatrix& Y) {
  if (Y.rows != X.rows || Y.cols != X.cols) Y.resize(X.rows, X.cols);
  int n = X.rows * X.cols;
  CUBLAS_CHECK(cublasDcopy(ctx.cublas.get(), n,
                           X.data.get(), 1, Y.data.get(), 1));
}

void ddiagmm(GpuContext& ctx,
             const DeviceBuffer<double>& d,
             const GpuDenseMatrix& X,
             GpuDenseMatrix& Z) {
  if (Z.rows != X.rows || Z.cols != X.cols) Z.resize(X.rows, X.cols);
  // cublasDdgmm: C = diag(d) * A  (CUBLAS_SIDE_LEFT)
  // m=rows, n=cols, A=X.data, lda=X.rows, x=d.get(), incx=1, C=Z.data, ldc=Z.rows
  CUBLAS_CHECK(cublasDdgmm(ctx.cublas.get(),
                           CUBLAS_SIDE_LEFT,
                           X.rows, X.cols,
                           X.data.get(), X.rows,
                           d.get(), 1,
                           Z.data.get(), Z.rows));
}

void ddiagmm(GpuContext& ctx,
             const DeviceBuffer<double>& d,
             const double* X, double* Z,
             int rows, int cols, int lda) {
  CUBLAS_CHECK(cublasDdgmm(ctx.cublas.get(),
                           CUBLAS_SIDE_LEFT,
                           rows, cols,
                           X, lda,
                           d.get(), 1,
                           Z, lda));
}

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
