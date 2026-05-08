/**
 * @file GpuLinearAlgebra.h
 * @brief cuBLAS / cuSPARSE wrappers for the VarPro GPU solver.
 *
 * All operations work on double-precision (FP64) column-major matrices.
 *
 * Sparse matrices are stored in CSR format on the device.
 * Dense matrices are stored column-major to match Eigen and cuBLAS conventions.
 *
 * Usage:
 *   GpuContext ctx;            // creates handles + stream
 *   GpuDenseMatrix Y(p, r);   // allocates p×r device matrix
 *   GpuCsrMatrix A;
 *   uploadEigenSparse(A, ctx, eigen_sparse_matrix);
 *   spmmCSR(ctx, A, Y, result);  // result = A * Y
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/CudaErrorCheck.h>
#include <VarProGPU/DeviceBuffer.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstddef>
#include <memory>
#include <vector>

namespace VarProGPU {

// ---------------------------------------------------------------------------
// GpuContext — cuBLAS + cuSPARSE handles sharing a single stream
// ---------------------------------------------------------------------------

struct GpuContext {
  CudaStream     stream;
  CuBlasHandle   cublas;
  CuSparseHandle cusparse;

  GpuContext() {
    cublas.setStream(stream.get());
    cusparse.setStream(stream.get());
  }

  void synchronize() { stream.synchronize(); }
};

// ---------------------------------------------------------------------------
// GpuDenseMatrix — column-major p×r device matrix
// ---------------------------------------------------------------------------

struct GpuDenseMatrix {
  int rows{0}, cols{0};
  DeviceBuffer<double> data;
  cusparseDnMatDescr_t descr{nullptr};

  GpuDenseMatrix() = default;

  GpuDenseMatrix(int r, int c) { resize(r, c); }

  ~GpuDenseMatrix() {
    if (descr) cusparseDestroyDnMat(descr);
  }

  GpuDenseMatrix(const GpuDenseMatrix&) = delete;
  GpuDenseMatrix& operator=(const GpuDenseMatrix&) = delete;

  GpuDenseMatrix(GpuDenseMatrix&& o) noexcept
      : rows(o.rows), cols(o.cols), data(std::move(o.data)), descr(o.descr) {
    o.rows = 0; o.cols = 0; o.descr = nullptr;
  }
  GpuDenseMatrix& operator=(GpuDenseMatrix&& o) noexcept {
    if (this != &o) {
      if (descr) { cusparseDestroyDnMat(descr); descr = nullptr; }
      rows = o.rows; cols = o.cols;
      data = std::move(o.data);
      descr = o.descr;
      o.rows = 0; o.cols = 0; o.descr = nullptr;
    }
    return *this;
  }

  void resize(int r, int c) {
    if (descr) { cusparseDestroyDnMat(descr); descr = nullptr; }
    rows = r; cols = c;
    data.allocate(static_cast<std::size_t>(r) * c);
    // column-major: leading dimension = rows
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &descr, r, c, /*ld=*/r, data.get(), CUDA_R_64F, CUSPARSE_ORDER_COL));
  }

  void zero() { data.zero(); }

  // Upload from Eigen column-major matrix (synchronous, default stream)
  void upload(const Eigen::MatrixXd& M) {
    if (M.rows() != rows || M.cols() != cols) resize(M.rows(), M.cols());
    data.upload(M.data(), static_cast<std::size_t>(rows) * cols);
  }

  // Upload on a specific stream so cuBLAS/cuSPARSE ops on that stream see the data
  void uploadAsync(const Eigen::MatrixXd& M, cudaStream_t stream) {
    if (M.rows() != rows || M.cols() != cols) resize(M.rows(), M.cols());
    data.uploadAsync(M.data(), static_cast<std::size_t>(rows) * cols, stream);
  }

  // Download to Eigen matrix
  Eigen::MatrixXd download() const {
    Eigen::MatrixXd M(rows, cols);
    data.download(M.data());
    return M;
  }
};

// ---------------------------------------------------------------------------
// GpuCsrMatrix — CSR sparse matrix on device
// ---------------------------------------------------------------------------

struct GpuCsrMatrix {
  int rows{0}, cols{0}, nnz{0};
  DeviceBuffer<int>    row_offsets;   // length rows+1
  DeviceBuffer<int>    col_indices;   // length nnz
  DeviceBuffer<double> values;        // length nnz
  cusparseSpMatDescr_t descr{nullptr};

  GpuCsrMatrix() = default;

  ~GpuCsrMatrix() {
    if (descr) cusparseDestroySpMat(descr);
  }

  GpuCsrMatrix(const GpuCsrMatrix&) = delete;
  GpuCsrMatrix& operator=(const GpuCsrMatrix&) = delete;

  GpuCsrMatrix(GpuCsrMatrix&& o) noexcept
      : rows(o.rows), cols(o.cols), nnz(o.nnz),
        row_offsets(std::move(o.row_offsets)),
        col_indices(std::move(o.col_indices)),
        values(std::move(o.values)),
        descr(o.descr) {
    o.rows = 0; o.cols = 0; o.nnz = 0; o.descr = nullptr;
  }
  GpuCsrMatrix& operator=(GpuCsrMatrix&& o) noexcept {
    if (this != &o) {
      if (descr) { cusparseDestroySpMat(descr); descr = nullptr; }
      rows = o.rows; cols = o.cols; nnz = o.nnz;
      row_offsets = std::move(o.row_offsets);
      col_indices = std::move(o.col_indices);
      values = std::move(o.values);
      descr = o.descr;
      o.rows = 0; o.cols = 0; o.nnz = 0; o.descr = nullptr;
    }
    return *this;
  }

  bool empty() const { return rows == 0; }
};

// ---------------------------------------------------------------------------
// uploadEigenSparse — transfer a row-major Eigen sparse matrix to device CSR
// ---------------------------------------------------------------------------

// Eigen's RowMajor SparseMatrix already stores data in CSR format.
// This function uploads the three CSR arrays directly.
inline void uploadEigenSparse(
    GpuCsrMatrix& gpu,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A) {
  if (gpu.descr) { cusparseDestroySpMat(gpu.descr); gpu.descr = nullptr; }

  gpu.rows = static_cast<int>(A.rows());
  gpu.cols = static_cast<int>(A.cols());
  gpu.nnz  = static_cast<int>(A.nonZeros());

  // Eigen RowMajor CSR: outerIndexPtr gives row offsets (length rows+1)
  gpu.row_offsets.upload(A.outerIndexPtr(), gpu.rows + 1);
  gpu.col_indices.upload(A.innerIndexPtr(), gpu.nnz);
  gpu.values.upload(A.valuePtr(), gpu.nnz);

  CUSPARSE_CHECK(cusparseCreateCsr(
      &gpu.descr,
      gpu.rows, gpu.cols, gpu.nnz,
      gpu.row_offsets.get(), gpu.col_indices.get(), gpu.values.get(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
}

// Overload for Eigen ColMajor — convert to RowMajor first
inline void uploadEigenSparse(
    GpuCsrMatrix& gpu,
    const Eigen::SparseMatrix<double>& A) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> Arm = A;
  uploadEigenSparse(gpu, Arm);
}

// ---------------------------------------------------------------------------
// spmmCSR — Y = alpha * A * X + beta * Y  (sparse A, dense X and Y)
// ---------------------------------------------------------------------------

void spmmCSR(GpuContext& ctx,
             const GpuCsrMatrix& A,
             const GpuDenseMatrix& X,
             GpuDenseMatrix& Y,
             double alpha = 1.0,
             double beta  = 0.0);

// ---------------------------------------------------------------------------
// spmmCSRT — Y = alpha * A^T * X + beta * Y
// ---------------------------------------------------------------------------

void spmmCSRT(GpuContext& ctx,
              const GpuCsrMatrix& A,
              const GpuDenseMatrix& X,
              GpuDenseMatrix& Y,
              double alpha = 1.0,
              double beta  = 0.0);

// ---------------------------------------------------------------------------
// daxpy — Y = Y + alpha * X   (element-wise on all p*r elements)
// ---------------------------------------------------------------------------

void daxpy(GpuContext& ctx, double alpha,
           const GpuDenseMatrix& X, GpuDenseMatrix& Y);

// ---------------------------------------------------------------------------
// dscal — X *= alpha
// ---------------------------------------------------------------------------

void dscal(GpuContext& ctx, double alpha, GpuDenseMatrix& X);

// ---------------------------------------------------------------------------
// ddot — returns tr(X^T Y) = vec(X) . vec(Y)
// (i.e., the Frobenius inner product of two p×r matrices)
// ---------------------------------------------------------------------------

double ddot(GpuContext& ctx,
            const GpuDenseMatrix& X,
            const GpuDenseMatrix& Y);

// ---------------------------------------------------------------------------
// dnrm2 — returns ‖X‖_F = sqrt(tr(X^T X))
// ---------------------------------------------------------------------------

double dnrm2(GpuContext& ctx, const GpuDenseMatrix& X);

// ---------------------------------------------------------------------------
// dcopy — Y ← X
// ---------------------------------------------------------------------------

void dcopy(GpuContext& ctx,
           const GpuDenseMatrix& X,
           GpuDenseMatrix& Y);

// ---------------------------------------------------------------------------
// ddiagmm — Z = diag(d) * X   (row scaling of a p×r matrix by a p-vector d)
//
// d must have at least X.rows elements on device.
// If Z is not sized correctly it will be resized.
// ---------------------------------------------------------------------------

void ddiagmm(GpuContext& ctx,
             const DeviceBuffer<double>& d,
             const GpuDenseMatrix& X,
             GpuDenseMatrix& Z);

// Overload: raw pointers with explicit dimensions and leading dimension.
// Z[i + j*lda] = d[i] * X[i + j*lda] for i=0..rows-1, j=0..cols-1.
void ddiagmm(GpuContext& ctx,
             const DeviceBuffer<double>& d,
             const double* X, double* Z,
             int rows, int cols, int lda);

// ---------------------------------------------------------------------------
// Scratch-buffer manager for SpMM
// Each cuSPARSE SpMM call may need an external workspace; we manage this
// with a reusable device buffer that grows as needed.
// ---------------------------------------------------------------------------

struct SpmmWorkspace {
  DeviceBuffer<char> buf;

  // Reserve at least `bytes` bytes
  void reserve(std::size_t bytes) {
    if (buf.size() < bytes) buf.allocate(bytes);
  }
};

// Internal — used by spmmCSR/spmmCSRT implementations
void spmmImpl(GpuContext& ctx,
              cusparseOperation_t opA,
              const GpuCsrMatrix& A,
              const GpuDenseMatrix& X,
              GpuDenseMatrix& Y,
              double alpha,
              double beta,
              SpmmWorkspace& ws);

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
