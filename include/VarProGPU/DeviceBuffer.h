/**
 * @file DeviceBuffer.h
 * @brief RAII wrappers for GPU device memory and CUDA handles.
 *
 * All device allocations go through these wrappers so that memory is freed
 * automatically even when exceptions are thrown.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/CudaErrorCheck.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace VarProGPU {

// ---------------------------------------------------------------------------
// DeviceBuffer<T> — typed, owning device allocation
// ---------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  // Non-copyable; movable
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
    o.ptr_ = nullptr;
    o.count_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      free();
      ptr_ = o.ptr_;
      count_ = o.count_;
      o.ptr_ = nullptr;
      o.count_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { free(); }

  void allocate(std::size_t count) {
    free();
    count_ = count;
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
    }
  }

  void reallocate(std::size_t count) {
    if (count != count_) allocate(count);
  }

  void zero() {
    if (ptr_ && count_ > 0)
      CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
  }

  // Upload from host vector
  void upload(const std::vector<T>& host_data) {
    reallocate(host_data.size());
    CUDA_CHECK(cudaMemcpy(ptr_, host_data.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  // Upload from raw host pointer
  void upload(const T* host_ptr, std::size_t count) {
    reallocate(count);
    CUDA_CHECK(
        cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Upload from raw host pointer on a specific stream (async, stream-ordered)
  void uploadAsync(const T* host_ptr, std::size_t count, cudaStream_t stream) {
    reallocate(count);
    CUDA_CHECK(cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  // Download to host vector
  void download(std::vector<T>& host_data) const {
    host_data.resize(count_);
    CUDA_CHECK(cudaMemcpy(host_data.data(), ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  // Download to raw host pointer (caller must have allocated enough space)
  void download(T* host_ptr) const {
    CUDA_CHECK(
        cudaMemcpy(host_ptr, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }

 private:
  void free() {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// CuBlasHandle — RAII cuBLAS handle
// ---------------------------------------------------------------------------

class CuBlasHandle {
 public:
  CuBlasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CuBlasHandle() {
    if (handle_) cublasDestroy(handle_);
  }
  CuBlasHandle(const CuBlasHandle&) = delete;
  CuBlasHandle& operator=(const CuBlasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }
  operator cublasHandle_t() const { return handle_; }

  void setStream(cudaStream_t s) {
    CUBLAS_CHECK(cublasSetStream(handle_, s));
  }

 private:
  cublasHandle_t handle_{};
};

// ---------------------------------------------------------------------------
// CuSparseHandle — RAII cuSPARSE handle
// ---------------------------------------------------------------------------

class CuSparseHandle {
 public:
  CuSparseHandle() { CUSPARSE_CHECK(cusparseCreate(&handle_)); }
  ~CuSparseHandle() {
    if (handle_) cusparseDestroy(handle_);
  }
  CuSparseHandle(const CuSparseHandle&) = delete;
  CuSparseHandle& operator=(const CuSparseHandle&) = delete;

  cusparseHandle_t get() const { return handle_; }
  operator cusparseHandle_t() const { return handle_; }

  void setStream(cudaStream_t s) {
    CUSPARSE_CHECK(cusparseSetStream(handle_, s));
  }

 private:
  cusparseHandle_t handle_{};
};

// ---------------------------------------------------------------------------
// CudaStream — RAII stream
// ---------------------------------------------------------------------------

class CudaStream {
 public:
  CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
  ~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; }

  void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

 private:
  cudaStream_t stream_{};
};

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
