# GPU Variable-Projection Solver — Design Document

## 1. Overview

This document describes the architecture, CPU/GPU split, data structures, kernels, and
numerical assumptions of the GPU-accelerated solver for the experiments in the paper
**"Sparse Variable Projection in Robotic Perception: Exploiting Separable Structure
for Efficient Nonlinear Optimization"**.

### Problem statement (paper notation)

The paper reduces each perception problem to a structured least-squares problem

```
min_{X_c ∈ M_c}  f(X_c) = 0.5 · tr(X_c^T  Q_sc  X_c)
```

where `Q_sc` is the *Schur complement* of the translation/unconstrained block:

```
Q_sc = Q_c − B · (C^T Ω C)^{-1} · B^T
```

with

| Symbol        | Meaning                                                                       |
|---------------|-------------------------------------------------------------------------------|
| `Q_c`         | Upper-left block of full data matrix (rotations + ranges)                     |
| `A_f`         | Incidence-like unconstrained-variable block (translations/landmarks)          |
| `C`           | Reduced incidence basis — `A_f` with one column removed (when valid)          |
| `Ω`           | Diagonal precision matrix for the unconstrained measurements                  |
| `B`           | Off-diagonal coupling block: `B^T = A_c^T Ω C`                               |
| `L L^T`       | Sparse Cholesky of `M = C^T Ω C` (one-time preprocessing)                    |

### Connection to existing code

The existing `Problem` class (src/Problem.cpp) already implements this structure under
`Formulation::Implicit`.  Mapping between paper symbols and code variables:

| Paper         | Code                                                  |
|---------------|-------------------------------------------------------|
| `Q_c`         | `Problem::Qmain_`                                     |
| `B^T` (reduced) | `Problem::TransOffDiagRed_` (= off-diag block, last column dropped) |
| `L`           | `Problem::LtransCholRed_` (SuiteSparse Cholesky of reduced Q33) |
| `X_c`         | Variable matrix `Y` (p×r, p = dim·n_poses + n_ranges) |

The matrix-free product is (Problem.cpp line 908–918):

```
Qsc · Y = Qmain · Y  −  TransOffDiagRed · L^{-1}(L^{-T}(TransOffDiagRed^T · Y))
```

**This IS the paper's Schur operator**; the GPU solver accelerates exactly this hot path.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VarPro Problem                            │
│  (graph data, measurements, manifold metadata)                   │
│  Language: C++17, Eigen sparse                                   │
└────────────┬────────────────────────────────────────────────────┘
             │  one-time preprocessing
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VarProPrecomputeResult                          │
│  Qmain (CSR), TransOffDiagRed (CSR), LtransCholFactor           │
│  Language: C++17, SuiteSparse                                    │
└────────────┬────────────────────────────────────────────────────┘
             │  upload sparse matrices once
             ▼
┌─────────────────────────────────────────────────────────────────┐
│               MatrixFreeSchurOperator                            │
│  ┌─────────────────────┐   ┌───────────────────────────────┐   │
│  │  CpuSchurOperator   │   │   GpuSchurOperator            │   │
│  │  (reference / debug)│   │   SpMM via cuSPARSE           │   │
│  │  Eigen sparse-dense │   │   trisolve on CPU (v1)        │   │
│  └─────────────────────┘   │   vector ops via cuBLAS       │   │
│                             └───────────────────────────────┘   │
└────────────┬────────────────────────────────────────────────────┘
             │  Hessian-vector products inside pTCG
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RTRSolver  (Riemannian Trust-Region)             │
│  ┌─────────────────────┐   ┌───────────────────────────────┐   │
│  │   CpuRTRSolver      │   │   GpuRTRSolver                │   │
│  │   wraps TNT library │   │   iterate Y kept on device    │   │
│  │   (existing code)   │   │   pTCG ops via cuBLAS         │   │
│  └─────────────────────┘   └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. CPU / GPU Split

### Phase 1 (one-time preprocessing — always CPU)

| Operation                          | Rationale                                              |
|------------------------------------|--------------------------------------------------------|
| Parse measurements, build graph    | I/O-bound, sequential, tiny fraction of total time     |
| Construct sparse data matrix Q     | Eigen triplet assembly — no GPU benefit                |
| Symbolic sparse Cholesky           | SuiteSparse AMD ordering — no GPU equivalent available |
| Numeric Cholesky L L^T of Q33      | Done once; SuiteSparse is faster than cuSPARSE for 1× |
| Build reduced incidence C          | Graph operation on CPU                                 |
| Build Qmain, TransOffDiagRed       | Extract blocks from Q — Eigen block extraction         |

### Phase 2 (hot path per RTR/pTCG iteration — GPU-accelerated)

| Operation                                     | Implementation          |
|-----------------------------------------------|-------------------------|
| `Qmain * Y`                                   | cuSPARSE SpMM           |
| `TransOffDiagRed^T * Y`                       | cuSPARSE SpMM           |
| `L^{-1}(L^{-T} P)`  (triangular solve)        | CPU (v1); cuSPARSE (v2) |
| `TransOffDiagRed * P`                         | cuSPARSE SpMM           |
| `QY − TransOffDiagRed*P`                      | cuBLAS Daxpy            |
| Tangent projection (Stiefel/Oblique blocks)   | Custom CUDA kernels     |
| Retraction (manifold projection)              | Custom CUDA kernels     |
| Inner product, norm, axpy in pTCG             | cuBLAS Ddot/Dnrm2/Daxpy |

The triangular solve is kept on CPU in v1 because:
- It involves a lower-triangular solve of an m×m system (m = n_translations − 1)
- The matrix m×r right-hand side (r ≤ 10) is small enough that D2H + solve + H2D
  is faster than the cuSPARSE setup overhead for medium problems
- The code is structured so swapping in `cusparseSpSV` requires touching only one
  method (`GpuSchurOperator::triSolve`), with no other changes

---

## 4. Data Structures

### 4.1 GpuCsrMatrix

```cpp
struct GpuCsrMatrix {
    int   rows, cols, nnz;
    int*  row_offsets;   // device, length rows+1
    int*  col_indices;   // device, length nnz
    double* values;      // device, length nnz
    cusparseSpMatDescr_t descr;  // cuSPARSE generic API descriptor
};
```

Populated once from Eigen CSR by `uploadEigenSparse()`.

### 4.2 DeviceMatrix

```cpp
struct DeviceMatrix {
    int rows, cols;
    double* data;        // device, column-major, rows*cols doubles
    cusparseDnMatDescr_t descr;  // cuSPARSE dense descriptor
};
```

Column-major to match Eigen's default and cuBLAS conventions.

### 4.3 VarProPrecomputeResult

```cpp
struct VarProPrecomputeResult {
    // CPU side
    SparseMatrix     Qmain;              // rotations+ranges block
    SparseMatrix     TransOffDiagRed;    // off-diagonal coupling (reduced)
    CholFactorPtr    LtransCholRed;      // Cholesky of reduced translation block
    // Problem metadata
    int p;   // variable matrix rows (dim*n_poses + n_ranges)
    int m;   // triangular solve dimension (n_translations - 1)
    int r;   // relaxation rank
};
```

### 4.4 GpuSchurData

```cpp
struct GpuSchurData {
    GpuCsrMatrix  Qmain_dev;
    GpuCsrMatrix  TransOffDiag_dev;
    GpuCsrMatrix  TransOffDiagT_dev;   // explicit transpose for SpMM efficiency
    // Scratch buffers (pre-allocated, reused every call)
    DeviceMatrix  P1_dev;   // m × r
    DeviceMatrix  P2_dev;   // m × r
    DeviceMatrix  QY_dev;   // p × r
    DeviceMatrix  P3_dev;   // p × r
    // Handles
    cusparseHandle_t  cusparse_handle;
    cublasHandle_t    cublas_handle;
    cudaStream_t      stream;
};
```

---

## 5. CUDA Kernels

### 5.1 Tangent Space Projection (`manifold_kernels.cu`)

**Stiefel projection** `Pi_{T_Y St} (V)`:
For each k×r block `Y_i` (rotation variable i):
```
V_i  ←  V_i − Y_i · sym(Y_i^T · V_i)
       = V_i − Y_i · ((Y_i^T · V_i + V_i^T · Y_i) / 2)
```
Implemented as a batched cuBLAS GEMM (Y_i^T · V_i) followed by a symmetrize kernel,
then another batched GEMM (Y_i · sym).

**Oblique projection** `Pi_{T_Y Ob} (V)`:
For each unit vector `y_i` (row of range block):
```
v_i  ←  v_i − (y_i^T · v_i) · y_i
```
Implemented as a custom kernel: one thread per range-variable, dot product + axpy.

**Euclidean projection**: identity (no-op).

### 5.2 Manifold Retraction (`manifold_kernels.cu`)

**Stiefel retraction** (polar): `R(Y_i, V_i) = polar(Y_i + V_i)`
Computed via batched SVD using cuSOLVER `cusolverDnDgesvd` per block.

**Oblique retraction**: `R(y_i, v_i) = (y_i + v_i) / ‖y_i + v_i‖`
Custom row-normalization kernel.

### 5.3 SymBlockDiagProduct (`manifold_kernels.cu`)

Used in the Riemannian Hessian computation (Problem.cpp line 1087):
```
sym(Y_i^T · G_i) · V_i   for each rotation block i
```
Batched cuBLAS SYRK + GEMM.

---

## 6. Numerical Assumptions

1. **Linear residuals**: The Schur complement preprocessing is one-time because all
   experiment residuals are linear in the translation/unconstrained variables.  If a
   nonlinear residual were added (e.g., a GPS factor), the preprocessing must be
   repeated each outer iteration.

2. **Incidence-like A_f**: The fast path drops one column of A_f to form the reduced
   basis C.  This is valid when A_f is the incidence matrix of a connected graph.
   Connectedness is checked before entering the fast path; otherwise a general
   null-space basis is computed via QR (slow path).

3. **Gauge fixing**: One translation is pinned (last index removed from Q33) to fix
   the gauge before Cholesky.  This is the "reduced" in `TransOffDiagRed_` and
   `LtransCholRed_`.  The pinned index corresponds to the origin pose.

4. **Positive definiteness of M**: The Cholesky factorization assumes M = C^T Ω C
   is positive definite.  This holds when:
   - Ω is positive diagonal (all precisions > 0)
   - C has full column rank (graph is connected)
   - At least one translation is pinned (gauge fixed)
   A Cholesky failure triggers a fallback that adds a small diagonal regularizer.

5. **Relaxation rank**: The solver uses a low-rank lifting Y ∈ St(k,r)^n with r ≥ k.
   A certificate of global optimality can be computed when r > k (SE-Sync style).

6. **Double precision**: All GPU computations use `double` (FP64).  The RTX 5090 has
   reduced FP64 throughput vs FP32; a future optimization could use mixed precision
   for the SpMM with FP32 accumulation and FP64 corrections.

---

## 7. Paper Faithfulness vs. Implementation Choices

### Faithful to paper

- Schur complement operator: exact formula (Section 3.2 of paper)
- Reduced incidence basis: drop last column of incidence matrix (Section 3.3)
- Gauge fixing by removing one translational variable (Section 3.3)
- Riemannian optimization on product manifold (Stiefel × Oblique × Euclidean)
- pTCG as inner solver for the trust-region subproblem
- One-time preprocessing (Cholesky + B matrix) for linear residuals

### Implementation choices (deviations documented)

| Choice                              | Reason                                         |
|-------------------------------------|------------------------------------------------|
| CPU triangular solve (v1)           | Simpler; cuSPARSE SpSV available for v2        |
| Explicit transpose `TransOffDiagRed^T` stored | Avoids runtime transpose in SpMM, trades memory for speed |
| Column-major dense matrices on GPU | Matches Eigen default + cuBLAS convention      |
| Single CUDA stream (v1)             | Simpler; can pipeline H2D/SpMM with 2 streams  |
| FP64 throughout                     | Correctness first; FP32 SpMM is future work    |
| TNT library for CPU RTR             | Reuses validated code; GpuRTRSolver for GPU path |
| cuBLAS inner products in pTCG       | Lower overhead than device-to-host reduction   |

---

## 8. Experiment Families

| Family   | Manifold           | Has range meas. | Has translations | Typical rank |
|----------|--------------------|-----------------|------------------|--------------|
| PGO      | Stiefel^n          | No              | Yes              | d+1 to d+3  |
| RA-SLAM  | Stiefel^n × Ob^r   | Yes             | Yes              | d+1 to d+5  |
| SfM      | ScaledStiefel^n    | No              | Yes (landmarks)  | d+1 to d+5  |
| SNL      | Euclidean^n        | Yes             | Yes (= poses)    | d+1 to d+3  |

All use the same Schur operator hot path; only the manifold projection differs.

---

## 9. Reproducing Paper Benchmarks

### Prerequisites
```bash
sudo apt install libsuitesparse-dev libeigen3-dev libopenblas-dev
# CUDA toolkit 11.x or later
```

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
make -j$(nproc)
```

### Run benchmarks
```bash
./bin/benchmark_solver --dataset examples/data/pgo/grid3D
./bin/benchmark_schur  --dataset examples/data/pgo/city10000 --reps 1000
```

### Run tests
```bash
./bin/test_schur_vs_explicit   # matrix-free matches explicit on small problems
./bin/test_fd_gradient         # gradient matches finite-difference approximation
./bin/test_convergence         # solver converges on PGO, SNL, SfM, RA-SLAM
```

---

## 10. Known Limitations and Future Work

- **cuSPARSE triangular solve (v2)**: Replace CPU Cholesky back-solve with
  `cusparseSpSV`; beneficial when m > 5000.
- **Parallel streams**: Issue SpMM for `Qmain*Y` and `TransOffDiag^T*Y` concurrently
  on two streams, then synchronize before the subtraction.
- **FP32 SpMM + FP64 residual**: Use `CUSPARSE_SPMM_COO_ALG4` in mixed precision.
- **Batched SVD for retraction**: `cusolverDnDgesvdBatched` once available and stable.
- **Distributed multi-GPU**: Each subgraph on one GPU; reduce over boundaries.
- **Python bindings**: Expose `GpuVarProSolver` via pybind11.
