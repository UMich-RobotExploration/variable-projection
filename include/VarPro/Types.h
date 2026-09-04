/**
 * @file types.h
 * @brief A file containing the basic types used by the VARPRO library.
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

#include "Optimization/Riemannian/TNT.h"

class NotImplementedException : public std::logic_error {
public:
  explicit NotImplementedException(std::string const &str)
      : std::logic_error(str + " not implemented") {}
};

using Index = Eigen::Index;

class MatrixShapeException : public std::logic_error {
public:
  MatrixShapeException(const std::string &func_name, Index exp_rows,
                       Index exp_cols, Index act_rows, Index act_cols)
      : std::logic_error(func_name + ": " + "expected matrix of shape (" +
                         std::to_string(exp_rows) + ", " +
                         std::to_string(exp_cols) + ") but got (" +
                         std::to_string(act_rows) + ", " +
                         std::to_string(act_cols) + ")") {}
};
inline void checkMatrixShape(const std::string &func_name, Index exp_rows,
                             Index exp_cols, Index act_rows, Index act_cols) {
  if (exp_rows != act_rows || exp_cols != act_cols) {
    throw MatrixShapeException(func_name, exp_rows, exp_cols, act_rows,
                               act_cols);
  }
}

namespace VarPro {

typedef double Scalar;

typedef Eigen::VectorXi VectorXi;
typedef Eigen::Index Index;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

enum class Formulation {
  // The original problem in which translations are explicitly represented
  Explicit,
  // The original problem, but where the translations are set to their optimal
  // values at each iterate
  ExplicitVarPro,
  // The problem in which translations are marginalized out, with the Schur
  // complement applied matrix-free
  Implicit,
  // The problem in which translations are marginalized out, with the Schur
  // complement Q_sc = Q_c - B M^{-1} B^T formed *explicitly* as a dense
  // matrix (the classical "reduced camera system" of dense bundle
  // adjustment). Identical to Implicit in every respect -- same variables,
  // same manifold, same preconditioner, same translation recovery -- except
  // that the operator application is a dense GEMM instead of the matrix-free
  // sparse triple product. Exists as a baseline: it is quadratic in memory
  // where Implicit is linear.
  Dense
};

/** True for the formulations that marginalize the translations out, i.e. whose
 * decision variable is the reduced (rotation + range) block rather than the
 * full variable. Implicit and Dense differ *only* in how the Schur complement
 * operator is applied, so every site that branches on "is this the reduced
 * problem?" must use this predicate rather than comparing to
 * Formulation::Implicit. */
inline bool isMarginalized(Formulation f) {
  return f == Formulation::Implicit || f == Formulation::Dense;
}

struct CertResults {
  bool is_certified;
  Scalar theta;
  Vector x;
  Matrix all_eigvecs;
  size_t num_iters;
};

/** Per SE-Sync:
 * We use row-major storage order to take advantage of fast (sparse-matrix) *
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> SparseMatrix;

// manifold operations
enum class StiefelRetraction { QR, Polar };
enum class ObliqueRetraction { Normalize };

/** The preconditioner applied to the inner tCG solver. */
enum class Preconditioner { None, Jacobi, BlockCholesky, RegularizedCholesky };

/** The initialization method used for the VARPRO algorithm. */
enum class Initialization { Random, Odometry };

/** A typedef for an "instrumentation function" that can be passed into
 * the Riemannian TNT solver. */
typedef Optimization::Riemannian::TNTUserFunction<Matrix, Matrix, Scalar,
                                                  Matrix>
    InstrumentationFunction;

/** A typedef for a separable structure update (based on Khosoussi et al.)
 * that can be passed into the Riemannian TNT solver. */
typedef Optimization::Riemannian::SeparableStructureUpdate<Matrix, Matrix>
    SeparableStructureUpdate;

} // namespace VarPro
