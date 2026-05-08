/**
 * @file test_schur_vs_explicit.cpp
 * @brief Unit test: matrix-free Schur product matches explicit Schur matrix.
 *
 * Tests:
 *   1. CpuSchurOperator::apply() vs ExplicitSchurReference::apply() on a
 *      small synthetic PGO problem (tinyGrid3D).
 *   2. The same test using GpuSchurOperator if CUDA is available.
 *   3. Regression: operator is symmetric (Q_sc = Q_sc^T).
 *   4. Regression: operator is PSD (all eigenvalues >= 0 after gauge fix).
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Types.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

// Path to smallest dataset
static const std::string PYFG_TINY =
    "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg";

static const double TOL = 1e-8;  // relative error tolerance

// ---------------------------------------------------------------------------
// Build a small Implicit-formulation problem and extract precompute result
// ---------------------------------------------------------------------------

static VarProGPU::VarProPrecomputeResult buildPrecompute(
    const std::string& path, int rank = 4) {
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(rank);
  return VarProGPU::buildPrecomputeResult(prob);
}

// ---------------------------------------------------------------------------
// Test 1: matrix-free vs explicit Schur on random input
// ---------------------------------------------------------------------------

static bool testSchurVsExplicit(const std::string& path) {
  std::cout << "[test_schur_vs_explicit] " << path << " ... ";
  VarProGPU::VarProPrecomputeResult pre = buildPrecompute(path);
  int p = pre.p, r = pre.r;

  VarProGPU::CpuSchurOperator mf(pre);
  VarProGPU::ExplicitSchurReference ex(pre);

  // Generate 5 random input matrices and compare
  for (int trial = 0; trial < 5; ++trial) {
    VarPro::Matrix X = VarPro::Matrix::Random(p, r);

    VarPro::Matrix mf_result = mf.apply(X);
    VarPro::Matrix ex_result = ex.apply(X);

    double err = (mf_result - ex_result).norm();
    double ref = ex_result.norm();
    double rel = (ref > 1e-12) ? (err / ref) : err;

    if (rel > TOL) {
      std::cout << "FAIL (trial " << trial << ", rel_err=" << rel << ")\n";
      return false;
    }
  }
  std::cout << "PASS\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 2: Schur operator symmetry  Q_sc == Q_sc^T
// ---------------------------------------------------------------------------

static bool testSchurSymmetry(const std::string& path) {
  std::cout << "[test_schur_symmetry] " << path << " ... ";
  VarProGPU::VarProPrecomputeResult pre = buildPrecompute(path);
  int p = pre.p;

  VarProGPU::ExplicitSchurReference ex(pre);
  const VarPro::Matrix& Qsc = ex.matrix();

  double asym = (Qsc - Qsc.transpose()).norm();
  double scale = Qsc.norm();
  double rel = (scale > 1e-12) ? (asym / scale) : asym;

  if (rel > TOL) {
    std::cout << "FAIL (asymmetry=" << rel << ")\n";
    return false;
  }
  std::cout << "PASS (asym=" << rel << ")\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 3: cost = 0.5 * tr(X^T Q_sc X) is non-negative
// ---------------------------------------------------------------------------

static bool testCostNonNegative(const std::string& path) {
  std::cout << "[test_cost_nonneg] " << path << " ... ";
  VarProGPU::VarProPrecomputeResult pre = buildPrecompute(path);
  int p = pre.p, r = pre.r;

  VarProGPU::CpuSchurOperator mf(pre);

  for (int trial = 0; trial < 10; ++trial) {
    VarPro::Matrix X = VarPro::Matrix::Random(p, r);
    double cost = mf.cost(X);
    if (cost < -1e-6) {
      std::cout << "FAIL (negative cost=" << cost << " trial=" << trial << ")\n";
      return false;
    }
  }
  std::cout << "PASS\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 4: Hessian-vector product matches finite-difference approximation
// ---------------------------------------------------------------------------

static bool testHessianFiniteDiff(const std::string& path) {
  std::cout << "[test_hessian_fd] " << path << " ... ";
  VarProGPU::VarProPrecomputeResult pre = buildPrecompute(path);
  int p = pre.p, r = pre.r;

  VarProGPU::CpuSchurOperator mf(pre);

  VarPro::Matrix X   = VarPro::Matrix::Random(p, r);
  VarPro::Matrix eta = VarPro::Matrix::Random(p, r);
  eta /= eta.norm();

  // Exact Hessian-vector product
  VarPro::Matrix Heta_exact = mf.hessianVectorProduct(X, eta);

  // Finite-difference approximation:
  // Hess[f](X)[eta] ≈ (grad f(X + eps*eta) - grad f(X - eps*eta)) / (2*eps)
  double eps = 1e-5;
  VarPro::Matrix Heta_fd =
      (mf.gradient(X + eps * eta) - mf.gradient(X - eps * eta)) / (2.0 * eps);

  double err = (Heta_exact - Heta_fd).norm();
  double ref = Heta_exact.norm();
  double rel = (ref > 1e-12) ? (err / ref) : err;

  if (rel > 1e-4) {  // FD has O(eps^2) error
    std::cout << "FAIL (rel_err=" << rel << ")\n";
    return false;
  }
  std::cout << "PASS (rel_err=" << rel << ")\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 5: GPU vs CPU (if CUDA available)
// ---------------------------------------------------------------------------

#ifdef VARPRO_HAVE_CUDA
static bool testGpuVsCpu(const std::string& path) {
  std::cout << "[test_gpu_vs_cpu] " << path << " ... ";
  VarProGPU::VarProPrecomputeResult pre = buildPrecompute(path);
  int p = pre.p, r = pre.r;

  VarProGPU::GpuContext ctx;
  VarProGPU::CpuSchurOperator cpu_op(pre);
  VarProGPU::GpuSchurOperator gpu_op(pre, ctx);

  for (int trial = 0; trial < 3; ++trial) {
    VarPro::Matrix X = VarPro::Matrix::Random(p, r);

    VarPro::Matrix cpu_result = cpu_op.apply(X);
    VarPro::Matrix gpu_result = gpu_op.apply(X);

    double err = (cpu_result - gpu_result).norm();
    double ref = cpu_result.norm();
    double rel = (ref > 1e-12) ? (err / ref) : err;

    if (rel > 1e-7) {
      std::cout << "FAIL (trial=" << trial << " rel_err=" << rel << ")\n";
      return false;
    }
  }
  std::cout << "PASS\n";
  return true;
}
#endif

// ---------------------------------------------------------------------------
// Test 6: Schur operator consistency with Problem::dataMatrixProduct
// ---------------------------------------------------------------------------

static bool testSchurMatchesProblem(const std::string& path) {
  std::cout << "[test_schur_matches_problem] " << path << " ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  int rank = 4;
  prob.setRank(rank);

  // Build precompute BEFORE calling buildPrecomputeResult (which reads the
  // matrices already set up by updateProblemData)
  VarProGPU::VarProPrecomputeResult pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::CpuSchurOperator mf(pre);

  int p = pre.p, r = pre.r;

  for (int trial = 0; trial < 5; ++trial) {
    VarPro::Matrix X = VarPro::Matrix::Random(p, r);

    // Problem::dataMatrixProduct (internal, but accessible via gradient test)
    // We use evaluateObjective and gradient instead.
    VarPro::Matrix grad_prob = prob.Euclidean_gradient(X);
    VarPro::Matrix grad_mf   = mf.gradient(X);

    double err = (grad_prob - grad_mf).norm();
    double ref = grad_prob.norm();
    double rel = (ref > 1e-12) ? (err / ref) : err;

    if (rel > TOL) {
      std::cout << "FAIL (trial=" << trial << " rel_err=" << rel << ")\n";
      return false;
    }
  }
  std::cout << "PASS\n";
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  int failures = 0;

  auto run = [&](bool result) {
    if (!result) ++failures;
  };

  run(testSchurVsExplicit(PYFG_TINY));
  run(testSchurSymmetry(PYFG_TINY));
  run(testCostNonNegative(PYFG_TINY));
  run(testHessianFiniteDiff(PYFG_TINY));
  run(testSchurMatchesProblem(PYFG_TINY));

#ifdef VARPRO_HAVE_CUDA
  run(testGpuVsCpu(PYFG_TINY));
#else
  std::cout << "[test_gpu_vs_cpu] SKIP (no CUDA)\n";
#endif

  if (failures == 0) {
    std::cout << "\nAll tests PASSED.\n";
    return 0;
  } else {
    std::cout << "\n" << failures << " test(s) FAILED.\n";
    return 1;
  }
}
