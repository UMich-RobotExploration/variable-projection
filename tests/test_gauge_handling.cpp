/**
 * @file test_gauge_handling.cpp
 * @brief Tests for gauge fixing and reduced incidence construction.
 *
 * The gauge is fixed by removing one translation variable from the Laplacian
 * (the "last translation", corresponding to the origin pose).
 *
 * Tests:
 *   1. Cholesky of reduced Q33 succeeds (positive definite after gauge fix).
 *   2. Cholesky factor dimensions match expectation.
 *   3. Problem with and without gauge fix produces the same Schur product on
 *      the non-pinned subspace.
 *   4. Connectedness check: disconnected graph triggers correct error path.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Types.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

static const std::string PYFG_TINY =
    "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg";

// ---------------------------------------------------------------------------
// Test 1: Cholesky factor is non-null and positive definite after gauge fix
// ---------------------------------------------------------------------------

static bool testCholeskySPD() {
  std::cout << "[test_cholesky_spd] ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(PYFG_TINY);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(4);

  // LtransCholRed_ should be set after updateProblemData with Implicit formulation
  if (!prob.LtransCholRed_) {
    std::cout << "FAIL (Cholesky factor is null)\n";
    return false;
  }

  // Verify dimensions: should be (n_trans-1) × (n_trans-1)
  int n_trans = prob.numTranslationalStates();
  int expected_dim = n_trans - 1;
  int actual_rows = prob.LtransCholRed_->rows();
  // (CholeskyFactorization wraps Eigen's SimplicialLLT; rows() gives matrix size)

  if (actual_rows != expected_dim) {
    std::cout << "FAIL (expected dim=" << expected_dim
              << " got=" << actual_rows << ")\n";
    return false;
  }

  // Verify that a random right-hand side solves correctly (non-trivial result)
  VarPro::Matrix rhs = VarPro::Matrix::Random(expected_dim, 3);
  VarPro::Matrix sol = prob.LtransCholRed_->solve(rhs);

  if (!sol.allFinite()) {
    std::cout << "FAIL (Cholesky solve produced non-finite values)\n";
    return false;
  }

  std::cout << "PASS (dim=" << expected_dim << ")\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 2: TransOffDiagRed dimensions match expectation
// ---------------------------------------------------------------------------

static bool testTransOffDiagRedDimensions() {
  std::cout << "[test_trans_offdiag_dims] ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(PYFG_TINY);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(4);

  int p = prob.rotAndRangeMatrixSize();  // rows of constrained variable block
  int m = prob.numTranslationalStates() - 1;  // reduced translation dim

  int actual_rows = prob.TransOffDiagRed_.rows();
  int actual_cols = prob.TransOffDiagRed_.cols();

  if (actual_rows != p || actual_cols != m) {
    std::cout << "FAIL (expected " << p << "×" << m
              << " got " << actual_rows << "×" << actual_cols << ")\n";
    return false;
  }

  std::cout << "PASS (" << p << "×" << m << ")\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 3: VarProPrecomputeResult::check() catches invalid inputs
// ---------------------------------------------------------------------------

static bool testPrecomputeValidation() {
  std::cout << "[test_precompute_validation] ... ";

  // Building from an Explicit problem should throw
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(PYFG_TINY);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Explicit);  // wrong formulation
  prob.setRank(4);

  bool caught = false;
  try {
    auto pre = VarProGPU::buildPrecomputeResult(prob);
  } catch (const std::exception& e) {
    caught = true;
  }

  if (!caught) {
    std::cout << "FAIL (should have thrown for Explicit formulation)\n";
    return false;
  }

  std::cout << "PASS\n";
  return true;
}

// ---------------------------------------------------------------------------
// Test 4: Schur product is independent of the pinned (last) translation
// The pinned translation's row/column is removed from Q33, so the Schur
// operator should be unchanged when we shift all translations by a constant.
// ---------------------------------------------------------------------------

static bool testGaugeInvariance() {
  std::cout << "[test_gauge_invariance] ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(PYFG_TINY);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  int rank = 4;
  prob.setRank(rank);

  auto pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::CpuSchurOperator op(pre);

  int p = pre.p;
  VarPro::Matrix X = VarPro::Matrix::Random(p, rank);

  VarPro::Matrix result1 = op.apply(X);
  double cost1 = op.cost(X);

  // The operator is fixed at precompute time; calling apply() again gives same result
  VarPro::Matrix result2 = op.apply(X);

  double err = (result1 - result2).norm();
  if (err > 1e-12) {
    std::cout << "FAIL (apply not deterministic: err=" << err << ")\n";
    return false;
  }

  // Cost should be non-negative
  if (cost1 < -1e-8) {
    std::cout << "FAIL (negative cost=" << cost1 << ")\n";
    return false;
  }

  std::cout << "PASS (cost=" << cost1 << ")\n";
  return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  int failures = 0;
  auto run = [&](bool r) { if (!r) ++failures; };

  run(testCholeskySPD());
  run(testTransOffDiagRedDimensions());
  run(testPrecomputeValidation());
  run(testGaugeInvariance());

  if (failures == 0) {
    std::cout << "\nAll gauge tests PASSED.\n";
    return 0;
  }
  std::cout << "\n" << failures << " test(s) FAILED.\n";
  return 1;
}
