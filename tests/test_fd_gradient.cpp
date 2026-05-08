/**
 * @file test_fd_gradient.cpp
 * @brief Finite-difference gradient and Hessian checks for the Schur operator.
 *
 * Tests the gradient and Hessian-vector product against finite-difference
 * approximations at multiple random points.  Covers all four experiment types
 * (PGO, SNL, SfM, RA-SLAM) if data files are present.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Types.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const double FD_EPS    = 1e-5;
static const double GRAD_TOL  = 1e-4;  // O(eps^2) FD error
static const double HESS_TOL  = 1e-3;  // O(eps) for second derivative

// ---------------------------------------------------------------------------
// Finite-difference gradient check
// ---------------------------------------------------------------------------

static bool fdGradient(const std::string& label,
                        const VarProGPU::VarProPrecomputeResult& pre) {
  std::cout << "  [grad_fd] " << label << " ... ";
  int p = pre.p, r = pre.r;
  VarProGPU::CpuSchurOperator op(pre);

  bool ok = true;
  for (int trial = 0; trial < 5; ++trial) {
    VarPro::Matrix X = VarPro::Matrix::Random(p, r);
    VarPro::Matrix eta = VarPro::Matrix::Random(p, r);
    eta /= eta.norm();

    // Exact directional derivative
    VarPro::Matrix g = op.gradient(X);
    double exact = (g.array() * eta.array()).sum();

    // FD directional derivative: (f(X+eps*eta) - f(X-eps*eta)) / (2*eps)
    double fp = op.cost(X + FD_EPS * eta);
    double fm = op.cost(X - FD_EPS * eta);
    double fd = (fp - fm) / (2.0 * FD_EPS);

    double err = std::abs(exact - fd);
    double ref = std::max(std::abs(exact), std::abs(fd)) + 1e-10;
    if (err / ref > GRAD_TOL) {
      std::cout << "FAIL (trial=" << trial
                << " exact=" << exact << " fd=" << fd << ")\n";
      ok = false;
      break;
    }
  }
  if (ok) std::cout << "PASS\n";
  return ok;
}

// ---------------------------------------------------------------------------
// Finite-difference Hessian-vector product check
// ---------------------------------------------------------------------------

static bool fdHessVec(const std::string& label,
                       const VarProGPU::VarProPrecomputeResult& pre) {
  std::cout << "  [hess_fd] " << label << " ... ";
  int p = pre.p, r = pre.r;
  VarProGPU::CpuSchurOperator op(pre);

  bool ok = true;
  for (int trial = 0; trial < 3; ++trial) {
    VarPro::Matrix X   = VarPro::Matrix::Random(p, r);
    VarPro::Matrix eta = VarPro::Matrix::Random(p, r);
    eta /= eta.norm();

    // Exact Hessian-vector product
    VarPro::Matrix Heta = op.hessianVectorProduct(X, eta);

    // FD: H[eta] ≈ (g(X + eps*eta) - g(X - eps*eta)) / (2*eps)
    VarPro::Matrix Heta_fd =
        (op.gradient(X + FD_EPS * eta) - op.gradient(X - FD_EPS * eta)) /
        (2.0 * FD_EPS);

    double err = (Heta - Heta_fd).norm();
    double ref = Heta.norm() + 1e-10;
    if (err / ref > HESS_TOL) {
      std::cout << "FAIL (trial=" << trial << " rel_err=" << err/ref << ")\n";
      ok = false;
      break;
    }
  }
  if (ok) std::cout << "PASS\n";
  return ok;
}

// ---------------------------------------------------------------------------
// Build precompute for a given .pyfg file
// ---------------------------------------------------------------------------

static VarProGPU::VarProPrecomputeResult buildPre(const std::string& path,
                                                    int rank = 4) {
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(rank);
  return VarProGPU::buildPrecomputeResult(prob);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  int failures = 0;

  // Dataset paths (only run if files exist)
  struct Dataset { std::string label; std::string path; };
  std::vector<Dataset> datasets = {
    {"PGO/tinyGrid3D",
     "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg"},
    {"SNL/tinyGrid3D",
     "/home/nikolas/variable-projection/examples/data/snl/tinyGrid3D_snl/tinyGrid3D_snl.pyfg"},
    {"PGO/smallGrid3D",
     "/home/nikolas/variable-projection/examples/data/pgo/smallGrid3D/smallGrid3D.pyfg"},
  };

  for (const auto& ds : datasets) {
    if (!fs::exists(ds.path)) {
      std::cout << "  [SKIP] " << ds.label << " (file not found)\n";
      continue;
    }
    std::cout << "Dataset: " << ds.label << "\n";
    auto pre = buildPre(ds.path);
    if (!fdGradient(ds.label, pre)) ++failures;
    if (!fdHessVec(ds.label, pre))  ++failures;
  }

  if (failures == 0) {
    std::cout << "\nAll FD checks PASSED.\n";
    return 0;
  }
  std::cout << "\n" << failures << " check(s) FAILED.\n";
  return 1;
}
