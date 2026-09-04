/**
 * @file test_dense_vs_implicit.cpp
 * @brief Equivalence test for Formulation::Dense against Formulation::Implicit.
 *
 * Dense forms the reduced system Q_sc = Q_c - B M^{-1} B^T explicitly and
 * multiplies by it; Implicit applies the same operator matrix-free. They must
 * therefore agree to round-off on:
 *
 *   1. the operator itself, and the objective / Riemannian gradient /
 *      Riemannian Hessian-vector product it induces, at arbitrary iterates,
 *   2. the opening iterations of a full solve from a common init.
 *
 * Note on (2): we deliberately do *not* require the two cost trajectories to
 * agree all the way to convergence. These are nonconvex trust-region solves,
 * and on the harder datasets they are chaotic at round-off level -- running
 * Implicit against itself from an initial iterate perturbed by 1e-16 relative
 * changes the iteration count by ~20% and the final cost by ~4e-4 on plaza1.
 * A whole-trajectory equality assertion would therefore be testing the
 * conditioning of the dataset, not the correctness of the dense operator. The
 * operator checks in (1) are the real gate; (2) confirms the two formulations
 * start down the identical path.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>
#include <VarPro/Types.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifndef VARPRO_EXAMPLES_DIR
#define VARPRO_EXAMPLES_DIR "/home/nikolas/variable-projection/examples"
#endif

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what << std::endl;
  if (!ok) ++g_failures;
}

VarPro::Problem buildProblem(const std::string &pyfg, int rank) {
  VarPro::Problem p = VarPro::parsePyfgTextToProblem(pyfg);
  if (p.isSfmProblem()) {
    p.convertToScaledStiefel();
    p.setScaleRegWeight(1e-2);
  }
  p.updateProblemData();
  p.setRank(rank);
  return p;
}

void runDataset(const std::string &name, const std::string &pyfg, int rank) {
  std::cout << "\n=== " << name << " (rank " << rank << ") ===" << std::endl;

  VarPro::Problem prob = buildProblem(pyfg, rank);

  // Common init, generated once in the reduced (marginalized) variable size.
  prob.setFormulation(VarPro::Formulation::Implicit);
  const VarPro::Matrix Y0 = prob.getRandomInitialGuess();

  // --- 1. operator + gradient agreement, at Y0 and at a second point --------
  auto compareAt = [&](const VarPro::Matrix &Y, const std::string &where) {
    prob.setFormulation(VarPro::Formulation::Implicit);
    const VarPro::Scalar cost_impl = prob.evaluateObjective(Y);
    const VarPro::Matrix egrad_impl = prob.Euclidean_gradient(Y);
    const VarPro::Matrix rgrad_impl = prob.Riemannian_gradient(Y);
    const VarPro::Matrix hess_impl =
        prob.Riemannian_Hessian_vector_product(Y, egrad_impl, Y);

    prob.setFormulation(VarPro::Formulation::Dense);
    const VarPro::Scalar cost_dense = prob.evaluateObjective(Y);
    const VarPro::Matrix egrad_dense = prob.Euclidean_gradient(Y);
    const VarPro::Matrix rgrad_dense = prob.Riemannian_gradient(Y);
    const VarPro::Matrix hess_dense =
        prob.Riemannian_Hessian_vector_product(Y, egrad_dense, Y);

    const double cost_rel =
        std::abs(cost_dense - cost_impl) / std::max(std::abs(cost_impl), 1.0);
    const double egrad_rel =
        (egrad_dense - egrad_impl).norm() / std::max(egrad_impl.norm(), 1.0);
    const double rgrad_rel =
        (rgrad_dense - rgrad_impl).norm() / std::max(rgrad_impl.norm(), 1.0);
    const double hess_rel =
        (hess_dense - hess_impl).norm() / std::max(hess_impl.norm(), 1.0);

    std::cout << "  " << where << ": rel diff cost=" << cost_rel
              << " egrad=" << egrad_rel << " rgrad=" << rgrad_rel
              << " hess=" << hess_rel << std::endl;

    check(cost_rel < 1e-10, where + ": objective matches");
    check(egrad_rel < 1e-10, where + ": Euclidean gradient matches");
    check(rgrad_rel < 1e-10, where + ": Riemannian gradient matches");
    check(hess_rel < 1e-10, where + ": Riemannian Hessian-vector product matches");
  };

  prob.setFormulation(VarPro::Formulation::Dense);
  std::cout << "  dense Q_sc: " << prob.densePrecomputeGB() << " GB, formed in "
            << prob.getDensePrecomputeTimeS() << " s" << std::endl;
  check(prob.hasDenseSchur(), "dense Schur complement was formed");

  compareAt(Y0, "at Y0");
  compareAt(prob.projectToManifold(prob.getRandomInitialGuess()), "at Y1");

  // --- 2. reduced-variable size is identical --------------------------------
  prob.setFormulation(VarPro::Formulation::Implicit);
  const int size_impl = prob.getExpectedVariableSize();
  prob.setFormulation(VarPro::Formulation::Dense);
  const int size_dense = prob.getExpectedVariableSize();
  check(size_impl == size_dense, "reduced variable size matches");

  // --- 3. the solve starts down the identical path --------------------------
  prob.setFormulation(VarPro::Formulation::Implicit);
  VarPro::ProblemResult res_impl = VarPro::solveProblem(prob, Y0, false);
  prob.setFormulation(VarPro::Formulation::Dense);
  VarPro::ProblemResult res_dense = VarPro::solveProblem(prob, Y0, false);

  std::cout << "  implicit: " << res_impl.objective_values.size()
            << " iters, final cost " << res_impl.f << std::endl;
  std::cout << "  dense   : " << res_dense.objective_values.size()
            << " iters, final cost " << res_dense.f << std::endl;

  // Only the opening iterations are required to match -- see the file header
  // for why whole-trajectory equality is not a valid assertion here.
  constexpr size_t kEarlyIters = 10;
  const size_t n = std::min({kEarlyIters, res_impl.objective_values.size(),
                             res_dense.objective_values.size()});
  check(n > 1, "both formulations produced a cost trajectory");
  double early_rel = 0.0, full_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double a = res_impl.objective_values[i];
    const double b = res_dense.objective_values[i];
    early_rel = std::max(early_rel, std::abs(a - b) / std::max(std::abs(a), 1.0));
  }
  for (size_t i = 0; i < std::min(res_impl.objective_values.size(),
                                  res_dense.objective_values.size());
       ++i) {
    const double a = res_impl.objective_values[i];
    const double b = res_dense.objective_values[i];
    full_rel = std::max(full_rel, std::abs(a - b) / std::max(std::abs(a), 1.0));
  }
  std::cout << "  max rel cost diff: first " << n << " iters = " << early_rel
            << ", whole trajectory = " << full_rel << " (informational)"
            << std::endl;
  check(early_rel < 1e-10, "opening cost trajectory matches");

  // --- 4. switching away from Dense releases the O(p^2) allocation ----------
  prob.setFormulation(VarPro::Formulation::Implicit);
  check(!prob.hasDenseSchur(), "dense Schur released on formulation switch");

  // --- 5. updateProblemData() refreshes a stale dense Schur (the IRLS path) --
  prob.setFormulation(VarPro::Formulation::Dense);
  auto &rpms = prob.getMutableRPMs();
  for (auto &m : rpms) m.cov *= 2.0;   // reweight, as one IRLS iteration would
  prob.updateProblemData();
  check(prob.hasDenseSchur(), "dense Schur rebuilt by updateProblemData()");

  const VarPro::Scalar cost_reweighted_dense = prob.evaluateObjective(Y0);
  prob.setFormulation(VarPro::Formulation::Implicit);
  const VarPro::Scalar cost_reweighted_impl = prob.evaluateObjective(Y0);
  const double reweight_rel =
      std::abs(cost_reweighted_dense - cost_reweighted_impl) /
      std::max(std::abs(cost_reweighted_impl), 1.0);
  std::cout << "  rel diff after reweight: " << reweight_rel << std::endl;
  check(reweight_rel < 1e-10, "reweighted objective matches (IRLS path)");
}

}  // namespace

int main() {
  const std::string examples = VARPRO_EXAMPLES_DIR;

  // PGO (no ranges, no landmarks)
  runDataset("tinyGrid3D", examples + "/data/pgo/tinyGrid3D/tinyGrid3D.pyfg", 5);
  // Range-aided SLAM (exercises the Oblique block of the reduced variable).
  // Deliberately a small one: on plaza1 the dense Q_sc is 4.1 GB and the
  // solve exhausts the solver's wall-clock budget, which is the paper result
  // but makes for a bad unit test.
  runDataset("single_drone", examples + "/data/raslam/single_drone/single_drone.pyfg", 5);
  // SfM (exercises the scaled-Stiefel manifold + scale regularizer)
  runDataset("bal-93", examples + "/data/sfm/bal-93/bal-93.pyfg", 5);

  std::cout << "\n"
            << (g_failures == 0 ? "All dense-vs-implicit checks passed."
                                : std::to_string(g_failures) + " CHECK(S) FAILED")
            << std::endl;
  return g_failures == 0 ? 0 : 1;
}
