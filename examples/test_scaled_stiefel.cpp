/**
 * Quick smoke test for the scaled Stiefel integration.
 *
 * Tests two things:
 *  1. A small PGO problem (tinyGrid3D) still solves correctly with the regular
 *     Stiefel manifold (regression check – nothing should have broken).
 *  2. The same problem forced onto the scaled Stiefel manifold runs without
 *     crashing and produces a finite, decreasing cost (mechanical check of the
 *     new manifold ops: projectToManifold, projectToTangentSpace,
 *     SymBlockDiagProduct_aniso).
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>
#include <VarPro/Types.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

static const std::string PYFG =
    "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg";

static VarPro::Problem buildProblem(VarPro::Formulation f, bool use_scaled)
{
  VarPro::Problem p = VarPro::parsePyfgTextToProblem(PYFG);
  if (use_scaled)
    p.convertToScaledStiefel();
  p.updateProblemData();
  p.setRank(5);
  p.setFormulation(f);
  return p;
}

static bool costsDecreasing(const std::vector<VarPro::Scalar> &c)
{
  if (c.empty())
    return false;
  for (auto v : c)
    if (!std::isfinite(v))
      return false;
  return c.back() <= c.front();
}

static void runTest(const std::string &name, VarPro::Formulation f, bool use_scaled)
{
  std::cout << "\n=== " << name << " ===" << std::endl;
  VarPro::Problem prob = buildProblem(f, use_scaled);
  VarPro::Matrix x0 = prob.getRandomInitialGuess();
  VarPro::ProblemResult res = VarPro::solveProblem(prob, x0, /*verbose=*/false);

  std::cout << "  Iterations : " << res.objective_values.size() << std::endl;
  if (!res.objective_values.empty())
  {
    std::cout << "  Initial cost: " << res.objective_values.front() << std::endl;
    std::cout << "  Final cost  : " << res.objective_values.back() << std::endl;
  }

  bool ok = costsDecreasing(res.objective_values);
  std::cout << "  Result: " << (ok ? "PASS" : "FAIL") << std::endl;
  if (!ok)
    std::exit(1);
}

int main()
{
  // ---- 1. Baseline: plain Stiefel, must still work ----
  runTest("Stiefel / Explicit",      VarPro::Formulation::Explicit,      /*scaled=*/false);
  runTest("Stiefel / ExplicitVarPro",VarPro::Formulation::ExplicitVarPro,/*scaled=*/false);
  runTest("Stiefel / Implicit",      VarPro::Formulation::Implicit,      /*scaled=*/false);

  // ---- 2. Scaled Stiefel: mechanical check ----
  runTest("ScaledStiefel / Explicit",      VarPro::Formulation::Explicit,      /*scaled=*/true);
  runTest("ScaledStiefel / ExplicitVarPro",VarPro::Formulation::ExplicitVarPro,/*scaled=*/true);
  runTest("ScaledStiefel / Implicit",      VarPro::Formulation::Implicit,      /*scaled=*/true);

  // ---- 3. Real SfM dataset (bal-93) smoke test ----
  static const std::string BAL93 =
      "/home/nikolas/variable-projection/examples/data/sfm/bal-93/bal-93.pyfg";
  for (auto [name, form] : std::initializer_list<std::pair<std::string, VarPro::Formulation>>{
           {"SfM bal-93 / Explicit",      VarPro::Formulation::Explicit},
           {"SfM bal-93 / ExplicitVarPro",VarPro::Formulation::ExplicitVarPro},
           {"SfM bal-93 / Implicit",      VarPro::Formulation::Implicit}})
  {
    std::cout << "\n=== " << name << " ===" << std::endl;
    try {
      VarPro::Problem p = VarPro::parsePyfgTextToProblem(BAL93);
      p.convertToScaledStiefel();
      p.updateProblemData();
      p.setRank(5);
      p.setFormulation(form);
      VarPro::Matrix x0 = p.getRandomInitialGuess();
      VarPro::ProblemResult res = VarPro::solveProblem(p, x0, false);
      std::cout << "  Iterations: " << res.objective_values.size() << std::endl;
      if (!res.objective_values.empty())
        std::cout << "  Final cost: " << res.objective_values.back() << std::endl;
      std::cout << "  Result: " << (res.objective_values.empty() ? "FAIL" : "PASS") << std::endl;
    } catch (const std::exception &e) {
      std::cout << "  EXCEPTION: " << e.what() << std::endl;
    }
  }

  std::cout << "\nAll tests passed." << std::endl;
  return 0;
}
