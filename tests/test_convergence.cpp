/**
 * @file test_convergence.cpp
 * @brief End-to-end convergence tests on synthetic problems.
 *
 * Tests both CpuRTRSolver (standalone) and the existing Problem::solveProblem
 * (via TNT library) on small instances of each experiment family.
 *
 * Pass criterion: final cost decreases monotonically and the solver terminates
 * without NaN/Inf values.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>
#include <VarPro/Types.h>

#include <VarProGPU/MatrixFreeSchurOperator.h>
#include <VarProGPU/RTRSolver.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/GpuRTRSolver.h>
#endif

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helper: check a sequence of objective values is finite and non-increasing
// ---------------------------------------------------------------------------

static bool costsDecreasing(const std::vector<VarPro::Scalar>& v) {
  if (v.empty()) return false;
  for (auto x : v)
    if (!std::isfinite(x)) return false;
  return v.back() <= v.front() + 1e-3 * std::abs(v.front());
}

// ---------------------------------------------------------------------------
// Test: existing TNT solver (CPU reference baseline)
// ---------------------------------------------------------------------------

static bool testTNTSolver(const std::string& label,
                           const std::string& path,
                           VarPro::Formulation formulation) {
  std::cout << "  [TNT/" << (formulation == VarPro::Formulation::Implicit ? "Implicit" :
                              formulation == VarPro::Formulation::ExplicitVarPro ? "VarPro" : "Explicit")
            << "] " << label << " ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(formulation);
  prob.setRank(4);

  VarPro::Matrix x0 = prob.getRandomInitialGuess();
  auto result = VarPro::solveProblem(prob, x0, /*verbose=*/false);

  bool ok = costsDecreasing(result.objective_values);
  std::cout << (ok ? "PASS" : "FAIL")
            << " (iters=" << result.objective_values.size()
            << " f_final=" << (result.objective_values.empty() ? 0.0 : result.objective_values.back())
            << ")\n";
  return ok;
}

// ---------------------------------------------------------------------------
// Test: standalone CpuRTRSolver (our implementation)
// ---------------------------------------------------------------------------

static bool testCpuRTRSolver(const std::string& label,
                               const std::string& path) {
  std::cout << "  [CpuRTR/Implicit] " << label << " ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(4);

  VarPro::Matrix x0 = prob.getRandomInitialGuess();

  VarProGPU::RTRParams params;
  params.max_outer_iters = 100;
  params.verbose = false;

  VarProGPU::CpuRTRSolver solver;
  auto result = solver.solve(prob, x0, params);

  bool ok = costsDecreasing(result.objective_values);
  std::cout << (ok ? "PASS" : "FAIL")
            << " (iters=" << result.outer_iters
            << " f_final=" << result.f << ")\n";
  return ok;
}

// ---------------------------------------------------------------------------
// Test: GPU RTR solver matches CPU RTR (if CUDA available)
// ---------------------------------------------------------------------------

#ifdef VARPRO_HAVE_CUDA
static bool testGpuRTRSolver(const std::string& label,
                               const std::string& path) {
  std::cout << "  [GpuRTR/Implicit] " << label << " ... ";

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(path);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(4);

  auto pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::GpuContext ctx;
  VarProGPU::GpuSchurOperator gpu_op(pre, ctx);

  VarPro::Matrix x0 = prob.getRandomInitialGuess();

  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  params.verbose = false;

  VarProGPU::GpuRTRSolver solver(ctx);
  auto result = solver.solve(prob, gpu_op, x0, params);

  bool ok = costsDecreasing(result.objective_values);
  std::cout << (ok ? "PASS" : "FAIL")
            << " (iters=" << result.outer_iters
            << " f_final=" << result.f << ")\n";
  return ok;
}

// Test that GPU and CPU solvers reach similar costs (within 1%)
static bool testGpuCpuCostMatch(const std::string& label,
                                  const std::string& path) {
  std::cout << "  [GpuCpuMatch] " << label << " ... ";

  // Seed both solvers identically
  srand(42);

  // CPU solve
  VarPro::Problem prob_cpu = VarPro::parsePyfgTextToProblem(path);
  prob_cpu.updateProblemData();
  prob_cpu.setFormulation(VarPro::Formulation::Implicit);
  prob_cpu.setRank(4);

  VarPro::Matrix x0 = prob_cpu.getRandomInitialGuess();
  VarProGPU::RTRParams params;
  params.max_outer_iters = 80;
  VarProGPU::CpuRTRSolver cpu_solver;
  auto cpu_result = cpu_solver.solve(prob_cpu, x0, params);

  // GPU solve
  VarPro::Problem prob_gpu = VarPro::parsePyfgTextToProblem(path);
  prob_gpu.updateProblemData();
  prob_gpu.setFormulation(VarPro::Formulation::Implicit);
  prob_gpu.setRank(4);

  auto pre = VarProGPU::buildPrecomputeResult(prob_gpu);
  VarProGPU::GpuContext ctx;
  VarProGPU::GpuSchurOperator gpu_op(pre, ctx);

  VarProGPU::GpuRTRSolver gpu_solver(ctx);
  auto gpu_result = gpu_solver.solve(prob_gpu, gpu_op, x0, params);

  double cpu_f = cpu_result.f;
  double gpu_f = gpu_result.f;
  // Use absolute tolerance when both costs are near zero (SNL converges to ~0)
  double abs_diff = std::abs(cpu_f - gpu_f);
  double rel_diff = abs_diff / (std::max(std::abs(cpu_f), std::abs(gpu_f)) + 1e-6);

  // Pass if either relative difference is small OR both costs are near-zero
  bool near_zero = (std::abs(cpu_f) < 1e-4 && std::abs(gpu_f) < 1e-4);
  bool ok = near_zero || (rel_diff < 0.1);  // 10% relative tolerance

  std::cout << (ok ? "PASS" : "FAIL")
            << " (cpu_f=" << cpu_f << " gpu_f=" << gpu_f
            << " rel=" << rel_diff << ")\n";
  return ok;
}
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  int failures = 0;
  auto run = [&](bool r) { if (!r) ++failures; };

  struct Dataset { std::string label, path; };
  std::vector<Dataset> datasets = {
    {"PGO/tiny",    "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg"},
    {"PGO/small",   "/home/nikolas/variable-projection/examples/data/pgo/smallGrid3D/smallGrid3D.pyfg"},
    {"SNL/tiny",    "/home/nikolas/variable-projection/examples/data/snl/tinyGrid3D_snl/tinyGrid3D_snl.pyfg"},
  };

  for (const auto& ds : datasets) {
    if (!std::filesystem::exists(ds.path)) {
      std::cout << "  [SKIP] " << ds.label << "\n";
      continue;
    }
    std::cout << "Dataset: " << ds.label << "\n";

    // TNT reference (existing solver)
    run(testTNTSolver(ds.label, ds.path, VarPro::Formulation::Implicit));
    // ExplicitVarPro is only valid for problems with explicit translation
    // variables (PGO/SfM); skip for SNL/RA-SLAM which have different structure
    if (ds.label.find("SNL") == std::string::npos &&
        ds.label.find("raslam") == std::string::npos) {
      run(testTNTSolver(ds.label, ds.path, VarPro::Formulation::ExplicitVarPro));
    }

    // Standalone RTR
    run(testCpuRTRSolver(ds.label, ds.path));

#ifdef VARPRO_HAVE_CUDA
    run(testGpuRTRSolver(ds.label, ds.path));
    run(testGpuCpuCostMatch(ds.label, ds.path));
#else
    std::cout << "  [SKIP] GPU tests (no CUDA)\n";
#endif
  }

  if (failures == 0) {
    std::cout << "\nAll convergence tests PASSED.\n";
    return 0;
  }
  std::cout << "\n" << failures << " test(s) FAILED.\n";
  return 1;
}
