/**
 * @file benchmark_solver.cpp
 * @brief End-to-end solver benchmark reproducing paper-style comparisons.
 *
 * Reports (per dataset × formulation × backend):
 *   - Preprocessing time
 *   - Solve time
 *   - Total time
 *   - Number of RTR (outer) iterations
 *   - Number of pTCG (inner) iterations (total)
 *   - Relative speedup over CPU baseline
 *   - Final objective value
 *
 * Usage:
 *   ./benchmark_solver [--dataset <dir>] [--rank <r>] [--seeds <N>]
 *
 * The "dataset" argument can be a .pyfg file or a directory containing one.
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

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double now_sec() {
  static auto t0 = Clock::now();
  return Sec(Clock::now() - t0).count();
}

static std::string findPyfg(const std::string& path) {
  if (fs::is_regular_file(path) &&
      path.size() >= 5 && path.rfind(".pyfg") == path.size() - 5) return path;
  if (fs::is_directory(path)) {
    for (const auto& e : fs::directory_iterator(path)) {
      if (e.path().extension() == ".pyfg") return e.path().string();
    }
  }
  return "";
}

struct SolveStats {
  std::string backend;
  std::string formulation;
  double preprocess_s;
  double solve_s;
  double total_s;
  int outer_iters;
  int total_inner_iters;
  double final_cost;
};

static void printHeader() {
  std::cout << std::left
            << std::setw(16) << "Backend"
            << std::setw(14) << "Formulation"
            << std::setw(10) << "Prep(s)"
            << std::setw(10) << "Solve(s)"
            << std::setw(10) << "Total(s)"
            << std::setw(8)  << "Outer"
            << std::setw(8)  << "Inner"
            << std::setw(14) << "Final cost"
            << "\n";
  std::cout << std::string(90, '-') << "\n";
}

static void printRow(const SolveStats& s) {
  std::cout << std::fixed << std::setprecision(4)
            << std::left
            << std::setw(16) << s.backend
            << std::setw(14) << s.formulation
            << std::setw(10) << s.preprocess_s
            << std::setw(10) << s.solve_s
            << std::setw(10) << s.total_s
            << std::setw(8)  << s.outer_iters
            << std::setw(8)  << s.total_inner_iters
            << std::setprecision(6) << std::setw(14) << s.final_cost
            << "\n";
}

// ---------------------------------------------------------------------------
// CPU TNT baseline
// ---------------------------------------------------------------------------

static SolveStats runTNT(const std::string& pyfg, int rank,
                          VarPro::Formulation form) {
  SolveStats s;
  s.formulation = (form == VarPro::Formulation::Implicit) ? "Implicit" :
                  (form == VarPro::Formulation::ExplicitVarPro) ? "VarPro" : "Explicit";
  s.backend = "TNT (CPU)";

  double t_start = now_sec();
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  prob.setFormulation(form);
  prob.setRank(rank);
  s.preprocess_s = now_sec() - t_start;

  VarPro::Matrix x0 = prob.getRandomInitialGuess();
  double t_solve = now_sec();
  auto result = VarPro::solveProblem(prob, x0, /*verbose=*/false);
  s.solve_s = now_sec() - t_solve;
  s.total_s = now_sec() - t_start;

  s.outer_iters = static_cast<int>(result.objective_values.size());
  s.total_inner_iters =
      std::accumulate(result.inner_iterations.begin(),
                      result.inner_iterations.end(), 0);
  s.final_cost = result.f;
  return s;
}

// ---------------------------------------------------------------------------
// CPU standalone RTR
// ---------------------------------------------------------------------------

static SolveStats runCpuRTR(const std::string& pyfg, int rank) {
  SolveStats s;
  s.formulation = "Implicit";
  s.backend = "CpuRTR";

  double t_start = now_sec();
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(rank);
  s.preprocess_s = now_sec() - t_start;

  VarPro::Matrix x0 = prob.getRandomInitialGuess();
  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  params.verbose = false;

  double t_solve = now_sec();
  VarProGPU::CpuRTRSolver solver;
  auto result = solver.solve(prob, x0, params);
  s.solve_s = now_sec() - t_solve;
  s.total_s = now_sec() - t_start;

  s.outer_iters = result.outer_iters;
  s.total_inner_iters =
      std::accumulate(result.inner_iters_per_outer.begin(),
                      result.inner_iters_per_outer.end(), 0);
  s.final_cost = result.f;
  return s;
}

// ---------------------------------------------------------------------------
// GPU RTR
// ---------------------------------------------------------------------------

#ifdef VARPRO_HAVE_CUDA
static SolveStats runGpuRTR(const std::string& pyfg, int rank,
                              VarProGPU::GpuContext& ctx) {
  SolveStats s;
  s.formulation = "Implicit";
  s.backend = "GpuRTR";

  double t_start = now_sec();
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(rank);
  auto pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::GpuSchurOperator gpu_op(pre, ctx);
  ctx.synchronize();
  s.preprocess_s = now_sec() - t_start;

  VarPro::Matrix x0 = prob.getRandomInitialGuess();
  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  params.verbose = false;

  double t_solve = now_sec();
  VarProGPU::GpuRTRSolver solver(ctx);
  auto result = solver.solve(prob, gpu_op, x0, params);
  ctx.synchronize();
  s.solve_s = now_sec() - t_solve;
  s.total_s = now_sec() - t_start;

  s.outer_iters = result.outer_iters;
  s.total_inner_iters =
      std::accumulate(result.inner_iters_per_outer.begin(),
                      result.inner_iters_per_outer.end(), 0);
  s.final_cost = result.f;
  return s;
}
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string dataset_arg =
      "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D";
  int rank  = 5;
  int seeds = 1;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dataset") == 0 && i + 1 < argc)
      dataset_arg = argv[++i];
    else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc)
      rank = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--seeds") == 0 && i + 1 < argc)
      seeds = std::atoi(argv[++i]);
  }

  std::string pyfg = findPyfg(dataset_arg);
  if (pyfg.empty()) {
    std::cerr << "No .pyfg file found at: " << dataset_arg << "\n";
    return 1;
  }

  std::cout << "=== End-to-End Solver Benchmark ===\n"
            << "Dataset: " << pyfg << "\n"
            << "Rank:    " << rank  << "\n"
            << "Seeds:   " << seeds << "\n\n";

  printHeader();

  for (int seed = 0; seed < seeds; ++seed) {
    srand(seed + 42);

    // TNT baselines
    printRow(runTNT(pyfg, rank, VarPro::Formulation::Implicit));
    printRow(runTNT(pyfg, rank, VarPro::Formulation::ExplicitVarPro));

    // Standalone CPU RTR
    printRow(runCpuRTR(pyfg, rank));

#ifdef VARPRO_HAVE_CUDA
    VarProGPU::GpuContext ctx;
    printRow(runGpuRTR(pyfg, rank, ctx));
#else
    std::cout << "(GPU benchmark skipped: no CUDA)\n";
#endif
  }

  return 0;
}
