#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarProGPU/RTRSolver.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#include <VarProGPU/GpuRTRSolver.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <iterator>
#include <filesystem>
#include <vector>
#include <experiment_utils.hpp>
int main() {
  auto prob = VarPro::parsePyfgTextToProblem(
    "/home/nikolas/variable-projection/examples/data/raslam/mrclam/mrclam2/mrclam2.pyfg");
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(5);
  auto pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::GpuContext ctx;
  VarProGPU::GpuSchurOperator op(pre, ctx);
  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  int failures = 0;
  std::string base = "/home/nikolas/variable-projection/examples/data/raslam/mrclam/mrclam2/inits/rank5_init";
  for (int i = 1; i <= 5; i++) {
    std::string fpath = base + std::to_string(i) + ".txt";
    VarPro::Matrix init = readInitializationFile(fpath, prob);
    VarProGPU::CpuRTRSolver cpu_solver;
    auto cpu_r = cpu_solver.solve(prob, init, params);
    VarProGPU::GpuRTRSolver gpu_solver(ctx);
    auto gpu_r = gpu_solver.solve(prob, op, init, params);
    std::cout << "init" << i
              << "  CPU: f=" << cpu_r.f << " iters=" << cpu_r.outer_iters << " stop=" << cpu_r.stop_reason
              << "  GPU: f=" << gpu_r.f << " iters=" << gpu_r.outer_iters << " stop=" << gpu_r.stop_reason
              << "\n";

    const double abs_diff = std::abs(cpu_r.f - gpu_r.f);
    const double rel_diff = abs_diff / (std::max(std::abs(cpu_r.f), std::abs(gpu_r.f)) + 1e-9);
    const bool cpu_converged = cpu_r.stop_reason != "max_iterations";
    const bool gpu_converged = gpu_r.stop_reason != "max_iterations";
    const bool cpu_fast_enough = cpu_r.outer_iters <= 80;
    const bool costs_match = rel_diff < 0.05;

    if (!cpu_converged || !gpu_converged || !cpu_fast_enough || !costs_match) {
      std::cerr << "MRCLAM2 implicit regression on init" << i
                << ": cpu_converged=" << cpu_converged
                << " gpu_converged=" << gpu_converged
                << " cpu_fast_enough=" << cpu_fast_enough
                << " rel_diff=" << rel_diff << "\n";
      ++failures;
    }
  }

  if (failures != 0) {
    return 1;
  }
}
