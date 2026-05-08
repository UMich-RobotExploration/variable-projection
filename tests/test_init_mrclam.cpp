#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>
#include <VarProGPU/RTRSolver.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#include <VarProGPU/GpuRTRSolver.h>
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
    "/home/nikolas/variable-projection/examples/data/raslam/mrclam/mrclam6/mrclam6.pyfg");
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(5);
  auto pre = VarProGPU::buildPrecomputeResult(prob);
  VarProGPU::GpuContext ctx;
  VarProGPU::GpuSchurOperator op(pre, ctx);
  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  std::string base = "/home/nikolas/variable-projection/examples/data/raslam/mrclam/mrclam6/inits/rank5_init";
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
  }
}
