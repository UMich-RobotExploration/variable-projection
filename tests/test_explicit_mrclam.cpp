#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
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
  prob.setFormulation(VarPro::Formulation::Explicit);
  prob.setRank(5);
  prob.updateProblemData();
  std::string fpath = "/home/nikolas/variable-projection/examples/data/raslam/mrclam/mrclam6/inits/rank5_init1.txt";
  VarPro::Matrix init = readInitializationFile(fpath, prob);
  std::cout << "Explicit init: " << init.rows() << "x" << init.cols() << " norm=" << init.norm() << "\n";
  VarProGPU::RTRParams params;
  params.max_outer_iters = 250;
  params.verbose = true;
  // CPU RTR
  VarProGPU::CpuRTRSolver cpu_solver;
  auto cpu_r = cpu_solver.solve(prob, init, params);
  std::cout << "\nCpuRTR Explicit: f=" << cpu_r.f << " iters=" << cpu_r.outer_iters << " stop=" << cpu_r.stop_reason << "\n\n";
  // GPU RTR
  VarProGPU::GpuContext ctx;
  VarProGPU::GpuExplicitOperator gpu_op(prob, ctx);
  VarProGPU::GpuRTRSolver gpu_solver(ctx);
  auto gpu_r = gpu_solver.solveExplicit(prob, gpu_op, init, params);
  std::cout << "GpuRTR Explicit: f=" << gpu_r.f << " iters=" << gpu_r.outer_iters << " stop=" << gpu_r.stop_reason << "\n";
}
