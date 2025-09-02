//
// Created by tim on 11/13/23.
//
#include <VarPro/VARPRO.h>
#include <VarPro/VARPRO_vis.h>

#include <thread> // NOLINT [build/c++11]

int main() {
  VarPro::Problem problem =
      VarPro::parsePyfgTextToProblem("./bin/data/factor_graph_small.pyfg");
  // "./bin/data/plaza2.pyfg");
  problem.updateProblemData();

  VarPro::Matrix x0 = problem.getRandomInitialGuess();

  int max_rank = 10;
  bool verbose = false;
  bool log_iterates = true;
  VarPro::ProblemResult res;
  res = solveProblem(problem, x0, max_rank, verbose, log_iterates);

  std::cout << "Testing with Random initialization" << std::endl;
  // Visualize the result
  VarPro::VarProVis viz{};
  double viz_hz = 10.0;
  // double viz_hz = 2.0;
  viz.run(problem, {res.second}, viz_hz, true);
  return 0;
}
