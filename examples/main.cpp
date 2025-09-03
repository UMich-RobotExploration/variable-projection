#include <VarPro/Solver.h>
#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/PyfgTextParser.h>

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " [input .pyfg file]" << std::endl;
    exit(1);
  }

  VarPro::Problem problem = VarPro::parsePyfgTextToProblem(argv[1]);
  problem.updateProblemData();

#ifdef GPERFTOOLS
  ProfilerStart("varpro.prof");
#endif

  VarPro::Matrix x0 = problem.getRandomInitialGuess();
  int max_rank = 10;

  VarPro::ProblemResult soln = VarPro::solveProblem(problem, x0, max_rank);
  VarPro::Matrix aligned_soln = problem.alignEstimateToOrigin(soln.first.x);

  // std::cout << "Solution: " << std::endl;
  // std::cout << aligned_soln << std::endl;

#ifdef GPERFTOOLS
  ProfilerStop();
#endif
}
