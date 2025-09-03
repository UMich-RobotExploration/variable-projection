#include <VarPro/Solver.h>
#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Utils.h>
#include <VarPro/Symbol.h>
#include <VarPro/PyfgTextParser.h>

#include <filesystem>
#include <set>
#include <vector>

#include <unsupported/Eigen/SparseExtra>

#include <json.hpp>
#include <experiment_utils.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

struct Config
{
  bool verbose;
  std::string abs_data_path;
  int max_rank;
};

Config parseConfig(const std::string &filename)
{
  // check if the file exists
  if (!std::filesystem::exists(filename))
  {
    std::cout << "Looking for file: " << filename << std::endl;
    throw std::runtime_error("Config file does not exist");
  }

  std::ifstream file(filename);
  json j;
  file >> j;

  Config config;
  config.verbose = j["verbose"];
  config.max_rank = j["max_rank"];
  config.abs_data_path = j["abs_data_path"];

  return config;
}

VarPro::ProblemResult solveProblem(std::string pyfg_fpath, int init_rank_jump,
                                   int max_rank,
                                   VarPro::Formulation formulation, InitType init_type,
                                   bool verbose = true)
{

  // set the problem parameters
  problem.setRank(problem.dim() + init_rank_jump);
  problem.setFormulation(formulation);
  problem.setPreconditioner(VarPro::Preconditioner::RegularizedCholesky);

  // update the problem data
  problem.updateProblemData();

  VarPro::Matrix x0;
  if (init_type == InitType::Random)
  {
    x0 = problem.getRandomInitialGuess();
  }
  else if (init_type == InitType::Odom)
  {
    x0 = getOdomInitialization(problem, pyfg_fpath);
  }

  // if we're in implicit mode, then we need to truncate x0
  // to not have translation variables
  if (formulation == VarPro::Formulation::Implicit)
  {
    x0 = x0.block(0, 0, problem.rotAndRangeMatrixSize(), x0.cols());
  }

#ifdef GPERFTOOLS
  ProfilerStart("varpro.prof");
#endif

  // solve the problem
  VarPro::ProblemResult soln = VarPro::solveProblem(problem, x0, verbose);

#ifdef GPERFTOOLS
  ProfilerStop();
#endif

  // append the filename (e.g., mrclam7) and the time to "results.txt"
  std::ofstream results_file("results.txt", std::ios_base::app);
  results_file << pyfg_fpath << " " << elapsed.count() << std::endl;
  results_file.close();

  VarPro::Matrix aligned_soln = problem.alignEstimateToOrigin(soln.first.x);
  saveSolutions(problem, aligned_soln, pyfg_fpath);

  return aligned_soln;
}

// "rank3_init10.txt": {
//   "cost": 3687.256904220291,
//   "iterations": 542,
//   "time": 53.784879541,
//   "formulation": "ExplicitVarPro",
// }
std::vector<int> getRanksToSweep(int min_rank, int max_rank)
{
  std::vector<int> ranks;
  for (int r = min_rank; r <= max_rank; r++)
  {
    ranks.push_back(r);
  }
  return ranks;
}

std::vector<VarPro::Formulation> getFormulationsToSweep()
{
  return {VarPro::Formulation::Explicit,
          VarPro::Formulation::ExplicitVarPro,
          VarPro::Formulation::Implicit};
}

std::vector<std::vector<std::string>> makeInitializationFiles(const std::string &dataset_path,
                                                              const std::vector<int> &ranks)
{
  // start by making a 2d array to hold all of the initialization file paths
  // the array should have len(ranks) rows and 10 columns
  std::vector<std::vector<std::string>> init_file_paths = {};
  for (int r : ranks)
  {
    std::vector<std::string> rank_init_file_paths;
    for (int i = 1; i <= 10; i++)
    {
      rank_init_file_paths.push_back(dataset_path + "/rank" + std::to_string(r) +
                                     "_init" + std::to_string(i) + ".txt");
    }
    init_file_paths.push_back(rank_init_file_paths);
  }
}

/**
 * @brief Takes as input the directory that contains a .pyfg file and many different
 * initializations (e.g., rank3_init10.txt, rank4_init10.txt, etc.)
 *
 * @param dataset_path the path to the dataset directory
 */
void sweepDataset(fs::path dataset_path)
{

  // find the .pyfg file in the directory
  std::string pyfg_fpath = findPyfgInDir(dataset_path).string();
  Config config = parseConfig("/home/alan/variable-projection/examples/config.json");

  VarPro::Problem problem =
      std::filesystem::exists(pyfg_fpath)
          ? VarPro::parsePyfgTextToProblem(pyfg_fpath)
          : VarPro::parsePyfgTextToProblem("./bin/" + pyfg_fpath);

  std::vector<int> ranks = getRanksToSweep(problem.dim(), config.max_rank);
  std::vector<VarPro::Formulation> formulations = getFormulationsToSweep();
  auto init_file_names = makeInitializationFiles(dataset_path.string(), ranks);

  // now lets iterate over all of the different configurations
  for (size_t r_idx = 0; r_idx < ranks.size(); r_idx++)
  {
    // set the rank
    int r = ranks[r_idx];
    problem.setRank(r);
    for (VarPro::Formulation formulation : formulations)
    {
      // set the formulation
      problem.setFormulation(formulation);
      for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
      {
        std::string init_fpath = init_file_names[r_idx][init_idx];
        // if file doesn't exist, sample a random initialization instead from
        // problem and write to file
        if (!std::filesystem::exists(init_fpath))
        {
          VarPro::Matrix random_init = problem.getRandomInitialGuess();
          writeInitializationFile(init_fpath, problem, random_init);
        }

        VarPro::Matrix init = readInitializationFile(init_fpath, problem);
        VarPro::ProblemResult result = VarPro::solveProblem(problem, init, true);
      }
    }
  }
}

int main(int argc, char **argv)
{
  Config config = parseConfig("/home/alan/variable-projection/examples/config.json");
  sweepDataset(config.abs_data_path);
}
