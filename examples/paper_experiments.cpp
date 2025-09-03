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

  // end timer
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "VARPRO took " << elapsed.count() << " seconds" << std::endl;

  std::cout << "Experiment result, name: " << pyfg_fpath
            << ", time: " << elapsed.count()
            << " seconds, cost: " << soln.first.f << ", marginalized: "
            << (formulation == VarPro::Formulation::Implicit)
            << ", init rank jump: " << init_rank_jump
            << ", init random: " << (init_type == InitType::Random)
            << std::endl;

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
//   "time": 53.784879541
// }

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

  VarPro::Problem problem =
      std::filesystem::exists(pyfg_fpath)
          ? VarPro::parsePyfgTextToProblem(pyfg_fpath)
          : VarPro::parsePyfgTextToProblem("./bin/" + pyfg_fpath);

  // we want to iterate over rank values from problem.dim() to 7
  int rank_start = problem.dim();
  int rank_end = 7;
  auto ranks = std::vector<int>();
  for (int r = rank_start; r <= rank_end; r++)
  {
    ranks.push_back(r);
  }

  // we want to try all three different formulation types: Explicit, ExplicitVarPro, Implicit
  std::vector<VarPro::Formulation> formulations = {
      VarPro::Formulation::Explicit,
      VarPro::Formulation::ExplicitVarPro,
      VarPro::Formulation::Implicit};

  // init file names = rank{r}_init{i}.txt for r in ranks and i in [1, 10]
  // make a 2d array of strings
  std::vector<std::vector<std::string>> init_file_names = {}
  for (int r : ranks)
  {
    std::vector<std::string> rank_init_file_names = {};
    for (int i = 1; i <= 10; i++)
    {
      rank_init_file_names.push_back("rank" + std::to_string(r) + "_init" + std::to_string(i) + ".txt");
    }
    init_file_names.push_back(rank_init_file_names);
  }

  // now lets iterate over all of the different configurations
  for (size_t r_idx = 0; r_idx < ranks.size(); r_idx++)
  {
    int r = ranks[r_idx];
    for (VarPro::Formulation formulation : formulations)
    {
      for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
      {
        std::string init_file_name = init_file_names[r_idx][init_idx];
        std::string init_fpath = dataset_path.string() + "/" + init_file_name;
        // if file doesn't exist, sample a random initialization instead from
        // problem and write to file
        if (!std::filesystem::exists(init_fpath))
        {
          VarPro::Matrix random_init = problem.getRandomInitialGuess();
          std::ofstream init_file(init_fpath);
          init_file << random_init << std::endl;
          init_file.close();
          std::cout << "Wrote random initialization to " << init_fpath << std::endl;
        }

}

int main(int argc, char **argv)
{
  std::vector<std::string> original_exp_files = {
      "data/plaza1.pyfg", "data/plaza2.pyfg", "data/single_drone.pyfg",
      "data/tiers.pyfg"}; // TIERS faster w/ random init

  auto mrclam_range_and_rpm_files = getRangeAndRpmMrclamFiles();

  std::vector<std::string> files = {};

  Config config = parseConfig("/home/alan/variable-projection/examples/config.json");

  for (auto file : files)
  {
    VarPro::Matrix soln = solveProblem(
        file, config.init_rank_jump, config.max_rank, config.preconditioner,
        config.formulation, config.init_type, config.verbose,
        config.log_iterates, config.show_iterates);
    std::cout << std::endl;
  }
}
