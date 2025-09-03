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

struct ExperimentResult
{
  std::string dataset_name;
  std::string init_file;
  std::vector<VarPro::Scalar> costs;
  std::vector<double> times;
  VarPro::Formulation formulation;
};

NLOHMANN_JSON_SERIALIZE_ENUM(VarPro::Formulation,
                             {{VarPro::Formulation::Explicit, "Explicit"},
                              {VarPro::Formulation::ExplicitVarPro, "ExplicitVarPro"},
                              {VarPro::Formulation::Implicit, "Implicit"}});

void to_json(json &j, const ExperimentResult &r)
{
  j = json{{"dataset_name", r.dataset_name},
           {"init_file", r.init_file},
           {"costs", r.costs},
           {"times", r.times},
           {"formulation", r.formulation}};
}

void from_json(const json &j, ExperimentResult &r)
{
  j.at("dataset_name").get_to(r.dataset_name);
  j.at("init_file").get_to(r.init_file);
  j.at("costs").get_to(r.costs);
  j.at("times").get_to(r.times);
  j.at("formulation").get_to(r.formulation);
}

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
      rank_init_file_paths.push_back(dataset_path + "/inits/rank" + std::to_string(r) +
                                     "_init" + std::to_string(i) + ".txt");
    }
    init_file_paths.push_back(rank_init_file_paths);
  }
  return init_file_paths;
}

ExperimentResult compileResult(const std::string &dataset_name,
                               const std::string &init_file,
                               const VarPro::ProblemResult &result,
                               VarPro::Formulation formulation)
{
  ExperimentResult exp_result;
  exp_result.dataset_name = dataset_name;
  exp_result.init_file = init_file;
  exp_result.costs = result.objective_values;
  exp_result.times = result.time;
  exp_result.formulation = formulation;
  return exp_result;
}

/**
 * @brief Takes as input the directory that contains a .pyfg file and many different
 * initializations (e.g., rank3_init10.txt, rank4_init10.txt, etc.)
 *
 * @param dataset_path the path to the dataset directory
 */
void sweepDataset(fs::path dataset_path, std::vector<ExperimentResult> &all_results, bool verbose = false)
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
  std::vector<ExperimentResult> current_results = {};

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
      problem.updateProblemData();
      for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
      {
        std::string init_fpath = init_file_names[r_idx][init_idx];
        // if file doesn't exist, sample a random initialization instead from
        // problem and write to file
        if (!std::filesystem::exists(init_fpath))
        {
          std::cout << "Initialization file " << init_fpath
                    << " does not exist. Sampling random initialization instead."
                    << std::endl;
          VarPro::Matrix random_init = problem.getRandomInitialGuess();
          writeInitializationFile(init_fpath, problem, random_init);
        }

        VarPro::Matrix init = readInitializationFile(init_fpath, problem);
        VarPro::ProblemResult result = VarPro::solveProblem(problem, init, verbose);
        current_results.push_back(compileResult(dataset_path.filename().string(),
                                                init_fpath, result, formulation));
      }
    }
  }

  // save the results to a json file in the experiment directory
  json j = current_results;
  std::ofstream file(dataset_path.string() + "/results.json");
  file << j << std::endl;

  // append the results to the all_results vector
  all_results.insert(all_results.end(), current_results.begin(), current_results.end());
}

int main(int argc, char **argv)
{
  Config config = parseConfig("/home/alan/variable-projection/examples/config.json");
  std::vector<fs::path> experiment_dirs = {};
  getExperimentDirsRecursive(config.abs_data_path, experiment_dirs);
  std::vector<ExperimentResult> all_results = {};
  for (const auto &dir : experiment_dirs)
  {
    std::cout << "Sweeping dataset in directory: " << dir << std::endl;
    sweepDataset(dir, all_results, config.verbose);
  }
  json j = all_results;
  // the output file should be the same directory as the config file abs_data_path
  std::ofstream file(config.abs_data_path + "/experiment_results.json");
  file << j << std::endl;
}
