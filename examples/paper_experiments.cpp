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
  int num_inits;
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
  config.num_inits = j["num_inits"];

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
  return {
      VarPro::Formulation::Explicit,
      VarPro::Formulation::ExplicitVarPro,
      VarPro::Formulation::Implicit
    };
}

std::vector<std::vector<std::string>> makeInitializationFiles(const std::string &dataset_path,
                                                              const std::vector<int> &ranks,
                                                              int num_inits)
{
  // start by making a 2d array to hold all of the initialization file paths
  // the array should have len(ranks) rows and 10 columns
  std::vector<std::vector<std::string>> init_file_paths = {};
  for (int r : ranks)
  {
    std::vector<std::string> rank_init_file_paths;
    for (int i = 1; i <= num_inits; i++)
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
  exp_result.formulation = formulation;

  // if result is empty (uninitialized), set the costs and times to empty vectors
  if (result.objective_values.size() == 0 || result.time.size() == 0)
  {
    exp_result.costs = {};
    exp_result.times = {};
  }
  else
  {
    exp_result.costs = result.objective_values;
    exp_result.times = result.time;
  }

  return exp_result;
}

std::string getExpDescription(fs::path pyfg_fpath, const VarPro::Problem &problem)
{
  std::string exp_name = pyfg_fpath.stem().string();
  std::string description = "Experiment: " + exp_name + ". ";

  description += "Formulation: ";
  switch (problem.getFormulation())
  {
  case VarPro::Formulation::Explicit:
    description += "Explicit. ";
    break;
  case VarPro::Formulation::ExplicitVarPro:
    description += "ExplicitVarPro. ";
    break;
  case VarPro::Formulation::Implicit:
    description += "Implicit. ";
    break;
  default:
    description += "Unknown. ";
    break;
  }
  description += "Relaxation rank: " + std::to_string(problem.getRelaxationRank()) + ".";
  return description;
}

std::vector<ExperimentResult> loadResultsFromFile(const std::string &filename)
{
  // check if the file exists
  if (!std::filesystem::exists(filename))
  {
    throw std::runtime_error("Results file does not exist");
  }

  std::ifstream file(filename);
  json j;
  file >> j;

  std::vector<ExperimentResult> results = j.get<std::vector<ExperimentResult>>();
  return results;
}

/**
 * @brief Takes as input the directory that contains a .pyfg file and many different
 * initializations (e.g., rank3_init10.txt, rank4_init10.txt, etc.)
 *
 * @param dataset_path the path to the dataset directory
 */
void sweepDataset(fs::path dataset_path, std::vector<ExperimentResult> &all_results, bool verbose = false)
{
  // if there is already a results.json file in the directory, load the existing
  // (cached) results and do not run the sweep again
  if (std::filesystem::exists(dataset_path / "results.json"))
  {
    std::cout << "Results file already exists in directory " << dataset_path
              << ". Skipping sweep." << std::endl;
    auto existing_results = loadResultsFromFile((dataset_path / "results.json").string());
    all_results.insert(all_results.end(), existing_results.begin(), existing_results.end());
    return;
  }

  // find the .pyfg file in the directory
  std::string pyfg_fpath = findPyfgInDir(dataset_path).string();
  Config config = parseConfig("/home/alan/variable-projection/examples/config.json");

  VarPro::Problem problem =
      std::filesystem::exists(pyfg_fpath)
          ? VarPro::parsePyfgTextToProblem(pyfg_fpath)
          : VarPro::parsePyfgTextToProblem("./bin/" + pyfg_fpath);
  problem.updateProblemData();

  std::vector<int> ranks = getRanksToSweep(problem.dim(), config.max_rank);
  std::vector<VarPro::Formulation> formulations = getFormulationsToSweep();
  auto init_file_names = makeInitializationFiles(dataset_path.string(), ranks, config.num_inits);
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
      for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
      {
        std::string init_fpath = init_file_names[r_idx][init_idx];
        // if file doesn't exist, sample a random initialization instead from
        // problem and write to file
        if (!std::filesystem::exists(init_fpath))
        {
          std::cout << "Initialization file " << init_fpath
                    << " does not exist. Writing a random initialization."
                    << std::endl;
          VarPro::Matrix random_init = problem.getRandomInitialGuess();
          writeInitializationFile(init_fpath, problem, random_init);
        }

        VarPro::Matrix init = readInitializationFile(init_fpath, problem);
        checkMatrixShape("sweepDataset::init",
                         problem.getExpectedVariableSize(), problem.getRelaxationRank(),
                         init.rows(), init.cols());
        std::cout << "Running " << getExpDescription(findPyfgInDir(dataset_path), problem)
                  << " on initialization file " << init_fpath << std::endl;
        VarPro::ProblemResult result = {};
        try
        {
          throw std::runtime_error("Skipping solveProblem");
          result = VarPro::solveProblem(problem, init, verbose);
        }
        catch (const std::runtime_error &e)
        {
          result.time = {};
          result.objective_values = {};
          std::cout << "Error solving problem: " << e.what() << std::endl;
        }
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

  // sort experiment_dirs based on the size of the directory (smallest to largest)
  std::sort(experiment_dirs.begin(), experiment_dirs.end(),
            [](const fs::path &a, const fs::path &b)
            {
              return dir_size(a) < dir_size(b);
            });

  std::vector<ExperimentResult> all_results = {};
  for (const auto &dir : experiment_dirs)
  {
    std::cout << "Sweeping dataset in directory: " << dir << std::endl;
    try
    {
      sweepDataset(dir, all_results, config.verbose);
    }
    catch (const std::invalid_argument &e)
    {
      std::cerr << "Error sweeping dataset in directory " << dir << ": " << e.what() << std::endl;
    }
  }
  json j = all_results;
  // the output file should be the same directory as the config file abs_data_path
  std::ofstream file(config.abs_data_path + "/experiment_results.json");
  file << j << std::endl;
}
