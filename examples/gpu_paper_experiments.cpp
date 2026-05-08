#include <VarPro/Solver.h>
#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Utils.h>
#include <VarPro/Symbol.h>
#include <VarPro/PyfgTextParser.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/GpuRTRSolver.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#endif

#include <filesystem>
#include <limits>
#include <set>
#include <vector>

#include <json.hpp>
#include <experiment_utils.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Config
{
  bool verbose;
  std::string abs_data_path;
  int min_rank;
  int max_rank;
  int num_inits;
  double scale_reg_weight = 1e-2;
};

struct ExperimentResult
{
  std::string dataset_name;
  std::string init_file;
  std::vector<VarPro::Scalar> costs;
  std::vector<double> times;
  VarPro::Formulation formulation;
  std::string backend;
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
           {"formulation", r.formulation},
           {"backend", r.backend}};
}

template <typename Scalar>
std::vector<Scalar> parseNumericVectorAllowNulls(const json &j,
                                                 const std::string &field_name)
{
  static_assert(std::numeric_limits<Scalar>::has_quiet_NaN,
                "Cached experiment results require a NaN sentinel for null values.");

  const json &field = j.at(field_name);
  if (!field.is_array())
  {
    throw std::runtime_error("Field '" + field_name + "' must be an array.");
  }

  std::vector<Scalar> values;
  values.reserve(field.size());
  for (size_t idx = 0; idx < field.size(); ++idx)
  {
    const json &entry = field.at(idx);
    if (entry.is_null())
    {
      values.push_back(std::numeric_limits<Scalar>::quiet_NaN());
      continue;
    }
    if (!entry.is_number())
    {
      throw std::runtime_error("Field '" + field_name + "' index " +
                               std::to_string(idx) +
                               " must be a number or null, but is " +
                               std::string(entry.type_name()) + ".");
    }
    values.push_back(entry.get<Scalar>());
  }

  return values;
}

void from_json(const json &j, ExperimentResult &r)
{
  j.at("dataset_name").get_to(r.dataset_name);
  j.at("init_file").get_to(r.init_file);
  r.costs = parseNumericVectorAllowNulls<VarPro::Scalar>(j, "costs");
  r.times = parseNumericVectorAllowNulls<double>(j, "times");
  j.at("formulation").get_to(r.formulation);
  if (j.contains("backend")) j.at("backend").get_to(r.backend);
}

Config parseConfig(const std::string &filename)
{
  if (!std::filesystem::exists(filename))
    throw std::runtime_error("Config file does not exist: " + filename);

  std::ifstream file(filename);
  json j;
  file >> j;

  Config config;
  config.verbose = j["verbose"];
  config.min_rank = j["min_rank"];
  config.max_rank = j["max_rank"];
  config.abs_data_path = j["abs_data_path"];
  config.num_inits = j["num_inits"];
  if (j.contains("scale_reg_weight"))
    config.scale_reg_weight = j["scale_reg_weight"];
  return config;
}

std::vector<int> getRanksToSweep(int min_rank, int max_rank)
{
  std::vector<int> ranks;
  for (int r = min_rank; r <= max_rank; r++)
    ranks.push_back(r);
  return ranks;
}

std::vector<VarPro::Formulation> getFormulationsToSweep()
{
  return {
      VarPro::Formulation::Explicit,
      VarPro::Formulation::ExplicitVarPro,
      VarPro::Formulation::Implicit};
}

std::vector<std::vector<std::string>> makeInitializationFiles(
    const std::string &dataset_path, const std::vector<int> &ranks, int num_inits)
{
  std::vector<std::vector<std::string>> init_file_paths;
  for (int r : ranks)
  {
    std::vector<std::string> rank_inits;
    for (int i = 1; i <= num_inits; i++)
      rank_inits.push_back(dataset_path + "/inits/rank" + std::to_string(r) +
                           "_init" + std::to_string(i) + ".txt");
    init_file_paths.push_back(rank_inits);
  }
  return init_file_paths;
}

std::string formName(VarPro::Formulation f)
{
  switch (f) {
    case VarPro::Formulation::Explicit: return "Explicit";
    case VarPro::Formulation::ExplicitVarPro: return "ExplicitVarPro";
    case VarPro::Formulation::Implicit: return "Implicit";
    default: return "Unknown";
  }
}

std::string getIntermediateResultsFilePath(const fs::path &dataset_path,
                                           int rank,
                                           VarPro::Formulation formulation,
                                           int init_idx)
{
  return (dataset_path / "cached_results" /
      ("gpu_results_rank" + std::to_string(rank) + "_" +
       formName(formulation) + "_init" + std::to_string(init_idx + 1) +
       ".json")).string();
}

#ifdef VARPRO_HAVE_CUDA

ExperimentResult runGpuSolve(
    const std::string &dataset_name,
    const std::string &init_fpath,
    VarPro::Problem &problem,
    const VarPro::Matrix &init,
    bool verbose)
{
  ExperimentResult result;
  result.dataset_name = dataset_name;
  result.init_file = init_fpath;
  result.formulation = problem.getFormulation();
  result.backend = "GpuRTR";

  try
  {
    VarProGPU::GpuContext ctx;
    VarProGPU::RTRParams params;
    params.max_outer_iters = 250;
    params.verbose = verbose;
    VarProGPU::GpuRTRSolver solver(ctx);

    VarProGPU::RTRResult gpu_result;

    if (problem.getFormulation() == VarPro::Formulation::Implicit)
    {
      // Implicit: use matrix-free Schur complement operator
      auto pre = VarProGPU::buildPrecomputeResult(problem);
      VarProGPU::GpuSchurOperator gpu_op(pre, ctx);
      gpu_result = solver.solve(problem, gpu_op, init, params);
    }
    else
    {
      // Explicit / ExplicitVarPro: single SpMM with full data_matrix
      VarProGPU::GpuExplicitOperator gpu_op(problem, ctx);
      gpu_result = solver.solveExplicit(problem, gpu_op, init, params);
    }

    result.costs = gpu_result.objective_values;
    result.times = gpu_result.elapsed_times;
  }
  catch (const std::runtime_error &e)
  {
    result.costs = {};
    result.times = {};
    std::cerr << "Error in GPU solve: " << e.what() << std::endl;
  }

  return result;
}

void sweepDataset(fs::path dataset_path, std::vector<ExperimentResult> &all_results,
                  const Config &config)
{
  // Check for cached results
  std::string results_file = (dataset_path / "gpu_results.json").string();
  if (std::filesystem::exists(results_file))
  {
    std::cout << "GPU results already exist in " << dataset_path
              << ". Skipping." << std::endl;
    std::ifstream file(results_file);
    json j;
    file >> j;
    auto existing = j.get<std::vector<ExperimentResult>>();
    all_results.insert(all_results.end(), existing.begin(), existing.end());
    return;
  }

  std::string pyfg_fpath = findPyfgInDir(dataset_path).string();
  VarPro::Problem problem = VarPro::parsePyfgTextToProblem(pyfg_fpath);
  if (problem.isSfmProblem())
  {
    problem.convertToScaledStiefel();
    problem.setScaleRegWeight(static_cast<VarPro::Scalar>(config.scale_reg_weight));
  }
  problem.updateProblemData();

  std::vector<int> ranks = getRanksToSweep(config.min_rank, config.max_rank);
  std::vector<VarPro::Formulation> formulations = getFormulationsToSweep();
  auto init_file_names = makeInitializationFiles(dataset_path.string(), ranks, config.num_inits);
  std::vector<ExperimentResult> current_results;

  for (size_t r_idx = 0; r_idx < ranks.size(); r_idx++)
  {
    int r = ranks[r_idx];
    problem.setRank(r);

    for (VarPro::Formulation formulation : formulations)
    {
      problem.setFormulation(formulation);

      for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
      {
        std::string init_fpath = init_file_names[r_idx][init_idx];

        // Generate initialization if needed
        if (!std::filesystem::exists(init_fpath))
        {
          std::cout << "Initialization file " << init_fpath
                    << " does not exist. Writing random initialization." << std::endl;
          VarPro::Matrix random_init = problem.getRandomInitialGuess();
          writeInitializationFile(init_fpath, problem, random_init);
        }

        // Check cached intermediate result
        std::string cache_path = getIntermediateResultsFilePath(
            dataset_path, r, formulation, init_idx);
        fs::create_directories(fs::path(cache_path).parent_path());
        if (std::filesystem::exists(cache_path))
        {
          std::cout << "Cached: " << cache_path << ". Skipping." << std::endl;
          std::ifstream file(cache_path);
          json j;
          file >> j;
          auto cached = j.get<std::vector<ExperimentResult>>();
          current_results.insert(current_results.end(), cached.begin(), cached.end());
          continue;
        }

        VarPro::Matrix init = readInitializationFile(init_fpath, problem);
        checkMatrixShape("gpu_sweep::init",
                         problem.getExpectedVariableSize(), problem.getRelaxationRank(),
                         init.rows(), init.cols());

        std::cout << "GPU: " << dataset_path.filename().string()
                  << " " << formName(formulation)
                  << " rank=" << r << " init=" << init_idx + 1 << std::endl;

        auto exp_result = runGpuSolve(
            dataset_path.filename().string(), init_fpath, problem, init,
            config.verbose);
        current_results.push_back(exp_result);

        // Cache intermediate result
        json j = std::vector<ExperimentResult>{exp_result};
        std::ofstream file(cache_path);
        file << j << std::endl;
      }
    }
  }

  // Save all results for this dataset
  json j = current_results;
  std::ofstream file(results_file);
  file << j << std::endl;

  all_results.insert(all_results.end(), current_results.begin(), current_results.end());
}

#endif // VARPRO_HAVE_CUDA

int main(int argc, char **argv)
{
#ifndef VARPRO_HAVE_CUDA
  std::cerr << "GPU support not compiled. Build with -DENABLE_GPU=ON.\n";
  return 1;
#else
  std::string config_path = "/home/nikolas/variable-projection/examples/config.json";
  if (argc > 1)
    config_path = argv[1];

  Config config = parseConfig(config_path);

  std::vector<fs::path> experiment_dirs;
  getExperimentDirsRecursive(config.abs_data_path, experiment_dirs);

  // Sort by directory size (smallest first)
  std::sort(experiment_dirs.begin(), experiment_dirs.end(),
            [](const fs::path &a, const fs::path &b)
            { return dir_size(a) < dir_size(b); });

  std::vector<ExperimentResult> all_results;
  for (const auto &dir : experiment_dirs)
  {
    std::cout << "GPU sweep: " << dir << std::endl;
    try
    {
      sweepDataset(dir, all_results, config);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error in " << dir << ": " << e.what() << std::endl;
    }
  }

  // Save aggregated results
  json j = all_results;
  std::ofstream file(config.abs_data_path + "/gpu_experiment_results.json");
  file << j << std::endl;
  std::cout << "Saved " << all_results.size() << " results to "
            << config.abs_data_path + "/gpu_experiment_results.json" << std::endl;
  return 0;
#endif
}
