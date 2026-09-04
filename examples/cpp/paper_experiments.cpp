#include <VarPro/Solver.h>
#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Utils.h>
#include <VarPro/Symbol.h>
#include <VarPro/PyfgTextParser.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <sstream>
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
  int min_rank;
  int max_rank;
  int num_inits;
  double scale_reg_weight = 1e-2;
  // Which formulations to sweep. Empty means "the default three" -- Dense is
  // opt-in because forming the reduced system explicitly costs p^2 doubles.
  std::vector<std::string> formulations;
  // Dataset directories whose path contains any of these substrings are
  // excluded from the recursive sweep. Dataset discovery is "any leaf dir
  // holding a .pyfg", which otherwise picks up the robust-experiment families
  // (cosmobench, nebula) that have no inits/ and are not part of the standard
  // benchmark.
  std::vector<std::string> skip_substrings;
  // SfM problems can be solved on the scaled-Stiefel manifold
  // (R_{>0} x St(p,k))^n with a log-barrier on the per-pose scales, or on the
  // plain Stiefel manifold. Both are supported; this selects which.
  bool sfm_use_scaled_stiefel = true;
  // Refuse to form a dense Schur complement larger than this. The run is
  // still recorded, with skip_reason set, so the table shows an honest "--"
  // rather than the dataset silently vanishing (or the sweep being OOM-killed).
  double max_dense_gb = 8.0;
};

struct ExperimentResult
{
  std::string dataset_name;
  std::string init_file;
  std::vector<VarPro::Scalar> costs;
  std::vector<double> times;
  VarPro::Formulation formulation;
  // Seconds of one-time precompute attributable to this formulation: the
  // implicit precompute (B, and the Cholesky factor of M) for the
  // marginalized formulations, plus the explicit Schur formation for Dense.
  double precompute_s = 0.0;
  // Size of the explicit reduced system, in GB. Reported for Dense even when
  // the run was skipped for being too large; 0 for the other formulations.
  double dense_gb = 0.0;
  // Empty on a normal run; otherwise why no solve was attempted.
  std::string skip_reason;
};

NLOHMANN_JSON_SERIALIZE_ENUM(VarPro::Formulation,
                             {{VarPro::Formulation::Explicit, "Explicit"},
                              {VarPro::Formulation::ExplicitVarPro, "ExplicitVarPro"},
                              {VarPro::Formulation::Implicit, "Implicit"},
                              {VarPro::Formulation::Dense, "Dense"}});

void to_json(json &j, const ExperimentResult &r)
{
  j = json{{"dataset_name", r.dataset_name},
           {"init_file", r.init_file},
           {"costs", r.costs},
           {"times", r.times},
           {"formulation", r.formulation},
           {"precompute_s", r.precompute_s},
           {"dense_gb", r.dense_gb},
           {"skip_reason", r.skip_reason}};
}

void from_json(const json &j, ExperimentResult &r)
{
  j.at("dataset_name").get_to(r.dataset_name);
  j.at("init_file").get_to(r.init_file);
  j.at("costs").get_to(r.costs);
  j.at("times").get_to(r.times);
  j.at("formulation").get_to(r.formulation);
  // Optional: results written before the Dense formulation existed lack these.
  r.precompute_s = j.value("precompute_s", 0.0);
  r.dense_gb = j.value("dense_gb", 0.0);
  r.skip_reason = j.value("skip_reason", std::string{});
}

std::string formulationName(VarPro::Formulation f)
{
  switch (f)
  {
  case VarPro::Formulation::Explicit:
    return "Explicit";
  case VarPro::Formulation::ExplicitVarPro:
    return "ExplicitVarPro";
  case VarPro::Formulation::Implicit:
    return "Implicit";
  case VarPro::Formulation::Dense:
    return "Dense";
  }
  return "Unknown";
}

VarPro::Formulation formulationFromName(const std::string &name)
{
  if (name == "Explicit")
    return VarPro::Formulation::Explicit;
  if (name == "ExplicitVarPro")
    return VarPro::Formulation::ExplicitVarPro;
  if (name == "Implicit")
    return VarPro::Formulation::Implicit;
  if (name == "Dense")
    return VarPro::Formulation::Dense;
  throw std::runtime_error("Unknown formulation name: '" + name +
                           "' (expected Explicit|ExplicitVarPro|Implicit|Dense)");
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
  config.min_rank = j["min_rank"];
  config.max_rank = j["max_rank"];
  config.abs_data_path = j["abs_data_path"];
  config.num_inits = j["num_inits"];
  if (j.contains("scale_reg_weight"))
    config.scale_reg_weight = j["scale_reg_weight"];
  if (j.contains("formulations"))
    config.formulations = j["formulations"].get<std::vector<std::string>>();
  if (j.contains("max_dense_gb"))
    config.max_dense_gb = j["max_dense_gb"];
  if (j.contains("skip_substrings"))
    config.skip_substrings = j["skip_substrings"].get<std::vector<std::string>>();
  if (j.contains("sfm_use_scaled_stiefel"))
    config.sfm_use_scaled_stiefel = j["sfm_use_scaled_stiefel"];

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

/**
 * Which formulations to sweep, in priority order:
 *   1. the VARPRO_FORMULATION env var (comma-separated), which isolates one
 *      formulation per process -- this is how the memory-footprint numbers are
 *      measured, since peak RSS is a per-process quantity;
 *   2. the "formulations" key in the config;
 *   3. the default three.
 *
 * Dense is deliberately absent from the default: it forms the reduced system
 * explicitly (p^2 doubles), which is 28 GB on the larger range-aided datasets.
 * It is a baseline you ask for, not one you get by accident.
 */
std::vector<VarPro::Formulation> getFormulationsToSweep(const Config &config)
{
  auto parseList = [](const std::string &csv)
  {
    std::vector<VarPro::Formulation> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
      // trim
      const auto b = tok.find_first_not_of(" \t");
      const auto e = tok.find_last_not_of(" \t");
      if (b == std::string::npos)
        continue;
      out.push_back(formulationFromName(tok.substr(b, e - b + 1)));
    }
    return out;
  };

  if (const char *env = std::getenv("VARPRO_FORMULATION"))
  {
    auto forms = parseList(env);
    if (!forms.empty())
      return forms;
  }

  if (!config.formulations.empty())
  {
    std::vector<VarPro::Formulation> out;
    for (const auto &name : config.formulations)
      out.push_back(formulationFromName(name));
    return out;
  }

  return {
      VarPro::Formulation::Explicit,
      VarPro::Formulation::ExplicitVarPro,
      VarPro::Formulation::Implicit};
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
                               VarPro::Formulation formulation,
                               const VarPro::Problem &problem)
{
  ExperimentResult exp_result;
  exp_result.dataset_name = dataset_name;
  exp_result.init_file = init_file;
  exp_result.formulation = formulation;

  // Per-formulation one-time precompute. Explicit/ExplicitVarPro use the raw
  // data matrix and pay none of it; the marginalized formulations pay for B
  // and the Cholesky factor of M; Dense additionally pays to form Q_sc.
  if (VarPro::isMarginalized(formulation))
  {
    exp_result.precompute_s = problem.getImplicitPrecomputeTimeS();
    if (formulation == VarPro::Formulation::Dense)
    {
      exp_result.precompute_s += problem.getDensePrecomputeTimeS();
      exp_result.dense_gb = problem.densePrecomputeGB();
    }
  }

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

  description += "Formulation: " + formulationName(problem.getFormulation()) + ". ";
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

std::string getIntermediateResultsFilePath(const fs::path &dataset_path,
                                           int rank,
                                           VarPro::Formulation formulation,
                                            int init_idx)
{
  // NB: this must name every formulation distinctly. It used to be a nested
  // ternary whose fallback arm was "Implicit", so any formulation added later
  // would silently share -- and reload -- the Implicit cache files.
  return dataset_path.string() + "/cached_results/results_rank" + std::to_string(rank) +
         "_" + formulationName(formulation) +
          "_init" + std::to_string(init_idx + 1) +
         ".json";
}

/**
 * @brief Takes as input the directory that contains a .pyfg file and many different
 * initializations (e.g., rank3_init10.txt, rank4_init10.txt, etc.)
 *
 * @param dataset_path the path to the dataset directory
 */
void sweepDataset(fs::path dataset_path, std::vector<ExperimentResult> &all_results,
                  const Config &config)
{
  const bool verbose = config.verbose;
  std::vector<VarPro::Formulation> formulations = getFormulationsToSweep(config);

  // If there is already a results.json in the directory, reuse it -- but only
  // if it actually covers every formulation we were asked for. Otherwise a
  // pre-existing results.json (written before Dense was requested) would
  // permanently mask the new formulation. Re-running is cheap: the
  // per-experiment intermediate caches below still short-circuit the
  // formulations that were already done.
  if (std::filesystem::exists(dataset_path / "results.json"))
  {
    auto existing_results = loadResultsFromFile((dataset_path / "results.json").string());
    std::set<std::string> have;
    for (const auto &r : existing_results)
      have.insert(formulationName(r.formulation));
    std::vector<std::string> missing;
    for (auto f : formulations)
      if (!have.count(formulationName(f)))
        missing.push_back(formulationName(f));

    if (missing.empty())
    {
      std::cout << "Results file already exists in directory " << dataset_path
                << ". Skipping sweep." << std::endl;
      all_results.insert(all_results.end(), existing_results.begin(), existing_results.end());
      return;
    }
    std::cout << "Results file in " << dataset_path << " is missing formulation(s):";
    for (const auto &m : missing)
      std::cout << " " << m;
    std::cout << " -- re-running sweep." << std::endl;
  }

  // find the .pyfg file in the directory
  std::string pyfg_fpath = findPyfgInDir(dataset_path).string();

  VarPro::Problem problem =
      std::filesystem::exists(pyfg_fpath)
          ? VarPro::parsePyfgTextToProblem(pyfg_fpath)
          : VarPro::parsePyfgTextToProblem("./bin/" + pyfg_fpath);
  // SfM uses the scaled-Stiefel manifold with a log-barrier on the per-pose
  // scales. gpu_paper_experiments and optimizer_diagnostics already do this;
  // the CPU driver was missing the branch, so for SfM datasets it was solving
  // a different problem than the GPU (plain Stiefel, no scale regularization).
  if (problem.isSfmProblem() && config.sfm_use_scaled_stiefel)
  {
    problem.convertToScaledStiefel();
    problem.setScaleRegWeight(static_cast<VarPro::Scalar>(config.scale_reg_weight));
  }
  problem.updateProblemData();

  std::vector<int> ranks = getRanksToSweep(config.min_rank, config.max_rank);
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
      // Guard the Dense formulation *before* setFormulation(), which is what
      // actually allocates the p x p reduced system. Record the run with a
      // skip_reason and the size we declined to allocate, so the results table
      // can print an honest "--" instead of the row disappearing (or the whole
      // sweep being OOM-killed partway through).
      if (formulation == VarPro::Formulation::Dense &&
          problem.densePrecomputeGB() > config.max_dense_gb)
      {
        std::cout << "Skipping Dense on " << dataset_path.filename().string()
                  << " rank " << r << ": reduced system is "
                  << problem.densePrecomputeGB() << " GB > max_dense_gb ("
                  << config.max_dense_gb << " GB)." << std::endl;
        for (size_t init_idx = 0; init_idx < init_file_names[r_idx].size(); init_idx++)
        {
          ExperimentResult skipped;
          skipped.dataset_name = dataset_path.filename().string();
          skipped.init_file = init_file_names[r_idx][init_idx];
          skipped.formulation = formulation;
          skipped.dense_gb = problem.densePrecomputeGB();
          skipped.skip_reason = "dense_too_large";
          current_results.push_back(skipped);
        }
        continue;
      }

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

        // have an intermediate results file that saves just the results for
        // this experiment. If it already exists, load it and skip the
        // experiment. Make the parent directory if it doesn't exist.
        std::string intermediate_results_fpath = getIntermediateResultsFilePath(dataset_path, r, formulation, init_idx);
        fs::create_directories(fs::path(intermediate_results_fpath).parent_path());
        if (std::filesystem::exists(intermediate_results_fpath))
        {
          std::cout << "Intermediate results file " << intermediate_results_fpath
                    << " already exists. Loading existing results and skipping experiment."
                    << std::endl;
          auto intermediate_results = loadResultsFromFile(intermediate_results_fpath);
          current_results.insert(current_results.end(), intermediate_results.begin(),
                                 intermediate_results.end());
          continue;
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
          result = VarPro::solveProblem(problem, init, verbose);
        }
        catch (const std::runtime_error &e)
        {
          result.time = {};
          result.objective_values = {};
          std::cout << "Error solving problem: " << e.what() << std::endl;
        }

        ExperimentResult exp_result = compileResult(dataset_path.filename().string(),
                                                    init_fpath, result, formulation,
                                                    problem);

        // write the intermediate results to file
        json j = std::vector<ExperimentResult>{exp_result};
        std::ofstream file(intermediate_results_fpath);
        file << j << std::endl;
        std::cout << "Wrote intermediate results to file " << intermediate_results_fpath << std::endl;

        // add the result to the current_results vector
        current_results.push_back(exp_result);
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
  // Accept a config path on argv, matching gpu_paper_experiments. This is what
  // lets a sweep script point the driver at a scratch config (e.g. one that
  // enables the Dense formulation, or raises max_dense_gb) without editing the
  // checked-in examples/config.json.
  std::string config_path = "/home/nikolas/variable-projection/examples/config.json";
  if (argc > 1)
    config_path = argv[1];

  Config config = parseConfig(config_path);

  {
    std::cout << "Sweeping formulations:";
    for (auto f : getFormulationsToSweep(config))
      std::cout << " " << formulationName(f);
    std::cout << " (max_dense_gb=" << config.max_dense_gb
              << ", sfm_manifold="
              << (config.sfm_use_scaled_stiefel ? "ScaledStiefel" : "Stiefel")
              << ")" << std::endl;
  }

  std::vector<fs::path> experiment_dirs = {};
  getExperimentDirsRecursive(config.abs_data_path, experiment_dirs);

  if (!config.skip_substrings.empty())
  {
    const size_t before = experiment_dirs.size();
    experiment_dirs.erase(
        std::remove_if(experiment_dirs.begin(), experiment_dirs.end(),
                       [&](const fs::path &d)
                       {
                         const std::string s = d.string();
                         for (const auto &skip : config.skip_substrings)
                           if (s.find(skip) != std::string::npos)
                             return true;
                         return false;
                       }),
        experiment_dirs.end());
    std::cout << "Skipped " << (before - experiment_dirs.size())
              << " dataset dir(s) matching skip_substrings; "
              << experiment_dirs.size() << " remain." << std::endl;
  }

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
      sweepDataset(dir, all_results, config);
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
