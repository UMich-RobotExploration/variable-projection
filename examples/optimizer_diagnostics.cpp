#include <VarPro/Solver.h>
#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>

#include "experiment_utils.hpp"

#include <Optimization/Riemannian/TNT.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

struct Options
{
  fs::path dataset_dir;
  fs::path init_file;
  fs::path output_csv;
  VarPro::Formulation formulation{VarPro::Formulation::Implicit};
  double scale_reg_weight{0.01};
  bool verbose{false};
};

struct TraceRow
{
  size_t iteration{0};
  double elapsed_time_s{0.0};
  double objective{0.0};
  double grad_norm{0.0};
  double trust_region_radius{0.0};
  size_t inner_iterations{0};
  double step_norm{0.0};
  double actual_decrease{0.0};
  double gain_ratio{0.0};
  bool accepted{false};
};

std::string formulationToString(VarPro::Formulation formulation)
{
  switch (formulation)
  {
  case VarPro::Formulation::Explicit:
    return "Explicit";
  case VarPro::Formulation::ExplicitVarPro:
    return "ExplicitVarPro";
  case VarPro::Formulation::Implicit:
    return "Implicit";
  }
  throw std::runtime_error("Unknown formulation");
}

VarPro::Formulation parseFormulation(const std::string &value)
{
  std::string normalized = value;
  for (char &c : normalized)
  {
    if (c == '-' || c == '_')
      c = ' ';
    else
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  if (normalized == "explicit")
    return VarPro::Formulation::Explicit;
  if (normalized == "explicit varpro" || normalized == "explicit var proj" ||
      normalized == "varpro")
    return VarPro::Formulation::ExplicitVarPro;
  if (normalized == "implicit")
    return VarPro::Formulation::Implicit;

  throw std::invalid_argument("Unknown formulation: " + value);
}

int extractRankFromInitFile(const fs::path &init_file)
{
  static const std::regex rank_pattern(R"(rank(\d+))", std::regex::icase);
  std::smatch match;
  const std::string init_name = init_file.filename().string();
  if (!std::regex_search(init_name, match, rank_pattern))
  {
    throw std::invalid_argument("Could not determine rank from init file name: " +
                                init_name);
  }
  return std::stoi(match[1].str());
}

std::string manifoldFamily(const VarPro::Problem &problem)
{
  if (problem.isScaledStiefel())
    return "scaled_stiefel";
  if (problem.numRangeMeasurements() > 0)
    return "stiefel_plus_oblique";
  return "stiefel_only";
}

std::string statusToString(Optimization::Riemannian::TNTStatus status)
{
  using Optimization::Riemannian::TNTStatus;
  switch (status)
  {
  case TNTStatus::Gradient:
    return "gradient";
  case TNTStatus::PreconditionedGradient:
    return "preconditioned_gradient";
  case TNTStatus::RelativeDecrease:
    return "relative_decrease";
  case TNTStatus::Stepsize:
    return "stepsize";
  case TNTStatus::TrustRegion:
    return "trust_region";
  case TNTStatus::IterationLimit:
    return "iteration_limit";
  case TNTStatus::ElapsedTime:
    return "elapsed_time";
  case TNTStatus::UserFunction:
    return "user_function";
  }
  return "unknown";
}

Options parseArgs(int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &flag) -> std::string
    {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing value for " + flag);
      return argv[++i];
    };

    if (arg == "--dataset-dir")
      options.dataset_dir = requireValue(arg);
    else if (arg == "--init-file")
      options.init_file = requireValue(arg);
    else if (arg == "--output-csv")
      options.output_csv = requireValue(arg);
    else if (arg == "--formulation")
      options.formulation = parseFormulation(requireValue(arg));
    else if (arg == "--scale-reg-weight")
      options.scale_reg_weight = std::stod(requireValue(arg));
    else if (arg == "--verbose")
      options.verbose = true;
    else
      throw std::invalid_argument("Unknown argument: " + arg);
  }

  if (options.dataset_dir.empty() || options.init_file.empty() ||
      options.output_csv.empty())
  {
    throw std::invalid_argument(
        "Usage: optimizer_diagnostics --dataset-dir <dir> --init-file <file> "
        "--output-csv <path> [--formulation implicit|explicit|explicit_varpro] "
        "[--scale-reg-weight <value>] [--verbose]");
  }

  return options;
}

void writeTraceCsv(const fs::path &output_csv, const std::vector<TraceRow> &rows)
{
  fs::create_directories(output_csv.parent_path());
  std::ofstream out(output_csv);
  if (!out.is_open())
    throw std::runtime_error("Could not open output CSV: " + output_csv.string());

  out << "iteration,elapsed_time_s,objective,grad_norm,trust_region_radius,"
         "inner_iterations,step_norm,actual_decrease,gain_ratio,accepted\n";
  out << std::fixed << std::setprecision(12);
  for (const TraceRow &row : rows)
  {
    out << row.iteration << "," << row.elapsed_time_s << "," << row.objective
        << "," << row.grad_norm << "," << row.trust_region_radius << ","
        << row.inner_iterations << "," << row.step_norm << ","
        << row.actual_decrease << "," << row.gain_ratio << ","
        << (row.accepted ? 1 : 0) << "\n";
  }
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    const Options options = parseArgs(argc, argv);
    const fs::path pyfg_path = findPyfgInDir(options.dataset_dir);

    VarPro::Problem problem =
        VarPro::parsePyfgTextToProblem(pyfg_path.string());
    if (problem.isSfmProblem())
    {
      problem.convertToScaledStiefel();
      problem.setScaleRegWeight(
          static_cast<VarPro::Scalar>(options.scale_reg_weight));
    }

    problem.setRank(extractRankFromInitFile(options.init_file));
    problem.setFormulation(options.formulation);
    problem.updateProblemData();

    const VarPro::Matrix init = readInitializationFile(options.init_file, problem);

    using HessianOp = Optimization::Riemannian::LinearOperator<
        VarPro::Matrix, VarPro::Matrix, VarPro::Matrix>;

    std::vector<TraceRow> rows;
    VarPro::InstrumentationFunction instrumentation =
        [&](size_t iteration, double elapsed_time, const VarPro::Matrix &x,
            VarPro::Scalar fx, const VarPro::Matrix &grad, const HessianOp &HessOp,
            VarPro::Scalar Delta, size_t num_STPCG_iters,
            const VarPro::Matrix &h, VarPro::Scalar df, VarPro::Scalar rho,
            bool accepted, VarPro::Matrix &NablaF_Y) -> bool
    {
      (void)x;
      (void)HessOp;
      (void)NablaF_Y;
      rows.push_back(TraceRow{
          iteration,
          elapsed_time,
          fx,
          grad.norm(),
          Delta,
          num_STPCG_iters,
          h.norm(),
          df,
          rho,
          accepted,
      });
      return false;
    };

    const VarPro::ProblemResult result =
        VarPro::solveProblem(problem, init, instrumentation, options.verbose);

    writeTraceCsv(options.output_csv, rows);

    size_t accepted_steps = 0;
    for (const TraceRow &row : rows)
      accepted_steps += static_cast<size_t>(row.accepted);

    std::cout << "dataset_dir: " << options.dataset_dir << "\n";
    std::cout << "pyfg: " << pyfg_path << "\n";
    std::cout << "formulation: " << formulationToString(options.formulation) << "\n";
    std::cout << "manifold_family: " << manifoldFamily(problem) << "\n";
    std::cout << "num_range_measurements: " << problem.numRangeMeasurements() << "\n";
    std::cout << "is_scaled_stiefel: " << (problem.isScaledStiefel() ? "true" : "false") << "\n";
    std::cout << "trace_csv: " << options.output_csv << "\n";
    std::cout << "outer_iterations: " << rows.size() << "\n";
    std::cout << "accepted_steps: " << accepted_steps << "\n";
    std::cout << "final_status: " << statusToString(result.status) << "\n";
    std::cout << "final_objective: " << result.f << "\n";
    std::cout << "final_grad_norm: " << result.gradfx_norm << "\n";
    std::cout << "elapsed_time_s: " << result.elapsed_time << "\n";

    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "optimizer_diagnostics failed: " << e.what() << std::endl;
    return 1;
  }
}
