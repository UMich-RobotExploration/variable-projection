/**
 * @file benchmark_precompute.cpp
 * @brief Time only the implicit-formulation precompute step.
 *
 * Parses each dataset (.pyfg) and calls updateProblemData(), which internally
 * runs fillImplicitFormulationMatrices() and records its wall-clock time as
 * implicit_precompute_time_s_. With --gpu (and a CUDA build) it additionally
 * times the GPU-side upload that converts the CPU precompute into a
 * GpuSchurOperator ready for solving.
 *
 * Output (one tab-separated row per dataset):
 *   <dataset>\t<cpu_precompute_s>\t<gpu_upload_s>\t<wall_s>\t<n_poses>\t<n_landmarks>\t<n_ranges>
 *
 * The gpu_upload_s column is 0 when --gpu is not passed (or CUDA is off).
 *
 * Usage:
 *   ./benchmark_precompute [--gpu] [--rank R] <dataset_or_dir> ...
 *
 * A directory argument is auto-resolved to the .pyfg file it contains.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#endif

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

static std::string findPyfg(const std::string& path) {
  if (fs::is_regular_file(path) &&
      path.size() >= 5 && path.rfind(".pyfg") == path.size() - 5) return path;
  if (fs::is_directory(path)) {
    for (const auto& e : fs::directory_iterator(path)) {
      if (e.path().extension() == ".pyfg") return e.path().string();
    }
  }
  return "";
}

int main(int argc, char** argv) {
  bool gpu = false;
  int rank = 5;
  std::vector<std::string> datasets;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gpu") {
      gpu = true;
    } else if (a == "--rank" && i + 1 < argc) {
      rank = std::stoi(argv[++i]);
    } else {
      datasets.push_back(a);
    }
  }
  if (datasets.empty()) {
    std::cerr << "usage: " << argv[0] << " [--gpu] [--rank R] <dataset_or_dir> ...\n";
    return 1;
  }

#ifndef VARPRO_HAVE_CUDA
  if (gpu) {
    std::cerr << "this binary was built without VARPRO_HAVE_CUDA; --gpu is a no-op\n";
    gpu = false;
  }
#endif

#ifdef VARPRO_HAVE_CUDA
  // Construct GpuContext once up-front so its initialisation isn't billed to
  // the first dataset's upload time.
  std::unique_ptr<VarProGPU::GpuContext> gpu_ctx;
  if (gpu) gpu_ctx = std::make_unique<VarProGPU::GpuContext>();
#endif

  std::cout << "path\tcpu_precompute_s\tgpu_upload_s\twall_s\tposes\tlandmarks\tranges\n";
  int failures = 0;
  for (const auto& arg : datasets) {
    const std::string pyfg = findPyfg(arg);
    if (pyfg.empty()) {
      std::cerr << "skip (no .pyfg): " << arg << "\n";
      ++failures;
      continue;
    }
    try {
      auto t0 = Clock::now();
      VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
      prob.updateProblemData();
      const double cpu_precompute_s = prob.getImplicitPrecomputeTimeS();

      double gpu_upload_s = 0.0;
#ifdef VARPRO_HAVE_CUDA
      if (gpu) {
        prob.setFormulation(VarPro::Formulation::Implicit);
        prob.setRank(rank);
        auto tg = Clock::now();
        auto pre = VarProGPU::buildPrecomputeResult(prob);
        VarProGPU::GpuSchurOperator gpu_op(pre, *gpu_ctx);
        gpu_ctx->synchronize();
        gpu_upload_s = Sec(Clock::now() - tg).count();
      }
#endif

      const double wall_s = Sec(Clock::now() - t0).count();
      std::cout << arg << "\t"
                << cpu_precompute_s << "\t"
                << gpu_upload_s << "\t"
                << wall_s << "\t"
                << prob.numPoses() << "\t"
                << prob.numLandmarks() << "\t"
                << prob.numRangeMeasurements() << "\n"
                << std::flush;
    } catch (const std::exception& e) {
      std::cerr << "error on " << arg << ": " << e.what() << "\n";
      ++failures;
    }
  }
  return failures == 0 ? 0 : 2;
}
