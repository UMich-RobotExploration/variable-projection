/**
 * @file benchmark_schur.cpp
 * @brief Benchmark the matrix-free Schur operator (CPU vs GPU).
 *
 * Reports:
 *   - Preprocessing time (sparse Cholesky + matrix upload)
 *   - Per-apply latency for CPU and GPU implementations
 *   - GPU speedup factor
 *
 * Usage:
 *   ./benchmark_schur [--dataset <path>] [--reps <N>] [--rank <r>]
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Types.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#endif

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct BenchArgs {
  std::string dataset =
      "/home/nikolas/variable-projection/examples/data/pgo/tinyGrid3D/tinyGrid3D.pyfg";
  int reps = 200;
  int rank = 5;
};

static BenchArgs parseArgs(int argc, char** argv) {
  BenchArgs args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dataset") == 0 && i + 1 < argc)
      args.dataset = argv[++i];
    else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
      args.reps = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc)
      args.rank = std::atoi(argv[++i]);
  }
  return args;
}

static void printRow(const std::string& name, double ms_per_op,
                      int p, int r, double speedup = 1.0) {
  std::cout << std::left  << std::setw(28) << name
            << std::right << std::setw(8)  << std::fixed << std::setprecision(3) << ms_per_op << " ms/op"
            << "   p=" << std::setw(6) << p
            << "  r=" << std::setw(3) << r;
  if (speedup > 1.0)
    std::cout << "   speedup=" << std::setprecision(2) << speedup << "×";
  std::cout << "\n";
}

int main(int argc, char** argv) {
  BenchArgs args = parseArgs(argc, argv);

  if (!fs::exists(args.dataset)) {
    std::cerr << "Dataset not found: " << args.dataset << "\n";
    return 1;
  }

  std::cout << "=== Schur Operator Benchmark ===\n"
            << "Dataset: " << args.dataset << "\n"
            << "Reps: "    << args.reps    << "\n"
            << "Rank: "    << args.rank    << "\n\n";

  // Build problem
  auto t0 = Clock::now();
  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(args.dataset);
  prob.updateProblemData();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(args.rank);
  double build_ms = Ms(Clock::now() - t0).count();
  std::cout << "Problem build:  " << build_ms << " ms\n";

  // Precompute (Cholesky + matrix extraction)
  t0 = Clock::now();
  auto pre = VarProGPU::buildPrecomputeResult(prob);
  double precomp_ms = Ms(Clock::now() - t0).count();
  std::cout << "Preprocessing:  " << precomp_ms << " ms\n";
  std::cout << "p=" << pre.p << "  m=" << pre.m << "  r=" << pre.r << "\n\n";

  // Random input
  VarPro::Matrix X = VarPro::Matrix::Random(pre.p, pre.r);

  // --- CPU benchmark ---
  VarProGPU::CpuSchurOperator cpu_op(pre);

  // Warmup
  for (int i = 0; i < 3; ++i) (void)cpu_op.apply(X);

  t0 = Clock::now();
  for (int i = 0; i < args.reps; ++i) (void)cpu_op.apply(X);
  double cpu_total = Ms(Clock::now() - t0).count();
  double cpu_ms    = cpu_total / args.reps;

  printRow("CPU (matrix-free)", cpu_ms, pre.p, pre.r);

#ifdef VARPRO_HAVE_CUDA
  // --- GPU upload time ---
  VarProGPU::GpuContext ctx;
  t0 = Clock::now();
  VarProGPU::GpuSchurOperator gpu_op(pre, ctx);
  ctx.synchronize();
  double gpu_upload_ms = Ms(Clock::now() - t0).count();
  std::cout << "GPU upload:     " << gpu_upload_ms << " ms\n";

  // GPU warmup
  for (int i = 0; i < 3; ++i) (void)gpu_op.apply(X);
  ctx.synchronize();

  t0 = Clock::now();
  for (int i = 0; i < args.reps; ++i) (void)gpu_op.apply(X);
  ctx.synchronize();
  double gpu_total = Ms(Clock::now() - t0).count();
  double gpu_ms    = gpu_total / args.reps;
  double speedup   = cpu_ms / gpu_ms;

  printRow("GPU (cuSPARSE)", gpu_ms, pre.p, pre.r, speedup);
#else
  std::cout << "GPU benchmark: SKIPPED (no CUDA)\n";
#endif

  // --- Summary ---
  std::cout << "\nSummary\n-------\n";
  std::cout << "CPU ms/op:       " << cpu_ms << "\n";
#ifdef VARPRO_HAVE_CUDA
  std::cout << "GPU ms/op:       " << gpu_ms << "\n";
  std::cout << "GPU speedup:     " << cpu_ms / gpu_ms << "×\n";
  std::cout << "Upload amortized over " << args.reps << " ops: "
            << gpu_upload_ms / args.reps << " ms/op\n";
#endif

  return 0;
}
