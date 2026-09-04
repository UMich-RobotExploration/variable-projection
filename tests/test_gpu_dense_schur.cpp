/**
 * @file test_gpu_dense_schur.cpp
 * @brief Device formation of the dense Schur complement vs the host formation.
 *
 * Formulation::Dense needs Q_sc = Q_c - B M^{-1} B^T formed explicitly.
 * Problem::fillDenseSchurMatrix() builds it on the host (blocked CHOLMOD
 * multi-RHS solves); VarProGPU::formDenseSchurOnDevice() builds it on the
 * device (cuSPARSE sparse-to-dense + cuDSS + SpMM). They must agree to
 * round-off, and the operator each induces must agree too.
 *
 * The two use different factorizations of M (CHOLMOD vs cuDSS, different
 * orderings), so the tolerance is a relative one on the scale of Q_sc rather
 * than exact equality.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Types.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#endif

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#ifndef VARPRO_EXAMPLES_DIR
#define VARPRO_EXAMPLES_DIR "/home/nikolas/variable-projection/examples"
#endif

namespace {

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what << std::endl;
  if (!ok) ++g_failures;
}

double seconds(std::chrono::high_resolution_clock::time_point a,
               std::chrono::high_resolution_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

#ifdef VARPRO_HAVE_CUDA
void runDataset(const std::string &name, const std::string &pyfg, int rank) {
  std::cout << "\n=== " << name << " (rank " << rank << ") ===" << std::endl;

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  prob.setRank(rank);
  prob.setFormulation(VarPro::Formulation::Dense);  // host formation

  const VarPro::Matrix &Qsc_host = prob.Qsc_dense_;
  const int p = static_cast<int>(Qsc_host.rows());
  std::cout << "  p = " << p << ", host formation "
            << prob.getDensePrecomputeTimeS() << " s" << std::endl;

  VarProGPU::GpuContext ctx;
  auto pre = VarProGPU::buildPrecomputeResult(prob);

  const auto t0 = std::chrono::high_resolution_clock::now();
  VarProGPU::GpuDenseSchurOperator gpu_op(pre, ctx);
  ctx.synchronize();
  const auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "  device formation " << seconds(t0, t1) << " s ("
            << prob.getDensePrecomputeTimeS() / seconds(t0, t1) << "x)"
            << std::endl;

  check(gpu_op.deviceFormed(), "Q_sc was formed on device");

  // 1. The matrices themselves.
  VarPro::Matrix Qsc_gpu = gpu_op.downloadQsc();
  const double scale = Qsc_host.cwiseAbs().maxCoeff();
  const double abs_err = (Qsc_gpu - Qsc_host).cwiseAbs().maxCoeff();
  const double rel_err = abs_err / scale;
  std::cout << "    max|Q_gpu - Q_host| = " << abs_err << " (rel " << rel_err
            << ", scale " << scale << ")" << std::endl;
  check(rel_err < 1e-10, "device Q_sc matches host Q_sc");

  // 2. Symmetry -- the operator application is a SYMM that reads only the
  //    lower triangle, so an asymmetric Q_sc would silently give wrong answers.
  const double asym =
      (Qsc_gpu - Qsc_gpu.transpose()).cwiseAbs().maxCoeff() / scale;
  std::cout << "    relative asymmetry = " << asym << std::endl;
  check(asym < 1e-10, "device Q_sc is symmetric");

  // 3. The induced operator, on a random iterate.
  VarPro::Matrix X = prob.getRandomInitialGuess();
  VarPro::Matrix Y_gpu = gpu_op.apply(X);
  VarPro::Matrix Y_host = Qsc_host * X;
  const double op_err =
      (Y_gpu - Y_host).cwiseAbs().maxCoeff() / Y_host.cwiseAbs().maxCoeff();
  std::cout << "    relative operator error = " << op_err << std::endl;
  check(op_err < 1e-10, "device operator matches host operator");
}
#endif  // VARPRO_HAVE_CUDA

}  // namespace

int main() {
#ifndef VARPRO_HAVE_CUDA
  std::cout << "Built without CUDA -- nothing to test." << std::endl;
  return 0;
#else
  if (!VarProGPU::gpuSparseSolverAvailable()) {
    std::cout << "Built without cuDSS -- device Schur formation unavailable, "
                 "skipping."
              << std::endl;
    return 0;
  }

  const std::string examples = VARPRO_EXAMPLES_DIR;
  runDataset("tinyGrid3D", examples + "/data/pgo/tinyGrid3D/tinyGrid3D.pyfg", 5);
  runDataset("single_drone",
             examples + "/data/raslam/single_drone/single_drone.pyfg", 5);
  runDataset("smallGrid3D",
             examples + "/data/pgo/smallGrid3D/smallGrid3D.pyfg", 5);

  std::cout << std::endl;
  if (g_failures == 0) {
    std::cout << "All GPU dense-Schur tests PASSED." << std::endl;
    return 0;
  }
  std::cout << g_failures << " check(s) FAILED." << std::endl;
  return 1;
#endif
}
