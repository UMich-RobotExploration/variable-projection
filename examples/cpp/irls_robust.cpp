/**
 * @file irls_robust.cpp
 * @brief Iteratively Reweighted Least Squares (IRLS) wrapper around the
 *        VarPro solver, supporting Geman-McClure (GM) and Truncated Least
 *        Squares (TLS) robust kernels. Implements Algorithm 2 of the SPARSER
 *        paper at the application level: at each outer iteration we
 *        recompute per-RPM weights from the current iterate, rescale the
 *        covariances, refresh the implicit precompute via
 *        Problem::updateProblemData(), and run a full inner TNT solve.
 *
 * The init is built from sequential pose-pose edges (odometry), matching
 * init_from_odometry.cpp. Only PGO-style problems (no ranges, no landmarks)
 * are handled in this first cut.
 *
 * Usage:
 *   ./irls_robust <pyfg> <init.tum> <final.tum>
 *                  [--kernel {gm|tls}]
 *                  [--formulation {explicit|expvp|impl|dense}]
 *                  [--c2 <c-squared>] [--max-irls <K>] [--rel-tol <tol>]
 *                  [--verbose]
 *
 * Output line (machine-readable, last line of stdout):
 *   IRLS_RESULT kernel=<k> form=<f> dim=<d> outer=<n> inner=<sum>
 *               final_cost=<c> robust_cost=<rc> total_s=<t>
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>

#ifdef VARPRO_HAVE_CUDA
#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/GpuRTRSolver.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#endif

#include <Eigen/Geometry>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <memory>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

struct PoseSE3 {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// ---------------------------------------------------------------------------
// Odom init (mirrors init_from_odometry.cpp)
// ---------------------------------------------------------------------------

PoseSE3 liftMeasurement(const VarPro::Matrix &R, const VarPro::Vector &t) {
  PoseSE3 out;
  if (R.rows() == 3) {
    out.R = R;
    out.t = t;
  } else if (R.rows() == 2) {
    out.R.setIdentity();
    out.R.topLeftCorner<2, 2>() = R;
    out.t.head<2>() = t;
    out.t.z() = 0.0;
  } else {
    throw std::runtime_error("liftMeasurement: only 2D/3D supported");
  }
  return out;
}

PoseSE3 compose(const PoseSE3 &a, const PoseSE3 &b) {
  PoseSE3 c;
  c.R = a.R * b.R;
  c.t = a.R * b.t + a.t;
  return c;
}

bool isSequentialOdom(const VarPro::Symbol &s1, const VarPro::Symbol &s2) {
  return s1.chr() == s2.chr() && s2.index() == s1.index() + 1;
}

// Sample a small random SE(d) perturbation. For d==2 only the in-plane axes
// are perturbed (z translation = 0, rotation around z). The rng draws are
// fixed-count per call (6 doubles for d==3, 3 doubles for d==2) so a given
// seed yields a deterministic, formulation-independent perturbation sequence.
PoseSE3 sampleEdgePerturbation(std::mt19937 &rng, int dim,
                                double sigma_rot_rad, double sigma_trans) {
  PoseSE3 dT;
  std::normal_distribution<double> N(0.0, 1.0);
  if (dim == 3) {
    Eigen::Vector3d w(sigma_rot_rad * N(rng),
                       sigma_rot_rad * N(rng),
                       sigma_rot_rad * N(rng));
    const double theta = w.norm();
    if (theta < 1e-12) {
      dT.R.setIdentity();
    } else {
      dT.R = Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
    }
    dT.t = Eigen::Vector3d(sigma_trans * N(rng),
                            sigma_trans * N(rng),
                            sigma_trans * N(rng));
  } else {
    const double theta = sigma_rot_rad * N(rng);
    dT.R.setIdentity();
    dT.R.topLeftCorner<2, 2>() << std::cos(theta), -std::sin(theta),
                                   std::sin(theta),  std::cos(theta);
    dT.t = Eigen::Vector3d(sigma_trans * N(rng), sigma_trans * N(rng), 0.0);
  }
  return dT;
}

std::unordered_map<VarPro::Key, PoseSE3>
chainOdometry(const VarPro::Problem &prob, int &n_odom, int &n_skipped,
              int init_seed = 0, double sigma_rot_rad = 0.0,
              double sigma_trans = 0.0) {
  const auto pose_map = prob.getPoseSymbolMap();
  std::map<VarPro::Symbol, std::pair<VarPro::Symbol, PoseSE3>> next_edge;
  n_odom = 0;
  n_skipped = 0;
  for (const auto &m : prob.getRPMs()) {
    if (isSequentialOdom(m.first_id, m.second_id)) {
      next_edge.emplace(m.first_id,
                         std::make_pair(m.second_id, liftMeasurement(m.R, m.t)));
      ++n_odom;
    } else {
      ++n_skipped;
    }
  }
  // Pre-perturb each odometry edge so noise depends only on the seed and the
  // (deterministic) edge ordering of next_edge — not on the chain traversal
  // order or the formulation. Edges are visited in std::map key order.
  if (init_seed > 0 && (sigma_rot_rad > 0.0 || sigma_trans > 0.0)) {
    std::mt19937 rng(static_cast<uint32_t>(init_seed));
    const int dim = prob.dim();
    for (auto &kv : next_edge) {
      PoseSE3 dT = sampleEdgePerturbation(rng, dim, sigma_rot_rad, sigma_trans);
      kv.second.second = compose(kv.second.second, dT);
    }
  }
  std::unordered_map<VarPro::Key, PoseSE3> global;
  for (const auto &[sym, _idx] : pose_map) {
    if (global.count(sym.key())) continue;
    global[sym.key()] = PoseSE3{};
    VarPro::Symbol cur = sym;
    while (true) {
      auto it = next_edge.find(cur);
      if (it == next_edge.end()) break;
      const VarPro::Symbol &nxt = it->second.first;
      const PoseSE3 &Tij = it->second.second;
      if (!pose_map.count(nxt) || global.count(nxt.key())) break;
      global[nxt.key()] = compose(global[cur.key()], Tij);
      cur = nxt;
    }
  }
  return global;
}

void writeTum(const std::string &path,
               const std::map<VarPro::Symbol, int> &pose_map,
               const std::unordered_map<VarPro::Key, PoseSE3> &global) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("could not open " + path);
  out << std::fixed;
  out.precision(9);
  for (const auto &[sym, idx] : pose_map) {
    auto it = global.find(sym.key());
    if (it == global.end()) continue;
    const PoseSE3 &T = it->second;
    Eigen::Quaterniond q(T.R);
    q.normalize();
    out << idx << ' '
        << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' '
        << q.x()    << ' ' << q.y()   << ' ' << q.z()   << ' ' << q.w()
        << '\n';
  }
}

// ---------------------------------------------------------------------------
// Variable-matrix construction + extraction (handles all 3 formulations)
// ---------------------------------------------------------------------------

// Build Y0 from per-pose SE(d) trajectory. Layout per Problem.h:
//   rows 0 .. n_poses*d-1                : R^T blocks (d x rank)
//   rows n_poses*d .. n_poses*d+n_r-1    : bearing unit vectors (1 x rank)
//   for Explicit/VarPro only, rows after : 1 x rank translation rows
VarPro::Matrix
buildVariableFromPoses(const VarPro::Problem &prob,
                        const std::unordered_map<VarPro::Key, PoseSE3> &global,
                        VarPro::Formulation form) {
  const int d = prob.dim();
  const int n_poses = prob.numPoses();
  const int n_r = prob.numRangeMeasurements();
  int n_rows = n_poses * d + n_r;
  if (!VarPro::isMarginalized(form)) {
    n_rows += n_poses + prob.numLandmarks();  // pose translations + landmarks
  }
  VarPro::Matrix Y = VarPro::Matrix::Zero(n_rows, d);

  for (const auto &[sym, idx] : prob.getPoseSymbolMap()) {
    auto it = global.find(sym.key());
    if (it == global.end()) continue;
    const PoseSE3 &T = it->second;
    if (d == 3) {
      Y.block(idx * d, 0, d, d) = T.R.transpose();
    } else {
      Y.block(idx * d, 0, d, d) = T.R.topLeftCorner<2, 2>().transpose();
    }
    if (!VarPro::isMarginalized(form)) {
      const int trans_offset = n_poses * d + n_r;
      for (int k = 0; k < d; ++k) Y(trans_offset + idx, k) = T.t(k);
    }
  }

  // Random unit-norm bearings for the range block (one row per range
  // measurement). Without this, projectToManifold would divide by zero on
  // the all-zero bearing rows and produce NaNs.
  if (n_r > 0) {
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<double> N(0.0, 1.0);
    for (int i = 0; i < n_r; ++i) {
      VarPro::Vector v(d);
      for (int j = 0; j < d; ++j) v(j) = N(rng);
      const double n = v.norm();
      if (n > 1e-12) v /= n;
      else v(0) = 1.0;
      Y.row(n_poses * d + i) = v.transpose();
    }
  }
  return Y;
}

// Lift a solver iterate to the full variable matrix (rotations + ranges +
// translations). For Implicit, translations are derived analytically; for
// Explicit/VarPro the iterate is already the full matrix.
VarPro::Matrix liftToFull(const VarPro::Problem &prob, const VarPro::Matrix &Y) {
  return VarPro::isMarginalized(prob.getFormulation())
             ? prob.getTranslationExplicitSolution(Y)
             : Y;
}

// Extract per-pose (R, t) from a lifted variable matrix.
std::unordered_map<VarPro::Key, PoseSE3>
posesFromFull(const VarPro::Problem &prob, const VarPro::Matrix &Xfull) {
  const int d = prob.dim();
  const int trans_offset = prob.rotAndRangeMatrixSize();
  std::unordered_map<VarPro::Key, PoseSE3> global;
  for (const auto &[sym, idx] : prob.getPoseSymbolMap()) {
    PoseSE3 T;
    VarPro::Matrix Rblock = Xfull.block(idx * d, 0, d, d);  // stored as R^T
    if (d == 3) {
      T.R = Rblock.transpose();
    } else {
      T.R.setIdentity();
      T.R.topLeftCorner<2, 2>() = Rblock.transpose();
    }
    VarPro::Matrix trow = Xfull.block(trans_offset + idx, 0, 1, d);
    T.t = (d == 3) ? Eigen::Vector3d(trow(0, 0), trow(0, 1), trow(0, 2))
                    : Eigen::Vector3d(trow(0, 0), trow(0, 1), 0.0);
    global[sym.key()] = T;
  }
  return global;
}

// Convenience wrapper kept for the post-solve TUM write path.
std::unordered_map<VarPro::Key, PoseSE3>
extractPoses(const VarPro::Problem &prob, const VarPro::Matrix &Y) {
  return posesFromFull(prob, liftToFull(prob, Y));
}

// ---------------------------------------------------------------------------
// Per-measurement residual + robust kernel
// ---------------------------------------------------------------------------

// Precision-weighted squared SE(d) residual for one RPM, computed from the
// *original* (unweighted) covariance — this is what feeds the robust kernel.
double rpmSquaredResidual(
    const VarPro::RelativePoseMeasurement &meas,
    const VarPro::Matrix &cov_original,
    const std::unordered_map<VarPro::Key, PoseSE3> &poses) {
  const auto it_i = poses.find(meas.first_id.key());
  const auto it_j = poses.find(meas.second_id.key());
  if (it_i == poses.end() || it_j == poses.end()) return 0.0;
  const PoseSE3 &Ti = it_i->second;
  const PoseSE3 &Tj = it_j->second;

  const int d = static_cast<int>(meas.R.rows());
  VarPro::Matrix Ri(d, d), Rj(d, d);
  VarPro::Vector ti(d), tj(d);
  if (d == 3) {
    Ri = Ti.R;
    Rj = Tj.R;
    ti = Ti.t;
    tj = Tj.t;
  } else {
    Ri = Ti.R.topLeftCorner<2, 2>();
    Rj = Tj.R.topLeftCorner<2, 2>();
    ti = Ti.t.head<2>();
    tj = Tj.t.head<2>();
  }

  // Residuals.
  VarPro::Matrix rR = Rj - Ri * meas.R;
  VarPro::Vector rT = tj - ti - Ri * meas.t;

  // Use the *original* covariance to derive precisions for the robust weight.
  VarPro::RelativePoseMeasurement meas_orig = meas;
  meas_orig.cov = cov_original;
  const double rot_prec = meas_orig.getRotPrecision();
  const double trans_prec = meas_orig.getTransPrecision();

  return rot_prec * rR.squaredNorm() + trans_prec * rT.squaredNorm();
}

// Precision-weighted squared range residual for one RangeMeasurement.
// Range residual is `tj - ti - u_ij * r̃` where u_ij is the bearing unit
// vector (lives in Xfull at row n_poses*d + range_idx) and r̃ is the measured
// distance. Precision = 1 / cov_original.
double rangeSquaredResidual(
    const VarPro::RangeMeasurement &meas,
    int range_idx,
    double cov_original,
    const VarPro::Matrix &Xfull,
    const VarPro::Problem &prob) {
  const int d = prob.dim();
  const int n_poses = prob.numPoses();
  const int bearing_row = n_poses * d + range_idx;

  VarPro::Vector u(d), ti(d), tj(d);
  const auto trans_i = prob.getTranslationIdx(meas.first_id);
  const auto trans_j = prob.getTranslationIdx(meas.second_id);
  for (int k = 0; k < d; ++k) {
    u(k) = Xfull(bearing_row, k);
    ti(k) = Xfull(trans_i, k);
    tj(k) = Xfull(trans_j, k);
  }
  VarPro::Vector r_vec = tj - ti - u * meas.r;
  const double precision = 1.0 / cov_original;
  return precision * r_vec.squaredNorm();
}

enum class Kernel { GemanMcClure, TruncatedLeastSquares };

double kernelRho(Kernel k, double s, double c2) {
  switch (k) {
    case Kernel::GemanMcClure:
      return (c2 * s) / (c2 + s);
    case Kernel::TruncatedLeastSquares:
      return std::min(s, c2);
  }
  return s;
}

double kernelWeight(Kernel k, double s, double c2) {
  switch (k) {
    case Kernel::GemanMcClure: {
      const double denom = c2 + s;
      return (c2 * c2) / (denom * denom);
    }
    case Kernel::TruncatedLeastSquares:
      return (s < c2) ? 1.0 : 0.0;
  }
  return 1.0;
}

const char *kernelName(Kernel k) {
  return k == Kernel::GemanMcClure ? "gm" : "tls";
}

const char *formName(VarPro::Formulation f) {
  switch (f) {
    case VarPro::Formulation::Explicit: return "explicit";
    case VarPro::Formulation::ExplicitVarPro: return "expvp";
    case VarPro::Formulation::Implicit: return "impl";
    case VarPro::Formulation::Dense: return "dense";
  }
  return "?";
}

}  // namespace

int main(int argc, char **argv) {
  std::string pyfg, init_tum, final_tum;
  Kernel kernel = Kernel::GemanMcClure;
  VarPro::Formulation form = VarPro::Formulation::Implicit;
  double c2 = 25.0;
  int max_irls = 20;
  double rel_tol = 1e-4;
  bool verbose = false;
  bool use_gpu = false;
  bool gnc = false;
  double gnc_init = 64.0;    // initial multiplier on c² (Yang et al. use ~1/μ ramp)
  double gnc_shrink = 1.4;   // per-iteration shrink factor
  int init_seed = 0;         // 0 = noiseless odom init (original behaviour)
  double init_noise_rot_deg = 2.0;  // per-edge rotation noise std (degrees)
  double init_noise_trans = 0.05;   // per-edge translation noise std
  std::string dump_dir;      // if set, write iter_<k>.tum per outer iteration
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--kernel" && i + 1 < argc) {
      const std::string k = argv[++i];
      if (k == "gm") kernel = Kernel::GemanMcClure;
      else if (k == "tls") kernel = Kernel::TruncatedLeastSquares;
      else { std::cerr << "bad --kernel: " << k << "\n"; return 1; }
    } else if (a == "--formulation" && i + 1 < argc) {
      const std::string f = argv[++i];
      if (f == "explicit") form = VarPro::Formulation::Explicit;
      else if (f == "expvp") form = VarPro::Formulation::ExplicitVarPro;
      else if (f == "impl" || f == "implicit") form = VarPro::Formulation::Implicit;
      else if (f == "dense") form = VarPro::Formulation::Dense;
      else { std::cerr << "bad --formulation: " << f << "\n"; return 1; }
    } else if (a == "--c2" && i + 1 < argc) {
      c2 = std::stod(argv[++i]);
    } else if (a == "--max-irls" && i + 1 < argc) {
      max_irls = std::stoi(argv[++i]);
    } else if (a == "--rel-tol" && i + 1 < argc) {
      rel_tol = std::stod(argv[++i]);
    } else if (a == "--gnc") {
      gnc = true;
    } else if (a == "--gnc-init" && i + 1 < argc) {
      gnc_init = std::stod(argv[++i]);
    } else if (a == "--gnc-shrink" && i + 1 < argc) {
      gnc_shrink = std::stod(argv[++i]);
    } else if (a == "--init-seed" && i + 1 < argc) {
      init_seed = std::stoi(argv[++i]);
    } else if (a == "--init-noise-rot-deg" && i + 1 < argc) {
      init_noise_rot_deg = std::stod(argv[++i]);
    } else if (a == "--init-noise-trans" && i + 1 < argc) {
      init_noise_trans = std::stod(argv[++i]);
    } else if (a == "--dump-iterates" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "--verbose") {
      verbose = true;
    } else if (a == "--gpu") {
      use_gpu = true;
    } else {
      positional.push_back(a);
    }
  }
  if (positional.size() != 3) {
    std::cerr << "usage: " << argv[0]
              << " <pyfg> <init.tum> <final.tum> [--kernel gm|tls] "
                 "[--formulation explicit|expvp|impl|dense] [--c2 X] [--max-irls K] "
                 "[--rel-tol T] [--gnc] [--gnc-init S] [--gnc-shrink F] "
                 "[--init-seed S] [--init-noise-rot-deg X] "
                 "[--init-noise-trans X] [--dump-iterates DIR] "
                 "[--verbose]\n";
    return 1;
  }
  pyfg = positional[0];
  init_tum = positional[1];
  final_tum = positional[2];

  const auto t_total_start = Clock::now();

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  const double first_precompute_s = prob.getImplicitPrecomputeTimeS();
  if (prob.numLandmarks() > 0) {
    std::cerr << "note: dataset has " << prob.numLandmarks()
              << " landmarks (their priors are kept unweighted)\n";
  }

  const auto pose_map = prob.getPoseSymbolMap();
  if (pose_map.empty()) {
    std::cerr << "no poses in dataset\n";
    return 1;
  }

  // Build odom init + write the init TUM. When --init-seed > 0, each
  // sequential odometry edge is perturbed by a small random SE(d) before the
  // chain is composed, so the resulting "odometry" drifts deterministically
  // per seed. seed=0 keeps the original noiseless behaviour.
  int n_odom = 0, n_skipped = 0;
  const double sigma_rot_rad = init_noise_rot_deg * M_PI / 180.0;
  auto init_poses = chainOdometry(prob, n_odom, n_skipped,
                                    init_seed, sigma_rot_rad, init_noise_trans);
  writeTum(init_tum, pose_map, init_poses);

  // Snapshot original RPM and range covariances; IRLS rewrites prob's covs
  // in place each outer iteration.
  std::vector<VarPro::Matrix> orig_rpm_covs;
  orig_rpm_covs.reserve(prob.numPosePoseMeasurements());
  for (const auto &m : prob.getRPMs()) orig_rpm_covs.push_back(m.cov);
  std::vector<double> orig_range_covs;
  orig_range_covs.reserve(prob.numRangeMeasurements());
  for (const auto &m : prob.getRangeMeasurements()) orig_range_covs.push_back(m.cov);

  // IRLS refreshes the problem data every outer iteration, so the precompute
  // is paid once per iteration rather than once per solve. Track the total:
  // for Dense this includes re-forming the p x p reduced system each time,
  // which is the dominant cost of that baseline in the robust setting.
  double total_precompute_s = prob.getImplicitPrecomputeTimeS();
  const double dense_gb = (form == VarPro::Formulation::Dense)
                              ? prob.densePrecomputeGB()
                              : 0.0;

  // Set formulation + rank = dim (collapse the relaxation to native SE(d)).
  // For Dense this is where the explicit reduced system is first formed.
  prob.setFormulation(form);
  prob.setRank(prob.dim());
  if (form == VarPro::Formulation::Dense) {
    total_precompute_s += prob.getDensePrecomputeTimeS();
    std::cerr << "note: Dense formulation will hold a "
              << prob.densePrecomputeGB()
              << " GB reduced system, re-formed every IRLS outer iteration\n";
  }

  VarPro::Matrix Y = buildVariableFromPoses(prob, init_poses, form);
  Y = prob.projectToManifold(Y);

  int total_inner = 0;
  // Time spent writing per-iteration iterate dumps, subtracted from the
  // reported cumulative time so --dump-iterates never inflates a timing.
  double dump_overhead_s = 0.0;
  double prev_robust_cost = std::numeric_limits<double>::infinity();
  double final_robust_cost = std::numeric_limits<double>::infinity();
  double final_inner_cost = std::numeric_limits<double>::infinity();
  int outer_iter = 0;

  // Optional warm-start: solve the unweighted (standard LSQ) problem once so
  // that the first weight update is computed at a sensible iterate. This is
  // only needed for datasets with ranges/landmarks — our range bearings are
  // initialized randomly, so the first IRLS residual evaluation would be
  // garbage without it.
  //
  // For pure PGO it actively *hurts*: on outlier-corrupted graphs, the
  // unweighted LSQ collapses the trajectory to origin (where identity-loop
  // outliers have zero residual), and IRLS then sees the outliers as
  // perfect inliers and the real inliers as outliers. Once that happens
  // there's no recovery, regardless of kernel or GNC schedule.
  //
  // Heuristic: only warm-start when the problem has range measurements
  // (those are the ones that need a random init for their bearing rows).
  // For pure PGO we trust the odometry init directly.
  //
  // (We deliberately skip an extra updateProblemData() here — it was already
  // called immediately after parsing, and the data matrix doesn't depend on
  // setFormulation/setRank.)
  const bool need_warm_start = prob.numRangeMeasurements() > 0;
  if (need_warm_start) {
    auto warm = VarPro::solveProblem(prob, Y, /*verbose=*/false);
    Y = warm.x;
    total_inner += static_cast<int>(warm.objective_values.size());
    if (verbose) {
      std::cout << "  warm-start LSQ cost=" << warm.f
                << " inner_iters=" << warm.objective_values.size() << "\n";
    }
  }

  // GNC: anneal the effective kernel scale from c2_init (very permissive)
  // down to the target c2. For GM, multiplying c² by μ ≥ 1 makes the kernel
  // closer to pure LSQ (weights → 1 everywhere); for TLS, the same multiplier
  // raises the rejection threshold so almost no measurements are clipped at
  // the start. We shrink toward the target by `gnc_shrink` per outer iter so
  // the algorithm sees the easy convex-like problem first and the true
  // non-convex objective last.
  double c2_eff = gnc ? c2 * gnc_init : c2;

  // GPU inner solve. The context and solver are created once; the *operator*
  // is rebuilt every outer iteration because IRLS rewrites the covariances,
  // which changes Qmain/B/M and therefore the whole precompute.
#ifdef VARPRO_HAVE_CUDA
  std::unique_ptr<VarProGPU::GpuContext> gpu_ctx;
  std::unique_ptr<VarProGPU::GpuRTRSolver> gpu_solver;
  VarProGPU::RTRParams gpu_params;
  // Built once, then refreshed in place each outer iteration: reweighting
  // changes values but not sparsity, so the CSR structure and the cuDSS
  // symbolic factorization are reused (Algorithm 2, lines 2 and 5). `gpu_pre`
  // must outlive the operators -- they hold a reference to it.
  VarProGPU::VarProPrecomputeResult gpu_pre;
  std::unique_ptr<VarProGPU::GpuSchurOperator> gpu_op_impl;
  std::unique_ptr<VarProGPU::GpuDenseSchurOperator> gpu_op_dense;
  std::unique_ptr<VarProGPU::GpuExplicitOperator> gpu_op_expl;
  if (use_gpu) {
    gpu_ctx = std::make_unique<VarProGPU::GpuContext>();
    gpu_solver = std::make_unique<VarProGPU::GpuRTRSolver>(*gpu_ctx);
    gpu_params.max_outer_iters = 250;
    gpu_params.verbose = false;
    // Same override convention as the CPU solver (src/Solver.cpp): <= 0 means
    // no limit, so an IRLS sweep can run uncapped without a rebuild.
    if (const char *e = std::getenv("VARPRO_MAX_ITERATIONS")) {
      const long n = std::atol(e);
      gpu_params.max_outer_iters = (n > 0) ? static_cast<int>(n) : 1000000;
    }
    if (const char *e = std::getenv("VARPRO_MAX_COMPUTATION_TIME")) {
      const double t = std::atof(e);
      gpu_params.max_time_seconds =
          (t > 0.0) ? t : std::numeric_limits<double>::max();
    }
    // Dense forms its reduced system on the device; the host copy would be
    // pure waste (and is the dominant cost at large p).
    if (form == VarPro::Formulation::Dense)
      prob.setHostDenseSchurEnabled(false);
  }
#else
  if (use_gpu) {
    std::cerr << "irls_robust: --gpu requested but this build has no CUDA "
                  "support (configure with -DENABLE_GPU=ON)\n";
    return 1;
  }
#endif

  std::cout << "IRLS kernel=" << kernelName(kernel)
            << " formulation=" << formName(form)
            << " dim=" << prob.dim()
            << " c2=" << c2;
  if (gnc) std::cout << " gnc=on c2_init=" << c2_eff
                      << " shrink=" << gnc_shrink;
  std::cout << " poses=" << pose_map.size()
            << " rpms=" << orig_rpm_covs.size()
            << " ranges=" << orig_range_covs.size() << "\n";

  for (outer_iter = 0; outer_iter < max_irls; ++outer_iter) {
    // 1. Per-measurement residuals + weights under the *current* effective c².
    //    Lift Y once and reuse for both RPM and range residual computations.
    VarPro::Matrix Xfull = liftToFull(prob, Y);
    // Iter-0 special case: for pure PGO we skip the warm-start LSQ, which
    // means Implicit's `liftToFull` would derive translations analytically
    // from the still-uniform weights — that's the LSQ-with-outliers
    // translation, *not* the odom one. Explicit/ExpVP read t_odom straight
    // out of Y. To make all three formulations evaluate iter-0 residuals at
    // the same (R_odom, t_odom), fall back to the odom-init pose map here.
    auto poses_now = (outer_iter == 0 && !need_warm_start)
                         ? init_poses
                         : posesFromFull(prob, Xfull);
    auto &rpms = prob.getMutableRPMs();
    double robust_cost = 0.0;
    for (size_t i = 0; i < rpms.size(); ++i) {
      const double s = rpmSquaredResidual(rpms[i], orig_rpm_covs[i], poses_now);
      const double w = kernelWeight(kernel, s, c2_eff);
      // Rescale cov so the precision = w * original_precision (since
      // precision is built from cov^{-1}, dividing cov by w scales precision
      // by w). A floor avoids singular cov from w=0 in TLS — those edges
      // become near-zero weight but remain numerically stable.
      const double w_floor = std::max(w, 1e-12);
      rpms[i].cov = orig_rpm_covs[i] / w_floor;
      robust_cost += kernelRho(kernel, s, c2_eff);
    }

    // 1b. Range-measurement weights + cov rescale.
    auto &ranges = prob.getMutableRangeMeasurements();
    for (size_t i = 0; i < ranges.size(); ++i) {
      const double s = rangeSquaredResidual(
          ranges[i], static_cast<int>(i), orig_range_covs[i], Xfull, prob);
      const double w = kernelWeight(kernel, s, c2_eff);
      const double w_floor = std::max(w, 1e-12);
      ranges[i].cov = orig_range_covs[i] / w_floor;
      robust_cost += kernelRho(kernel, s, c2_eff);
    }

    // 2. Refresh the precompute under the new weights and solve inner.
    prob.updateProblemData();
    total_precompute_s +=
        prob.getImplicitPrecomputeTimeS() + prob.getDensePrecomputeTimeS();
    VarPro::Matrix x_next;
    VarPro::Scalar f_next = 0;
    int n_inner = 0;
#ifdef VARPRO_HAVE_CUDA
    if (use_gpu) {
      VarProGPU::RTRResult r;
      if (form == VarPro::Formulation::Implicit) {
        gpu_pre = VarProGPU::buildPrecomputeResult(prob);
        if (!gpu_op_impl || !gpu_op_impl->update())
          gpu_op_impl = std::make_unique<VarProGPU::GpuSchurOperator>(gpu_pre, *gpu_ctx);
        r = gpu_solver->solve(prob, *gpu_op_impl, Y, gpu_params);
      } else if (form == VarPro::Formulation::Dense) {
        gpu_pre = VarProGPU::buildPrecomputeResult(prob);
        if (!gpu_op_dense || !gpu_op_dense->update())
          gpu_op_dense = std::make_unique<VarProGPU::GpuDenseSchurOperator>(gpu_pre, *gpu_ctx);
        r = gpu_solver->solve(prob, *gpu_op_dense, Y, gpu_params);
      } else {
        if (!gpu_op_expl || !gpu_op_expl->update())
          gpu_op_expl = std::make_unique<VarProGPU::GpuExplicitOperator>(prob, *gpu_ctx);
        r = gpu_solver->solveExplicit(prob, *gpu_op_expl, Y, gpu_params);
      }
      x_next = r.x;
      f_next = r.f;
      n_inner = static_cast<int>(r.objective_values.size());
    } else
#endif
    {
      auto result = VarPro::solveProblem(prob, Y, /*verbose=*/false);
      x_next = result.x;
      f_next = result.f;
      n_inner = static_cast<int>(result.objective_values.size());
    }
    Y = x_next;
    total_inner += n_inner;
    final_inner_cost = f_next;

    // Machine-readable per-outer-iteration trace. `t` is cumulative wall
    // seconds from the same origin as IRLS_RESULT's total_s, less dump I/O,
    // so the last IRLS_ITER t matches the published wall time.
    const double t_now =
        Sec(Clock::now() - t_total_start).count() - dump_overhead_s;
    std::cout << "IRLS_ITER"
              << " k=" << outer_iter
              << " t=" << t_now
              << " c2_eff=" << c2_eff
              << " robust_cost=" << robust_cost
              << " inner_cost=" << f_next
              << " inner_iters=" << n_inner
              << "\n";

    // Dump this iterate so ATE can be computed per iteration offline. Uses
    // the same extractPoses path as the final TUM write, so iterate k here is
    // exactly what the solver would have returned had it stopped at k.
    if (!dump_dir.empty()) {
      const auto t_dump0 = Clock::now();
      std::ostringstream name;
      name << dump_dir << "/iter_" << std::setw(3) << std::setfill('0')
           << outer_iter << ".tum";
      writeTum(name.str(), pose_map, extractPoses(prob, Y));
      dump_overhead_s += Sec(Clock::now() - t_dump0).count();
    }

    const bool at_target = c2_eff <= c2 * (1.0 + 1e-9);
    if (verbose) {
      std::cout << "  iter " << outer_iter
                << " c2_eff=" << c2_eff
                << " robust_cost=" << robust_cost
                << " inner_cost=" << f_next
                << " inner_iters=" << n_inner << "\n";
    }

    final_robust_cost = robust_cost;

    // Convergence only after we've reached the target c²: the early-GNC
    // robust cost is computed with c²_eff ≠ c², so it's not comparable to
    // later iterations.
    if (at_target) {
      const double denom = std::max(std::abs(prev_robust_cost), 1.0);
      const double rel_change = std::abs(prev_robust_cost - robust_cost) / denom;
      if (outer_iter > 0 && rel_change < rel_tol) {
        ++outer_iter;
        break;
      }
      prev_robust_cost = robust_cost;
    } else {
      // Still annealing — shrink c²_eff toward the target.
      c2_eff = std::max(c2_eff / gnc_shrink, c2);
      // Don't track prev_robust_cost across c² changes (different objective).
      prev_robust_cost = std::numeric_limits<double>::infinity();
    }
  }

  // Write the final trajectory using the *converged* iterate (cov state in
  // prob doesn't affect pose extraction).
  auto final_poses = extractPoses(prob, Y);
  writeTum(final_tum, pose_map, final_poses);

  const double total_s = Sec(Clock::now() - t_total_start).count();

  std::cout << "wrote " << init_tum << " (odom init)\n"
            << "wrote " << final_tum << " (post-IRLS)\n"
            << "  outer iterations: " << outer_iter << "\n"
            << "  total inner iters: " << total_inner << "\n"
            << "  final robust cost: " << final_robust_cost << "\n"
            << "  final inner cost:  " << final_inner_cost << "\n"
            << "  total wall time:   " << total_s << " s\n";
  std::cout << "IRLS_RESULT"
            << " kernel=" << kernelName(kernel)
            << " form=" << formName(form)
            << " gnc=" << (gnc ? 1 : 0)
            << " init_seed=" << init_seed
            << " dim=" << prob.dim()
            << " outer=" << outer_iter
            << " inner=" << total_inner
            << " final_cost=" << final_inner_cost
            << " robust_cost=" << final_robust_cost
            << " total_s=" << total_s
            << " precompute_s=" << first_precompute_s
            << " total_precompute_s=" << total_precompute_s
            << " dense_gb=" << dense_gb
            << "\n";
  return 0;
}
