/**
 * @file gtsam_gnc_pgo.cpp
 * @brief GTSAM baseline for PGO with Graduated Non-Convexity + Geman-McClure
 *        (GNC-GM), built to be a drop-in apples-to-apples comparison against
 *        irls_robust.cpp on the same pyfg datasets and from the same odometry
 *        init.
 *
 * Both 2D (EDGE_SE2 / VERTEX_SE2) and 3D (EDGE_SE3:QUAT / VERTEX_SE3:QUAT)
 * datasets are handled. Loop closures and odometry edges are added as
 * BetweenFactor<Pose2|Pose3> with full-covariance Gaussian noise; the first
 * vertex in file order is anchored with a tight prior (the pyfg format doesn't
 * carry a gauge).
 *
 * The init is built by chaining sequential pose-pose edges (i, i+1 along the
 * same symbol prefix), matching irls_robust.cpp::chainOdometry exactly so the
 * GTSAM and VarPro runs start from byte-identical trajectories. The optional
 * --init-seed knob also matches: each odometry edge is perturbed by a per-seed
 * SE(d) random sample before the chain composition.
 *
 * Output line (machine-readable, last line of stdout):
 *   GTSAM_GNC_RESULT loss=gm dim=<d> outer=<n> inner=<m> mu_final=<mu>
 *                    inliers=<k> outliers=<k> final_cost=<c> initial_cost=<c0>
 *                    total_s=<t>
 *
 * Usage:
 *   ./gtsam_gnc_pgo <pyfg> <init.tum> <final.tum>
 *                   [--barc-prob P]   (chi-square inlier confidence, default 0.99)
 *                   [--mu-step S]     (default 1.4)
 *                   [--max-iters K]   (default 100)
 *                   [--rel-tol T]     (default 1e-5)
 *                   [--init-seed S]
 *                   [--init-noise-rot-deg X] (default 2.0)
 *                   [--init-noise-trans X]   (default 0.05)
 *                   [--verbose]
 */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// ---------------------------------------------------------------------------
// pyfg parsing (PGO-only: VERTEX_SE2/SE3 + EDGE_SE2/SE3:QUAT)
// ---------------------------------------------------------------------------

struct VertexRecord {
  std::string name;   // e.g. "A0"
  int file_index;     // order in which it appeared
};

struct EdgeRecord {
  int dim;            // 2 or 3
  std::string from;
  std::string to;
  // 3D pose: full SE(3) measurement; 2D pose lives in the (x, y) plane.
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  // 6x6 covariance in pyfg ordering: [t_x, t_y, t_z, r_x, r_y, r_z]
  // (for 2D: 3x3 in [x, y, theta]).
  Eigen::MatrixXd cov;
};

// VERTEX_SE{2,3}:[QUAT:]PRIOR — same fields as an edge, but unary.
struct PriorRecord {
  int dim;
  std::string name;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  Eigen::MatrixXd cov;
};

struct PoseSE3 {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Pure-syntactic odom detection — must match irls_robust.cpp::isSequentialOdom.
// Returns the (char, index) pair for a symbol string like "A12"; throws if it
// doesn't fit that shape.
std::pair<char, int> parseSym(const std::string &s) {
  if (s.empty()) throw std::runtime_error("empty symbol");
  char c = s[0];
  if (!std::isalpha(static_cast<unsigned char>(c)))
    throw std::runtime_error("symbol must start with letter: " + s);
  int idx = std::stoi(s.substr(1));
  return {c, idx};
}

bool isSequentialOdom(const std::string &a, const std::string &b) {
  auto [ca, ia] = parseSym(a);
  auto [cb, ib] = parseSym(b);
  return ca == cb && ib == ia + 1;
}

Eigen::Matrix3d quatToR(double qx, double qy, double qz, double qw) {
  Eigen::Quaterniond q(qw, qx, qy, qz);
  q.normalize();
  return q.toRotationMatrix();
}

Eigen::Matrix3d angleToR2(double theta) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  R.topLeftCorner<2, 2>() << std::cos(theta), -std::sin(theta),
                              std::sin(theta),  std::cos(theta);
  return R;
}

Eigen::MatrixXd readSymmetric(std::istringstream &iss, int n) {
  Eigen::MatrixXd M(n, n);
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      double v;
      if (!(iss >> v))
        throw std::runtime_error("malformed covariance block");
      M(i, j) = v;
      M(j, i) = v;
    }
  }
  return M;
}

struct PyfgGraph {
  int dim = 0;                                   // 2 or 3 (first vertex wins)
  std::vector<VertexRecord> vertices;            // file order
  std::unordered_map<std::string, int> name_to_idx;
  std::vector<EdgeRecord> edges;
  std::vector<PriorRecord> priors;
};

PyfgGraph parsePyfg(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open pyfg: " + path);
  PyfgGraph g;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    std::string tag;
    iss >> tag;
    if (tag == "VERTEX_SE2") {
      double ts;
      std::string name;
      iss >> ts >> name;
      if (g.name_to_idx.count(name)) continue;
      g.name_to_idx[name] = static_cast<int>(g.vertices.size());
      g.vertices.push_back({name, static_cast<int>(g.vertices.size())});
      if (g.dim == 0) g.dim = 2;
      else if (g.dim != 2) throw std::runtime_error("mixed 2D/3D vertices");
    } else if (tag == "VERTEX_SE3:QUAT") {
      double ts;
      std::string name;
      iss >> ts >> name;
      if (g.name_to_idx.count(name)) continue;
      g.name_to_idx[name] = static_cast<int>(g.vertices.size());
      g.vertices.push_back({name, static_cast<int>(g.vertices.size())});
      if (g.dim == 0) g.dim = 3;
      else if (g.dim != 3) throw std::runtime_error("mixed 2D/3D vertices");
    } else if (tag == "EDGE_SE2") {
      double ts;
      std::string a, b;
      iss >> ts >> a >> b;
      double dx, dy, dth;
      iss >> dx >> dy >> dth;
      EdgeRecord e;
      e.dim = 2;
      e.from = a;
      e.to = b;
      e.R = angleToR2(dth);
      e.t = Eigen::Vector3d(dx, dy, 0.0);
      e.cov = readSymmetric(iss, 3);
      g.edges.push_back(std::move(e));
    } else if (tag == "EDGE_SE3:QUAT") {
      double ts;
      std::string a, b;
      iss >> ts >> a >> b;
      double dx, dy, dz, qx, qy, qz, qw;
      iss >> dx >> dy >> dz >> qx >> qy >> qz >> qw;
      EdgeRecord e;
      e.dim = 3;
      e.from = a;
      e.to = b;
      e.R = quatToR(qx, qy, qz, qw);
      e.t = Eigen::Vector3d(dx, dy, dz);
      e.cov = readSymmetric(iss, 6);
      g.edges.push_back(std::move(e));
    } else if (tag == "VERTEX_SE3:QUAT:PRIOR") {
      double ts;
      std::string name;
      iss >> ts >> name;
      double dx, dy, dz, qx, qy, qz, qw;
      iss >> dx >> dy >> dz >> qx >> qy >> qz >> qw;
      PriorRecord p;
      p.dim = 3;
      p.name = name;
      p.R = quatToR(qx, qy, qz, qw);
      p.t = Eigen::Vector3d(dx, dy, dz);
      p.cov = readSymmetric(iss, 6);
      g.priors.push_back(std::move(p));
    } else if (tag == "VERTEX_SE2:PRIOR") {
      double ts;
      std::string name;
      iss >> ts >> name;
      double dx, dy, dth;
      iss >> dx >> dy >> dth;
      PriorRecord p;
      p.dim = 2;
      p.name = name;
      p.R = angleToR2(dth);
      p.t = Eigen::Vector3d(dx, dy, 0.0);
      p.cov = readSymmetric(iss, 3);
      g.priors.push_back(std::move(p));
    }
    // Silently skip landmarks / ranges for the PGO-only baseline.
  }
  if (g.dim == 0)
    throw std::runtime_error("no SE2/SE3 vertices parsed from " + path);
  return g;
}

// ---------------------------------------------------------------------------
// Odom init (must match irls_robust.cpp::chainOdometry)
// ---------------------------------------------------------------------------

PoseSE3 compose(const PoseSE3 &a, const PoseSE3 &b) {
  PoseSE3 c;
  c.R = a.R * b.R;
  c.t = a.R * b.t + a.t;
  return c;
}

PoseSE3 sampleEdgePerturbation(std::mt19937 &rng, int dim,
                                double sigma_rot_rad, double sigma_trans) {
  PoseSE3 dT;
  std::normal_distribution<double> N(0.0, 1.0);
  if (dim == 3) {
    Eigen::Vector3d w(sigma_rot_rad * N(rng),
                       sigma_rot_rad * N(rng),
                       sigma_rot_rad * N(rng));
    const double theta = w.norm();
    if (theta < 1e-12) dT.R.setIdentity();
    else dT.R = Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
    dT.t = Eigen::Vector3d(sigma_trans * N(rng),
                            sigma_trans * N(rng),
                            sigma_trans * N(rng));
  } else {
    const double theta = sigma_rot_rad * N(rng);
    dT.R = angleToR2(theta);
    dT.t = Eigen::Vector3d(sigma_trans * N(rng), sigma_trans * N(rng), 0.0);
  }
  return dT;
}

std::unordered_map<std::string, PoseSE3>
chainOdometry(const PyfgGraph &g, int &n_odom, int &n_skipped,
              int init_seed, double sigma_rot_rad, double sigma_trans) {
  // Mirror the std::map<Symbol, ...> ordering used by VarPro by keying on
  // (char, idx) — the perturbation traversal order must match exactly.
  std::map<std::pair<char, int>, std::pair<std::string, PoseSE3>> next_edge;
  n_odom = 0;
  n_skipped = 0;
  for (const auto &e : g.edges) {
    if (isSequentialOdom(e.from, e.to)) {
      PoseSE3 T{e.R, e.t};
      next_edge.emplace(parseSym(e.from), std::make_pair(e.to, T));
      ++n_odom;
    } else {
      ++n_skipped;
    }
  }
  if (init_seed > 0 && (sigma_rot_rad > 0.0 || sigma_trans > 0.0)) {
    std::mt19937 rng(static_cast<uint32_t>(init_seed));
    const int dim = g.dim;
    for (auto &kv : next_edge) {
      PoseSE3 dT = sampleEdgePerturbation(rng, dim, sigma_rot_rad, sigma_trans);
      kv.second.second = compose(kv.second.second, dT);
    }
  }
  std::unordered_map<std::string, PoseSE3> global;
  // Walk the chains in the same order VarPro does (std::map<Symbol, idx>
  // iteration is keyed by (char, idx), which matches our parseSym keys).
  std::map<std::pair<char, int>, std::string> ordered_names;
  for (const auto &v : g.vertices) ordered_names[parseSym(v.name)] = v.name;
  for (const auto &[sym, name] : ordered_names) {
    if (global.count(name)) continue;
    global[name] = PoseSE3{};
    std::pair<char, int> cur = sym;
    std::string cur_name = name;
    while (true) {
      auto it = next_edge.find(cur);
      if (it == next_edge.end()) break;
      const std::string &nxt_name = it->second.first;
      const PoseSE3 &Tij = it->second.second;
      if (!g.name_to_idx.count(nxt_name) || global.count(nxt_name)) break;
      global[nxt_name] = compose(global[cur_name], Tij);
      cur = parseSym(nxt_name);
      cur_name = nxt_name;
    }
  }
  return global;
}

// ---------------------------------------------------------------------------
// TUM output (identical schema to irls_robust.cpp::writeTum)
// ---------------------------------------------------------------------------

void writeTum(const std::string &path, const PyfgGraph &g,
               const std::unordered_map<std::string, PoseSE3> &global) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open " + path);
  out << std::fixed;
  out.precision(9);
  // VarPro orders by std::map<Symbol, idx>, which sorts on the parsed
  // (char, idx) pair — replicate that here.
  std::map<std::pair<char, int>, std::pair<int, std::string>> ordered;
  for (const auto &v : g.vertices)
    ordered[parseSym(v.name)] = {v.file_index, v.name};
  for (const auto &[sym, p] : ordered) {
    auto it = global.find(p.second);
    if (it == global.end()) continue;
    const PoseSE3 &T = it->second;
    Eigen::Quaterniond q(T.R);
    q.normalize();
    out << p.first << ' '
        << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' '
        << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
  }
}

// ---------------------------------------------------------------------------
// GTSAM bridge
// ---------------------------------------------------------------------------

// Convert pyfg 6x6 covariance ([t, r] block order) to GTSAM Pose3 ordering
// ([r, t]) by swapping the 3x3 blocks. GTSAM's BetweenFactor<Pose3> linearizes
// errors as (rotation, translation), so the noise model must match.
Eigen::MatrixXd reorderPose3Cov(const Eigen::MatrixXd &cov_tr) {
  Eigen::MatrixXd cov(6, 6);
  // Permutation P = [r-block, t-block] from [t-block, r-block].
  Eigen::PermutationMatrix<6> P;
  P.indices() << 3, 4, 5, 0, 1, 2;
  cov = P * cov_tr * P.transpose();
  return cov;
}

gtsam::Key keyForName(const std::string &name) {
  // Use the GTSAM Symbol packing: (uppercase char, index). pyfg symbols are
  // already single-letter-prefixed names like "A12" / "L7".
  return gtsam::Symbol(name[0], static_cast<std::uint64_t>(std::stoll(name.substr(1))));
}

gtsam::Pose2 poseToPose2(const PoseSE3 &T) {
  const double theta = std::atan2(T.R(1, 0), T.R(0, 0));
  return gtsam::Pose2(T.t.x(), T.t.y(), theta);
}

gtsam::Pose3 poseToPose3(const PoseSE3 &T) {
  return gtsam::Pose3(gtsam::Rot3(T.R), gtsam::Point3(T.t));
}

PoseSE3 pose2ToPose(const gtsam::Pose2 &p) {
  PoseSE3 T;
  T.R = angleToR2(p.theta());
  T.t = Eigen::Vector3d(p.x(), p.y(), 0.0);
  return T;
}

PoseSE3 pose3ToPose(const gtsam::Pose3 &p) {
  PoseSE3 T;
  T.R = p.rotation().matrix();
  T.t = p.translation();
  return T;
}

}  // namespace

int main(int argc, char **argv) {
  std::string pyfg_path, init_tum, final_tum;
  double barc_prob = 0.99;
  double mu_step = 1.4;
  int max_iters = 100;
  double rel_tol = 1e-5;
  int init_seed = 0;
  double init_noise_rot_deg = 2.0;
  double init_noise_trans = 0.05;
  bool verbose = false;
  bool strip_priors = false;
  bool warm_start_lm = false;  // run plain LM once before robust kernel kicks in
  bool no_gnc = false;         // skip GNC entirely (debug: plain LM)
  bool use_gtsam_gnc = false;  // use GTSAM's GncOptimizer instead of the hand
                                // rolled IRLS-GM loop (kept for comparison; fails
                                // on high-outlier multi-robot data because its
                                // first weight is computed from a post-LM iterate
                                // — see irls_robust.cpp for the reference loop)
  double c2 = 25.0;            // GM kernel inlier threshold (squared whitened
                                // residual) — same default as irls_robust.cpp
  double gnc_init_mul = 64.0;  // initial c² multiplier when --gnc is on
  double gnc_shrink = 1.4;     // per-iter c² shrink factor
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--barc-prob" && i + 1 < argc) barc_prob = std::stod(argv[++i]);
    else if (a == "--mu-step" && i + 1 < argc) mu_step = std::stod(argv[++i]);
    else if (a == "--max-iters" && i + 1 < argc) max_iters = std::stoi(argv[++i]);
    else if (a == "--rel-tol" && i + 1 < argc) rel_tol = std::stod(argv[++i]);
    else if (a == "--init-seed" && i + 1 < argc) init_seed = std::stoi(argv[++i]);
    else if (a == "--init-noise-rot-deg" && i + 1 < argc)
      init_noise_rot_deg = std::stod(argv[++i]);
    else if (a == "--init-noise-trans" && i + 1 < argc)
      init_noise_trans = std::stod(argv[++i]);
    else if (a == "--verbose") verbose = true;
    else if (a == "--strip-priors") strip_priors = true;
    else if (a == "--warm-start") warm_start_lm = true;
    else if (a == "--no-gnc") no_gnc = true;
    else if (a == "--use-gtsam-gnc") use_gtsam_gnc = true;
    else if (a == "--c2" && i + 1 < argc) c2 = std::stod(argv[++i]);
    else if (a == "--gnc-init-mul" && i + 1 < argc)
      gnc_init_mul = std::stod(argv[++i]);
    else if (a == "--gnc-shrink" && i + 1 < argc)
      gnc_shrink = std::stod(argv[++i]);
    else positional.push_back(a);
  }
  if (positional.size() != 3) {
    std::cerr << "usage: " << argv[0]
              << " <pyfg> <init.tum> <final.tum> [--barc-prob P] "
                 "[--mu-step S] [--max-iters K] [--rel-tol T] "
                 "[--init-seed S] [--init-noise-rot-deg X] "
                 "[--init-noise-trans X] [--verbose]\n";
    return 1;
  }
  pyfg_path = positional[0];
  init_tum  = positional[1];
  final_tum = positional[2];

  const auto t_total_start = Clock::now();

  PyfgGraph g = parsePyfg(pyfg_path);
  const int dim = g.dim;

  // Odom init — byte-identical to irls_robust.cpp.
  int n_odom = 0, n_skipped = 0;
  const double sigma_rot_rad = init_noise_rot_deg * M_PI / 180.0;
  auto init_poses = chainOdometry(g, n_odom, n_skipped, init_seed,
                                    sigma_rot_rad, init_noise_trans);
  writeTum(init_tum, g, init_poses);

  // Build NonlinearFactorGraph + initial Values.
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;

  for (const auto &v : g.vertices) {
    auto it = init_poses.find(v.name);
    if (it == init_poses.end()) continue;
    gtsam::Key k = keyForName(v.name);
    if (dim == 2) initial.insert(k, poseToPose2(it->second));
    else          initial.insert(k, poseToPose3(it->second));
  }

  // Add priors. Multi-robot pyfg files (cosmobench, nebula) declare one
  // VERTEX_*:PRIOR per robot, which is what gauges the inter-robot graph.
  // Without those, every robot's first pose starts at identity (since the odom
  // chain initializes each unvisited symbol there) and LM cannot disambiguate
  // the multi-robot gauge from loop closures alone. If --strip-priors is set,
  // we honor the pre-strip behaviour for parity with the VarPro IRLS sweep,
  // which strips them because VarPro's solver crashes on tight covariances.
  int n_priors_added = 0;
  if (!strip_priors) {
    for (const auto &p : g.priors) {
      if (!g.name_to_idx.count(p.name)) continue;
      gtsam::Key k = keyForName(p.name);
      if (p.dim == 2) {
        auto noise = gtsam::noiseModel::Gaussian::Covariance(p.cov);
        graph.add(gtsam::PriorFactor<gtsam::Pose2>(
            k, gtsam::Pose2(p.t.x(), p.t.y(),
                             std::atan2(p.R(1, 0), p.R(0, 0))),
            noise));
      } else {
        auto noise = gtsam::noiseModel::Gaussian::Covariance(reorderPose3Cov(p.cov));
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            k, gtsam::Pose3(gtsam::Rot3(p.R), gtsam::Point3(p.t)), noise));
      }
      ++n_priors_added;
    }
  }
  // Fall back: if the dataset declared no priors AND --strip-priors was *not*
  // requested (e.g. plain g2o-style PGO files in examples/data/pgo/), anchor
  // the first vertex tightly to fix the gauge. When --strip-priors is set the
  // user explicitly wants zero priors — matching what VarPro IRLS sees, which
  // handles the rank-deficient Hessian via SPQR/pseudoinverse internally.
  if (!strip_priors && n_priors_added == 0 && !g.vertices.empty()) {
    const auto &v0 = g.vertices.front();
    gtsam::Key k0 = keyForName(v0.name);
    auto p0_it = init_poses.find(v0.name);
    if (p0_it == init_poses.end())
      throw std::runtime_error("first vertex has no init pose");
    if (dim == 2) {
      auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(3) << 1e-6, 1e-6, 1e-8).finished());
      graph.add(gtsam::PriorFactor<gtsam::Pose2>(
          k0, poseToPose2(p0_it->second), prior_noise));
    } else {
      auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << 1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6).finished());
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(
          k0, poseToPose3(p0_it->second), prior_noise));
    }
  }

  // BetweenFactors for every edge. We track each edge factor's index so the
  // IRLS loop can re-weight just the edges (priors keep weight = 1).
  std::vector<size_t> edge_indices;
  edge_indices.reserve(g.edges.size());
  for (const auto &e : g.edges) {
    gtsam::Key ka = keyForName(e.from);
    gtsam::Key kb = keyForName(e.to);
    edge_indices.push_back(graph.size());
    if (e.dim == 2) {
      auto noise = gtsam::noiseModel::Gaussian::Covariance(e.cov);
      gtsam::Pose2 meas(e.t.x(), e.t.y(), std::atan2(e.R(1, 0), e.R(0, 0)));
      graph.add(gtsam::BetweenFactor<gtsam::Pose2>(ka, kb, meas, noise));
    } else {
      auto noise = gtsam::noiseModel::Gaussian::Covariance(reorderPose3Cov(e.cov));
      gtsam::Pose3 meas(gtsam::Rot3(e.R), gtsam::Point3(e.t));
      graph.add(gtsam::BetweenFactor<gtsam::Pose3>(ka, kb, meas, noise));
    }
  }

  const double initial_cost = graph.error(initial);

  // Optional plain-LM warm start. Standard GTSAM PGO pipeline solves a single
  // unweighted LM pass from odom before any robust kernel; useful as an apples-
  // to-apples knob, off by default to preserve "GNC straight from odom".
  if (warm_start_lm) {
    gtsam::LevenbergMarquardtParams pre_params;
    pre_params.setMaxIterations(200);
    if (verbose) pre_params.setVerbosityLM("SUMMARY");
    gtsam::LevenbergMarquardtOptimizer pre(graph, initial, pre_params);
    initial = pre.optimize();
    if (verbose) std::cout << "  warm-start LM cost: " << graph.error(initial) << "\n";
  }

  // GNC parameters.
  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setMaxIterations(200);
  lm_params.setRelativeErrorTol(1e-7);
  lm_params.setAbsoluteErrorTol(1e-7);
  if (verbose) lm_params.setVerbosityLM("SUMMARY");

  gtsam::GncParams<gtsam::LevenbergMarquardtParams> gnc_params(lm_params);
  gnc_params.setLossType(gtsam::GncLossType::GM);
  // For BetweenFactor<Pose2>/<Pose3>, the whitened error has dofs = dim of the
  // pose (3 or 6). barcSq defaults to chi-square inverse at barc_prob with that
  // many dofs, which is what we want.
  gnc_params.setMuStep(mu_step);
  gnc_params.setRelativeCostTol(rel_tol);
  gnc_params.maxIterations = static_cast<size_t>(max_iters);
  if (verbose)
    gnc_params.setVerbosityGNC(
        gtsam::GncParams<gtsam::LevenbergMarquardtParams>::Verbosity::SUMMARY);

  // -------------------------------------------------------------------------
  // Robust solve.
  //
  // Default path = hand-rolled IRLS-GM with GNC c²-annealing, modeled directly
  // on irls_robust.cpp::main (the VarPro-side baseline). The key fix vs.
  // gtsam::GncOptimizer<LM> is that the **first outer iter computes weights
  // from the odom-init residuals**, not from a post-LM-corrupted iterate.
  // GncOptimizer does the latter, which silently destroys multi-robot
  // cosmobench data: vanilla LM from odom fits the 10% outliers and pulls the
  // trajectory ~100m off-track before GNC ever sees a weight update.
  //
  // --use-gtsam-gnc reactivates the broken-on-cosmobench reference path.
  // -------------------------------------------------------------------------
  gtsam::Values result;
  gtsam::Vector w;
  int outer_iters_done = 0;
  int total_inner_iters = 0;
  double final_robust_cost = std::numeric_limits<double>::infinity();

  if (no_gnc) {
    gtsam::LevenbergMarquardtOptimizer plain(graph, initial, lm_params);
    result = plain.optimize();
    w = gtsam::Vector::Ones(graph.size());
  } else if (use_gtsam_gnc) {
    gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams>>
        gnc(graph, initial, gnc_params);
    gnc.setInlierCostThresholdsAtProbability(barc_prob);
    result = gnc.optimize();
    w = gnc.getWeights();
  } else {
    // --- IRLS-GM with GNC c²-annealing -----------------------------------
    // Snapshot original (unweighted) noise models so we can rescale them per
    // outer iter without losing the source-of-truth Σ.
    std::vector<gtsam::SharedNoiseModel> orig_noises(graph.size());
    for (size_t i = 0; i < graph.size(); ++i) {
      auto nmf = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(graph[i]);
      if (nmf) orig_noises[i] = nmf->noiseModel();
    }

    // Edge-factor set (priors are not robustified — they encode the gauge and
    // should never be downweighted).
    std::vector<char> is_edge(graph.size(), 0);
    for (size_t ei : edge_indices) is_edge[ei] = 1;

    double c2_eff = c2 * gnc_init_mul;  // initial permissive threshold
    gtsam::Values current = initial;
    double prev_robust_cost = std::numeric_limits<double>::infinity();
    w = gtsam::Vector::Ones(graph.size());

    for (int outer = 0; outer < max_iters; ++outer) {
      // 1) Compute per-factor whitened squared residual against the *original*
      //    noise model, then apply the GM kernel at the current c²_eff.
      double robust_cost = 0.0;
      for (size_t i = 0; i < graph.size(); ++i) {
        if (!orig_noises[i] || !is_edge[i]) {
          w(i) = 1.0;
          continue;
        }
        auto nmf = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(graph[i]);
        if (!nmf) { w(i) = 1.0; continue; }
        // factor.error(values) is 0.5 * ||r||²_Σ ; the IRLS / GM kernel
        // operates on ||r||²_Σ itself, so multiply by 2.
        const double s = 2.0 * nmf->error(current);
        const double denom = c2_eff + s;
        const double w_i = (c2_eff * c2_eff) / (denom * denom);
        w(i) = std::max(w_i, 1e-12);   // TLS-style floor for numerical safety
        robust_cost += (c2_eff * s) / (c2_eff + s);  // ρ_GM(s; c²_eff)
      }

      // 2) Rebuild a weighted graph: noise covariance Σ_i ← Σ_i / w_i, i.e.
      //    information Λ_i ← w_i · Λ_i. Priors and non-edge factors keep their
      //    original noise model.
      gtsam::NonlinearFactorGraph weighted;
      weighted.resize(graph.size());
      for (size_t i = 0; i < graph.size(); ++i) {
        if (!orig_noises[i]) { weighted[i] = graph[i]; continue; }
        auto nmf = std::dynamic_pointer_cast<gtsam::NoiseModelFactor>(graph[i]);
        if (!nmf || !is_edge[i]) { weighted[i] = graph[i]; continue; }
        auto gauss = std::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
            orig_noises[i]);
        if (!gauss) { weighted[i] = graph[i]; continue; }
        gtsam::Matrix new_info = w(i) * gauss->information();
        auto new_noise = gtsam::noiseModel::Gaussian::Information(new_info);
        weighted[i] = nmf->cloneWithNewNoiseModel(new_noise);
      }

      // 3) Solve weighted LM from the current iterate. We keep `current`
      //    advancing each outer iter (warm-start the inner solve from the
      //    previous solution) — matches IRLS's behaviour.
      gtsam::LevenbergMarquardtOptimizer lm(weighted, current, lm_params);
      gtsam::Values next = lm.optimize();
      total_inner_iters += static_cast<int>(lm.iterations());
      current = next;

      const bool at_target = c2_eff <= c2 * (1.0 + 1e-9);
      if (verbose) {
        std::cout << "  iter " << outer
                  << " c2_eff=" << c2_eff
                  << " robust_cost=" << robust_cost
                  << " inner_cost=" << weighted.error(current)
                  << " inner_iters=" << lm.iterations() << "\n";
      }
      final_robust_cost = robust_cost;
      outer_iters_done = outer + 1;

      // 4) Convergence check only after c²_eff has annealed to the target;
      //    early-GNC robust_cost is on a different objective and not
      //    comparable across iters.
      if (at_target) {
        const double denom_t = std::max(std::abs(prev_robust_cost), 1.0);
        const double rel_change =
            std::abs(prev_robust_cost - robust_cost) / denom_t;
        if (outer > 0 && rel_change < rel_tol) break;
        prev_robust_cost = robust_cost;
      } else {
        c2_eff = std::max(c2_eff / gnc_shrink, c2);
        prev_robust_cost = std::numeric_limits<double>::infinity();
      }
    }
    result = current;
  }

  const double final_cost = graph.error(result);
  // Inlier/outlier count from final weights (threshold at 0.5).
  int inliers = 0, outliers = 0;
  for (int i = 0; i < w.size(); ++i) {
    if (w(i) >= 0.5) ++inliers; else ++outliers;
  }

  // Write final TUM.
  std::unordered_map<std::string, PoseSE3> final_poses;
  for (const auto &v : g.vertices) {
    gtsam::Key k = keyForName(v.name);
    if (!result.exists(k)) continue;
    if (dim == 2) final_poses[v.name] = pose2ToPose(result.at<gtsam::Pose2>(k));
    else          final_poses[v.name] = pose3ToPose(result.at<gtsam::Pose3>(k));
  }
  writeTum(final_tum, g, final_poses);

  const double total_s = Sec(Clock::now() - t_total_start).count();

  std::cout << "GTSAM-GNC loss=gm dim=" << dim
            << " poses=" << g.vertices.size()
            << " edges=" << g.edges.size()
            << " odom=" << n_odom
            << " loops=" << n_skipped << "\n";
  std::cout << "  initial cost: " << initial_cost << "\n"
            << "  final cost:   " << final_cost << "\n"
            << "  inliers/outliers: " << inliers << "/" << outliers << "\n"
            << "  wall time: " << total_s << " s\n";
  std::cout << "GTSAM_GNC_RESULT"
            << " loss=gm"
            << " dim=" << dim
            << " init_seed=" << init_seed
            << " edges=" << g.edges.size()
            << " inliers=" << inliers
            << " outliers=" << outliers
            << " outer=" << outer_iters_done
            << " inner=" << total_inner_iters
            << " initial_cost=" << initial_cost
            << " final_cost=" << final_cost
            << " robust_cost=" << final_robust_cost
            << " total_s=" << total_s
            << "\n";
  return 0;
}
