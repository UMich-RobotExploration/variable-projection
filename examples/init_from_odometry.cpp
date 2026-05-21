/**
 * @file init_from_odometry.cpp
 * @brief Build an initial pose trajectory by chaining sequential pose-pose
 *        (odometry) edges, then optionally run the VarPro solver from that
 *        init and export both the init and final trajectories as TUM files.
 *
 * The "sequential" check: an edge (s1, s2) is treated as odometry iff the two
 * symbols share the same prefix character and index(s2) == index(s1) + 1.
 * Non-sequential edges (loop closures, e.g. (A0, A50)) are ignored — this is
 * the cheap dead-reckoning initialization the solver starts from.
 *
 * Usage:
 *   ./init_from_odometry <pyfg> <out_init.tum> [--final <out_final.tum>]
 *
 * With `--final`, the solver is invoked at rank = problem dim (so the
 * relaxation collapses back to SO(d) × R^d, matching the input trajectory's
 * native dimension). The Implicit formulation is used; translations are
 * recovered analytically via the Schur complement after the solve.
 *
 * TUM format (one pose per line):
 *   <timestamp> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
 *
 * timestamp = pose index. Works for both 2D and 3D problems — 2D rotations
 * are embedded as a rotation about z and 2D translations get z = 0.
 */

#include <VarPro/Problem.h>
#include <VarPro/PyfgTextParser.h>
#include <VarPro/Solver.h>

#include <Eigen/Geometry>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct PoseSE3 {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Lift a measurement to SE(3) so we can store everything in one map regardless
// of whether the source problem is 2D or 3D.
PoseSE3 liftMeasurement(const VarPro::Matrix& R, const VarPro::Vector& t) {
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
    throw std::runtime_error("liftMeasurement: only 2D and 3D supported");
  }
  return out;
}

PoseSE3 compose(const PoseSE3& a, const PoseSE3& b) {
  PoseSE3 c;
  c.R = a.R * b.R;
  c.t = a.R * b.t + a.t;
  return c;
}

bool isSequentialOdom(const VarPro::Symbol& s1, const VarPro::Symbol& s2) {
  return s1.chr() == s2.chr() && s2.index() == s1.index() + 1;
}

std::unordered_map<VarPro::Key, PoseSE3> chainOdometry(
    const VarPro::Problem& prob, int& n_odom, int& n_skipped) {
  const auto pose_map = prob.getPoseSymbolMap();
  std::map<VarPro::Symbol, std::pair<VarPro::Symbol, PoseSE3>> next_edge;
  n_odom = 0;
  n_skipped = 0;
  for (const auto& m : prob.getRPMs()) {
    if (isSequentialOdom(m.first_id, m.second_id)) {
      next_edge.emplace(m.first_id,
                         std::make_pair(m.second_id, liftMeasurement(m.R, m.t)));
      ++n_odom;
    } else {
      ++n_skipped;
    }
  }
  std::unordered_map<VarPro::Key, PoseSE3> global;
  for (const auto& [sym, _idx] : pose_map) {
    const VarPro::Key key = sym.key();
    if (global.count(key)) continue;
    global[key] = PoseSE3{};
    VarPro::Symbol cur = sym;
    while (true) {
      auto it = next_edge.find(cur);
      if (it == next_edge.end()) break;
      const VarPro::Symbol& nxt = it->second.first;
      const PoseSE3& Tij = it->second.second;
      if (!pose_map.count(nxt) || global.count(nxt.key())) break;
      global[nxt.key()] = compose(global[cur.key()], Tij);
      cur = nxt;
    }
  }
  return global;
}

void writeTumFromPoses(
    const std::string& path,
    const std::map<VarPro::Symbol, int>& pose_map,
    const std::unordered_map<VarPro::Key, PoseSE3>& global) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("could not open " + path + " for writing");
  out << std::fixed;
  out.precision(9);
  for (const auto& [sym, idx] : pose_map) {
    auto it = global.find(sym.key());
    if (it == global.end()) continue;
    const PoseSE3& T = it->second;
    Eigen::Quaterniond q(T.R);
    q.normalize();
    out << idx << ' '
        << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' '
        << q.x()    << ' ' << q.y()   << ' ' << q.z()   << ' ' << q.w()
        << '\n';
  }
}

// Build the rank-dim variable matrix Y consumed by the Implicit solver from a
// per-pose SE(d) trajectory. Range-bearing unit vectors (RA-SLAM/SNL) get
// random unit-norm rows since odometry says nothing about them.
VarPro::Matrix buildVariableFromPoses(
    const VarPro::Problem& prob,
    const std::unordered_map<VarPro::Key, PoseSE3>& global) {
  const int d = prob.dim();
  const int n_poses = prob.numPoses();
  const int n_ranges = prob.numRangeMeasurements();
  const int n_rows = n_poses * d + n_ranges;  // = rotAndRangeMatrixSize()
  VarPro::Matrix Y = VarPro::Matrix::Zero(n_rows, d);

  for (const auto& [sym, idx] : prob.getPoseSymbolMap()) {
    auto it = global.find(sym.key());
    if (it == global.end()) continue;
    const PoseSE3& T = it->second;
    // Stored layout: Y.block(idx*d, 0, d, d) == R^T  (see getRotationFromSymbol).
    if (d == 3) {
      Y.block(idx * d, 0, d, d) = T.R.transpose();
    } else {  // d == 2
      Y.block(idx * d, 0, d, d) = T.R.topLeftCorner<2, 2>().transpose();
    }
  }

  // Random unit-norm bearings for the range block. Bearing rows are 1 x rank.
  if (n_ranges > 0) {
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<double> N(0.0, 1.0);
    for (int i = 0; i < n_ranges; ++i) {
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

// After the Implicit solve, recover global poses (R_i, t_i) from the final
// variable matrix Y by computing the analytic translation update and reading
// each pose's rotation block.
std::unordered_map<VarPro::Key, PoseSE3> posesFromVariable(
    const VarPro::Problem& prob, const VarPro::Matrix& Y) {
  const int d = prob.dim();
  VarPro::Matrix Xfull = prob.getTranslationExplicitSolution(Y);
  const int trans_offset = prob.rotAndRangeMatrixSize();

  std::unordered_map<VarPro::Key, PoseSE3> global;
  for (const auto& [sym, idx] : prob.getPoseSymbolMap()) {
    PoseSE3 T;
    VarPro::Matrix Rblock = Xfull.block(idx * d, 0, d, d);  // R^T per layout
    if (d == 3) {
      T.R = Rblock.transpose();
    } else {
      T.R.setIdentity();
      T.R.topLeftCorner<2, 2>() = Rblock.transpose();
    }
    VarPro::Matrix trow = Xfull.block(trans_offset + idx, 0, 1, d);  // 1 x d
    if (d == 3) {
      T.t = Eigen::Vector3d(trow(0, 0), trow(0, 1), trow(0, 2));
    } else {
      T.t = Eigen::Vector3d(trow(0, 0), trow(0, 1), 0.0);
    }
    global[sym.key()] = T;
  }
  return global;
}

}  // namespace

int main(int argc, char** argv) {
  std::string pyfg;
  std::string init_tum;
  std::string final_tum;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--final" && i + 1 < argc) {
      final_tum = argv[++i];
    } else if (pyfg.empty()) {
      pyfg = a;
    } else if (init_tum.empty()) {
      init_tum = a;
    }
  }
  if (pyfg.empty() || init_tum.empty()) {
    std::cerr << "usage: " << argv[0]
              << " <pyfg> <out_init.tum> [--final <out_final.tum>]\n";
    return 1;
  }

  VarPro::Problem prob = VarPro::parsePyfgTextToProblem(pyfg);
  prob.updateProblemData();
  const auto pose_map = prob.getPoseSymbolMap();
  if (pose_map.empty()) {
    std::cerr << "no pose vertices in " << pyfg << "\n";
    return 1;
  }

  int n_odom = 0, n_skipped = 0;
  auto init_global = chainOdometry(prob, n_odom, n_skipped);
  writeTumFromPoses(init_tum, pose_map, init_global);

  std::cout << "wrote " << init_tum << " (odometry init)\n"
            << "  dim:    " << prob.dim() << "\n"
            << "  poses:  " << pose_map.size() << "\n"
            << "  odom edges chained:           " << n_odom << "\n"
            << "  non-sequential edges skipped: " << n_skipped << "\n";

  if (final_tum.empty()) return 0;

  // ------------------------------------------------------------------
  // Optimize at rank = dim and write the final trajectory.
  // ------------------------------------------------------------------
  const int rank = prob.dim();
  prob.setFormulation(VarPro::Formulation::Implicit);
  prob.setRank(rank);
  // setFormulation/setRank don't re-run the precompute; only updateProblemData
  // does. The Implicit precompute happens inside updateProblemData above.

  VarPro::Matrix Y0 = buildVariableFromPoses(prob, init_global);
  Y0 = prob.projectToManifold(Y0);

  const double f0 = prob.evaluateObjective(Y0);
  std::cout << "starting objective at odom init: " << f0 << "\n";
  auto result = VarPro::solveProblem(prob, Y0, /*verbose=*/false);
  std::cout << "final objective:                 " << result.f << "\n";
  std::cout << "outer iterations:                " << result.objective_values.size() << "\n";

  auto final_global = posesFromVariable(prob, result.x);
  writeTumFromPoses(final_tum, pose_map, final_global);
  std::cout << "wrote " << final_tum << " (post-solve)\n";
  return 0;
}
