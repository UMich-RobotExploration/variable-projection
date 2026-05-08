//
// Created by Tim Magoun on 10/31/23.
//
#include <VarPro/PyfgTextParser.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <set>
//
#include <chrono>
#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace VarPro
{

  Matrix fromAngle(double angle_rad);
  Matrix fromQuat(double qx, double qy, double qz, double qw);

  Vector readVector(std::istringstream &iss, int dim);
  Scalar readScalar(std::istringstream &iss);
  Matrix readQuat(std::istringstream &iss);
  Matrix readSymmetric(std::istringstream &iss, int dim);

  /**
   * @brief Enum to keep track of the different types of items in a PyFG file
   *
   * @details This contains all of the types in the PyFG format that we currently
   * have support for.
   */
  enum PyFGType
  {
    POSE_TYPE_2D,
    POSE_TYPE_3D,
    POSE_PRIOR_2D,
    POSE_PRIOR_3D,
    LANDMARK_TYPE_2D,
    LANDMARK_TYPE_3D,
    LANDMARK_PRIOR_2D,
    LANDMARK_PRIOR_3D,
    REL_POSE_POSE_TYPE_2D,
    REL_POSE_POSE_TYPE_3D,
    REL_POSE_LANDMARK_TYPE_2D,
    REL_POSE_LANDMARK_TYPE_3D,
    RANGE_MEASURE_TYPE,
  };

  int getDimFromPyfgFirstLine(const std::string &filename)
  {
    // Check if the file exists and we can read it
    std::ifstream in_file(filename);
    if (!in_file.good())
    {
      throw std::runtime_error("Could not open file " + filename);
    }

    const std::map<std::string, PyFGType> PyFGStringToType{
        {"VERTEX_SE2", POSE_TYPE_2D},
        {"VERTEX_SE3:QUAT", POSE_TYPE_3D},
        {"VERTEX_SE2:PRIOR", POSE_PRIOR_2D},
        {"VERTEX_SE3:QUAT:PRIOR", POSE_PRIOR_3D},
        {"VERTEX_XY", LANDMARK_TYPE_2D},
        {"VERTEX_XYZ", LANDMARK_TYPE_3D},
        {"VERTEX_XY:PRIOR", LANDMARK_PRIOR_2D},
        {"VERTEX_XYZ:PRIOR", LANDMARK_PRIOR_3D},
        {"EDGE_SE2", REL_POSE_POSE_TYPE_2D},
        {"EDGE_SE3:QUAT", REL_POSE_POSE_TYPE_3D},
        {"EDGE_SE2_XY", REL_POSE_LANDMARK_TYPE_2D},
        {"EDGE_SE3_XYZ", REL_POSE_LANDMARK_TYPE_3D},
        {"EDGE_RANGE", RANGE_MEASURE_TYPE}};

    // get just the first line and close the file
    std::string line;
    std::getline(in_file, line);
    in_file.close();

    // Get the item type with the first word
    std::istringstream iss(line);
    std::string item_type;

    double timestamp;
    // A bunch of placeholder strings to be used for populating different types
    std::string sym1, sym2;

    if (!(iss >> item_type))
    {
      throw std::runtime_error("Could not read item type from line " + line);
    }

    if (PyFGStringToType.find(item_type) == PyFGStringToType.end())
    {
      throw std::runtime_error("Unknown item type " + item_type);
    }

    switch (PyFGStringToType.find(item_type)->second)
    {
    case POSE_TYPE_2D:
      return 2;
    case POSE_TYPE_3D:
      return 3;
    case LANDMARK_TYPE_2D:
      return 2;
    case LANDMARK_TYPE_3D:
      return 3;
    default:
      throw std::runtime_error("Could not determine dimension from first line " +
                               line);
    }
  }

  std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timers;
  void tic(const std::string &desc, const bool run = true)
  {
    if (!run)
      return;
    timers[desc] = std::chrono::high_resolution_clock::now();
  }

  void toc(const std::string &desc, const bool run = true)
  {
    if (!run)
      return;
    auto it = timers.find(desc);
    if (it == timers.end())
    {
      std::cerr << "TOC: no matching tic() found for \"" << desc << "\"\n";
      return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - it->second).count();
    std::cout << "TOC: " << desc << " took " << duration << " ms." << std::endl;
  }

  Problem parsePyfgTextToProblem(const std::string &filename)
  {
    // Note: This currently ignores all groundtruth measurements embedded
    // in the file
    int dim = getDimFromPyfgFirstLine(filename);
    int relaxation_rank = dim;
    VarPro::Formulation formulation = VarPro::Formulation::Explicit;
    VarPro::Preconditioner preconditioner =
        VarPro::Preconditioner::RegularizedCholesky;
    VarPro::Problem problem(dim, relaxation_rank, formulation, preconditioner);

    bool kTimeParsing = false;

    const std::map<std::string, PyFGType> PyFGStringToType{
        {"VERTEX_SE2", POSE_TYPE_2D},
        {"VERTEX_SE3:QUAT", POSE_TYPE_3D},
        {"VERTEX_SE2:PRIOR", POSE_PRIOR_2D},
        {"VERTEX_SE3:QUAT:PRIOR", POSE_PRIOR_3D},
        {"VERTEX_XY", LANDMARK_TYPE_2D},
        {"VERTEX_XYZ", LANDMARK_TYPE_3D},
        {"VERTEX_XY:PRIOR", LANDMARK_PRIOR_2D},
        {"VERTEX_XYZ:PRIOR", LANDMARK_PRIOR_3D},
        {"EDGE_SE2", REL_POSE_POSE_TYPE_2D},
        {"EDGE_SE3:QUAT", REL_POSE_POSE_TYPE_3D},
        {"EDGE_SE2_XY", REL_POSE_LANDMARK_TYPE_2D},
        {"EDGE_SE3_XYZ", REL_POSE_LANDMARK_TYPE_3D},
        {"EDGE_RANGE", RANGE_MEASURE_TYPE}};

    std::ifstream in_file(filename);
    if (!in_file.good())
    {
      throw std::runtime_error("Could not open file " + filename);
    }

    // --- Dedup sets (std::unordered_set only) ---
    // For variables, dedup by symbol string.
    std::unordered_set<std::string> seen_pose_syms;
    std::unordered_set<std::string> seen_landmark_syms;

    // For priors/measurements, dedup by the raw line (exact-duplicate lines).
    // This exactly matches your current semantics where equality is on the full object.
    std::unordered_set<std::string> seen_pose_priors;
    std::unordered_set<std::string> seen_landmark_priors;
    std::unordered_set<std::string> seen_rel_pose_pose;
    std::unordered_set<std::string> seen_rel_pose_landmark;
    std::unordered_set<std::string> seen_ranges;

    // --- Batch storage ---
    std::vector<Symbol> pose_vars;
    std::vector<Symbol> landmark_vars;

    std::vector<PosePrior> pose_priors;
    std::vector<LandmarkPrior> landmark_priors;

    std::vector<RelativePoseMeasurement> rel_pose_pose_meas;
    std::vector<RelativePoseLandmarkMeasurement> rel_pose_landmark_meas;
    std::vector<RangeMeasurement> range_meas;

    bool saw_any_prior = false;

    // Read line by line
    tic("Parsing PyFG file", kTimeParsing);
    std::string line;
    while (std::getline(in_file, line))
    {
      if (line.empty())
        continue;

      std::istringstream iss(line);
      std::string item_type;
      if (!(iss >> item_type))
      {
        throw std::runtime_error("Could not read item type from line " + line);
      }
      auto it = PyFGStringToType.find(item_type);
      if (it == PyFGStringToType.end())
      {
        throw std::runtime_error("Unknown item type " + item_type);
      }

      double timestamp; // used when present
      std::string sym1, sym2;

      switch (it->second)
      {
      case POSE_TYPE_2D:
      case POSE_TYPE_3D:
      {
        if (iss >> timestamp >> sym1)
        {
          if (seen_pose_syms.insert(sym1).second)
          {
            pose_vars.emplace_back(Symbol(sym1));
          }
        }
        else
        {
          throw std::runtime_error("Could not read pose variable from line " + line);
        }
      }
      break;

      case POSE_PRIOR_2D:
      {
        if (iss >> timestamp >> sym1)
        {
          if (seen_pose_priors.insert(line).second)
          {
            auto xy = readVector(iss, 2);
            auto R = fromAngle(readScalar(iss));
            auto cov = readSymmetric(iss, 3);
            pose_priors.push_back(PosePrior{Symbol(sym1), R, xy, cov});
            saw_any_prior = true;
          }
        }
        else
        {
          throw std::runtime_error("Could not read pose prior from line " + line);
        }
      }
      break;

      case POSE_PRIOR_3D:
      {
        if (iss >> timestamp >> sym1)
        {
          if (seen_pose_priors.insert(line).second)
          {
            auto xyz = readVector(iss, 3);
            auto R = readQuat(iss);
            auto cov = readSymmetric(iss, 6);
            pose_priors.push_back(PosePrior{Symbol(sym1), R, xyz, cov});
            saw_any_prior = true;
          }
        }
        else
        {
          throw std::runtime_error("Could not read pose prior from line " + line);
        }
      }
      break;

      case LANDMARK_TYPE_2D:
      case LANDMARK_TYPE_3D:
      {
        // Support both formats:
        //   with timestamp:    VERTEX_XY <timestamp> <symbol> <x> <y>
        //   without timestamp: VERTEX_XY <symbol> <x> <y>
        std::string first_token;
        if (!(iss >> first_token))
        {
          throw std::runtime_error("Could not read landmark variable from line " + line);
        }
        // Try to parse first_token as a double (timestamp).
        // If it fails, first_token is the symbol itself.
        try {
          std::stod(first_token);
          // It's a timestamp; read the actual symbol next
          if (!(iss >> sym1))
          {
            throw std::runtime_error("Could not read landmark variable from line " + line);
          }
        } catch (const std::invalid_argument &) {
          sym1 = first_token;
        }
        if (seen_landmark_syms.insert(sym1).second)
        {
          landmark_vars.emplace_back(Symbol(sym1));
        }
      }
      break;

      case LANDMARK_PRIOR_2D:
      {
        if (iss >> timestamp >> sym1)
        {
          if (seen_landmark_priors.insert(line).second)
          {
            auto xy = readVector(iss, 2);
            auto cov = readSymmetric(iss, 2);
            landmark_priors.push_back(LandmarkPrior{Symbol(sym1), xy, cov});
            saw_any_prior = true;
          }
        }
        else
        {
          throw std::runtime_error("Could not read landmark prior from line " + line);
        }
      }
      break;

      case LANDMARK_PRIOR_3D:
      {
        if (iss >> timestamp >> sym1)
        {
          if (seen_landmark_priors.insert(line).second)
          {
            auto xyz = readVector(iss, 3);
            auto cov = readSymmetric(iss, 3);
            landmark_priors.push_back(LandmarkPrior{Symbol(sym1), xyz, cov});
            saw_any_prior = true;
          }
        }
        else
        {
          throw std::runtime_error("Could not read landmark prior from line " + line);
        }
      }
      break;

      case REL_POSE_POSE_TYPE_2D:
      {
        if (iss >> timestamp >> sym1 >> sym2)
        {
          if (seen_rel_pose_pose.insert(line).second)
          {
            auto xy = readVector(iss, 2);
            auto R = fromAngle(readScalar(iss));
            auto cov = readSymmetric(iss, 3);
            rel_pose_pose_meas.push_back(
                RelativePoseMeasurement{Symbol(sym1), Symbol(sym2), R, xy, cov});
          }
        }
        else
        {
          throw std::runtime_error(
              "Could not read relative pose measurement from line " + line);
        }
      }
      break;

      case REL_POSE_POSE_TYPE_3D:
      {
        if (iss >> timestamp >> sym1 >> sym2)
        {
          if (seen_rel_pose_pose.insert(line).second)
          {
            auto xyz = readVector(iss, 3);
            auto R = readQuat(iss);
            auto cov = readSymmetric(iss, 6);
            rel_pose_pose_meas.push_back(
                RelativePoseMeasurement{Symbol(sym1), Symbol(sym2), R, xyz, cov});
          }
        }
        else
        {
          throw std::runtime_error(
              "Could not read relative pose measurement from line " + line);
        }
      }
      break;

      case REL_POSE_LANDMARK_TYPE_2D:
      {
        if (iss >> timestamp >> sym1 >> sym2)
        {
          if (seen_rel_pose_landmark.insert(line).second)
          {
            auto xy = readVector(iss, 2);
            auto cov = readSymmetric(iss, 2);
            rel_pose_landmark_meas.push_back(
                RelativePoseLandmarkMeasurement{Symbol(sym1), Symbol(sym2), xy, cov});
          }
        }
        else
        {
          throw std::runtime_error(
              "Could not read relative pose-landmark measurement from line " + line);
        }
      }
      break;

      case REL_POSE_LANDMARK_TYPE_3D:
      {
        if (iss >> timestamp >> sym1 >> sym2)
        {
          if (seen_rel_pose_landmark.insert(line).second)
          {
            auto xyz = readVector(iss, 3);
            auto cov = readSymmetric(iss, 3);
            rel_pose_landmark_meas.push_back(
                RelativePoseLandmarkMeasurement{Symbol(sym1), Symbol(sym2), xyz, cov});
          }
        }
        else
        {
          throw std::runtime_error(
              "Could not read relative pose-landmark measurement from line " + line);
        }
      }
      break;

      case RANGE_MEASURE_TYPE:
      {
        if (iss >> timestamp >> sym1 >> sym2)
        {
          if (seen_ranges.insert(line).second)
          {
            auto range = readScalar(iss);
            auto cov = readScalar(iss);
            range_meas.push_back(
                RangeMeasurement{Symbol(sym1), Symbol(sym2), range, cov});
          }
        }
        else
        {
          throw std::runtime_error("Could not read range measurement from line " + line);
        }
      }
      break;
      } // switch
    } // while getline
    in_file.close();
    toc("Parsing PyFG file", kTimeParsing);

    // --- Remap undeclared edge symbols to orphaned declared symbols.
    //
    // Some datasets have a systematic off-by-one: vertices are 0-indexed
    // (L0..L(N-1)) but edge references are 1-indexed (L1..L(N)).  This leaves
    // one declared symbol unreferenced ("orphan") and one referenced symbol
    // undeclared.  Adding the undeclared symbol as a NEW variable would create
    // an isolated node in the translational graph, making Q33 singular and
    // breaking the VarPro/Implicit Cholesky.  Instead, we remap each undeclared
    // symbol to the corresponding orphan so every declared variable is connected.

    // Collect all symbols that appear in at least one edge.
    std::unordered_set<std::string> all_ref_pose_syms, all_ref_lm_syms;
    for (const auto &m : rel_pose_landmark_meas)
    {
      all_ref_pose_syms.insert(m.first_id.string());
      all_ref_lm_syms.insert(m.second_id.string());
    }
    for (const auto &m : rel_pose_pose_meas)
    {
      all_ref_pose_syms.insert(m.first_id.string());
      all_ref_pose_syms.insert(m.second_id.string());
    }
    for (const auto &m : range_meas)
    {
      // first_id and second_id can each be either a pose or a landmark
      // (e.g. SNL has range edges between two landmarks).  Classify by which
      // declared-symbol set the id appears in; default to pose for undeclared.
      if (seen_landmark_syms.count(m.first_id.string()))
        all_ref_lm_syms.insert(m.first_id.string());
      else
        all_ref_pose_syms.insert(m.first_id.string());
      if (seen_landmark_syms.count(m.second_id.string()))
        all_ref_lm_syms.insert(m.second_id.string());
      else
        all_ref_pose_syms.insert(m.second_id.string());
    }

    // Orphans: declared as a vertex but referenced by no edge.
    std::vector<Symbol> orphan_pose_vars, orphan_lm_vars;
    for (const auto &s : pose_vars)
      if (!all_ref_pose_syms.count(s.string()))
        orphan_pose_vars.push_back(s);
    for (const auto &s : landmark_vars)
      if (!all_ref_lm_syms.count(s.string()))
        orphan_lm_vars.push_back(s);

    // Undeclared: referenced by an edge but not declared as a vertex.
    std::vector<Symbol> undecl_pose_syms, undecl_lm_syms;
    for (const auto &sym : all_ref_pose_syms)
      if (!seen_pose_syms.count(sym))
        undecl_pose_syms.emplace_back(Symbol(sym));
    for (const auto &sym : all_ref_lm_syms)
      if (!seen_landmark_syms.count(sym))
        undecl_lm_syms.emplace_back(Symbol(sym));

    // Sort both lists by Symbol index so the pairing is deterministic.
    auto sym_cmp = [](const Symbol &a, const Symbol &b) { return a.key() < b.key(); };
    std::sort(orphan_pose_vars.begin(), orphan_pose_vars.end(), sym_cmp);
    std::sort(orphan_lm_vars.begin(),   orphan_lm_vars.end(),   sym_cmp);
    std::sort(undecl_pose_syms.begin(), undecl_pose_syms.end(), sym_cmp);
    std::sort(undecl_lm_syms.begin(),   undecl_lm_syms.end(),   sym_cmp);

    // Build remap tables: undeclared → orphan (paired by sorted order).
    std::unordered_map<std::string, Symbol> pose_remap, lm_remap;
    for (size_t i = 0; i < std::min(undecl_pose_syms.size(), orphan_pose_vars.size()); ++i)
      pose_remap.emplace(undecl_pose_syms[i].string(), orphan_pose_vars[i]);
    for (size_t i = 0; i < std::min(undecl_lm_syms.size(), orphan_lm_vars.size()); ++i)
      lm_remap.emplace(undecl_lm_syms[i].string(), orphan_lm_vars[i]);

    // Apply remaps to all measurements.
    auto remap_sym = [](const Symbol &s,
                        const std::unordered_map<std::string, Symbol> &tbl) -> Symbol
    {
      auto it = tbl.find(s.string());
      return (it != tbl.end()) ? it->second : s;
    };
    for (auto &m : rel_pose_landmark_meas)
    {
      m.first_id  = remap_sym(m.first_id,  pose_remap);
      m.second_id = remap_sym(m.second_id, lm_remap);
    }
    for (auto &m : rel_pose_pose_meas)
    {
      m.first_id  = remap_sym(m.first_id,  pose_remap);
      m.second_id = remap_sym(m.second_id, pose_remap);
    }
    for (auto &m : range_meas)
    {
      // Apply the same pose-vs-landmark classification used when building the
      // remap tables (see all_ref_* loop above).
      if (seen_landmark_syms.count(m.first_id.string()))
        m.first_id = remap_sym(m.first_id, lm_remap);
      else
        m.first_id = remap_sym(m.first_id, pose_remap);
      if (seen_landmark_syms.count(m.second_id.string()))
        m.second_id = remap_sym(m.second_id, lm_remap);
      else
        m.second_id = remap_sym(m.second_id, pose_remap);
    }
    // Any undeclared symbols that had no orphan to pair with are added as new
    // variables (fallback for datasets without orphans).
    for (size_t i = orphan_pose_vars.size(); i < undecl_pose_syms.size(); ++i)
      if (seen_pose_syms.insert(undecl_pose_syms[i].string()).second)
        pose_vars.emplace_back(undecl_pose_syms[i]);
    for (size_t i = orphan_lm_vars.size(); i < undecl_lm_syms.size(); ++i)
      if (seen_landmark_syms.insert(undecl_lm_syms[i].string()).second)
        landmark_vars.emplace_back(undecl_lm_syms[i]);

    // --- Perform the actual inserts (each exactly once) ---
    // Variables first (so any following factors refer to existing indices)
    std::string time_poses_str = "Adding " + std::to_string(pose_vars.size()) + " pose variables";
    tic(time_poses_str, kTimeParsing);
    for (const auto &s : pose_vars)
      problem.addPoseVariable(s);
    toc(time_poses_str, kTimeParsing);

    std::string time_lms_str = "Adding " + std::to_string(landmark_vars.size()) + " landmark variables";
    tic(time_lms_str, kTimeParsing);
    for (const auto &s : landmark_vars)
      problem.addLandmarkVariable(s);
    toc(time_lms_str, kTimeParsing);

    // Priors (these may trigger addOriginPose() once inside Problem, as before)
    std::string time_priors_str = "Adding " + std::to_string(pose_priors.size()) + " pose priors and " +
                                  std::to_string(landmark_priors.size()) + " landmark priors";
    tic(time_priors_str, kTimeParsing);
    for (const auto &p : pose_priors)
      problem.addPosePrior(p);
    for (const auto &p : landmark_priors)
      problem.addLandmarkPrior(p);
    toc(time_priors_str, kTimeParsing);

    // Measurements
    std::string time_rp_str = "Adding " + std::to_string(rel_pose_pose_meas.size()) +
                              " relative pose measurements";
    tic(time_rp_str, kTimeParsing);
    for (const auto &m : rel_pose_pose_meas)
      problem.addRelativePoseMeasurement(m);
    toc(time_rp_str, kTimeParsing);

    std::string time_rpl_str = "Adding " + std::to_string(rel_pose_landmark_meas.size()) +
                               " relative pose-landmark measurements";
    tic(time_rpl_str, kTimeParsing);
    problem.reserveRelativePoseLandmarkMeasurements(rel_pose_landmark_meas.size());
    for (const auto &m : rel_pose_landmark_meas)
      problem.addRelativePoseLandmarkMeasurement(m);
    toc(time_rpl_str, kTimeParsing);

    std::string time_range_str = "Adding " + std::to_string(range_meas.size()) +
                                 " range measurements";
    tic(time_range_str, kTimeParsing);
    for (const auto &m : range_meas)
      problem.addRangeMeasurement(m);
    toc(time_range_str, kTimeParsing);

    return problem;
  }

  Matrix fromAngle(double angle_rad)
  {
    Matrix rotation_matrix_2d(2, 2);
    rotation_matrix_2d << cos(angle_rad), -sin(angle_rad), sin(angle_rad),
        cos(angle_rad);
    return rotation_matrix_2d;
  }

  Matrix fromQuat(double qx, double qy, double qz, double qw)
  {
    Eigen::Quaterniond q(qw, qx, qy, qz);
    auto rot_mat = q.toRotationMatrix();
    // Not sure why we can't cast it directly?
    Matrix result(3, 3);
    result << rot_mat(0, 0), rot_mat(0, 1), rot_mat(0, 2), rot_mat(1, 0),
        rot_mat(1, 1), rot_mat(1, 2), rot_mat(2, 0), rot_mat(2, 1), rot_mat(2, 2);
    return result;
  }

  Scalar readScalar(std::istringstream &iss)
  {
    Scalar result;
    if (iss >> result)
    {
      return result;
    }
    else
    {
      throw std::runtime_error("Could not read scalar");
    }
  }

  Vector readVector(std::istringstream &iss, int dim)
  {
    Vector result(dim);
    for (int i{0}; i < dim; i++)
    {
      if (iss >> result(i))
      {
        continue;
      }
      else
      {
        throw std::runtime_error("Could not read vector");
      }
    }
    return result;
  }

  /**
   * @brief Reads a xyzw quaternion from a string stream
   * @param iss string stream to read from
   * @return Rotation matrix representation of the quaternion
   */
  Matrix readQuat(std::istringstream &iss)
  {
    Vector result(4, 1);
    for (int i{0}; i < 4; i++)
    {
      if (iss >> result(i))
      {
        continue;
      }
      else
      {
        throw std::runtime_error("Could not read quaternion");
      }
    }
    return fromQuat(result(0), result(1), result(2), result(3));
  }

  /**
   * @brief Reads a symmetric matrix from a string stream in column-major order
   *
   * @param iss input stream to read from
   * @param dim dimension of the matrix
   * @return dim x dim matrix of doubles
   */
  Matrix readSymmetric(std::istringstream &iss, int dim)
  {
    Matrix cov(dim, dim);
    double val;
    for (int i{0}; i < dim; i++)
    {
      for (int j{i}; j < dim; j++)
      {
        if (iss >> val)
        {
          cov(i, j) = val;
          cov(j, i) = val;
        }
        else
        {
          std::cout << "Attempted to parse covariance matrix. i:" << i
                    << " j:" << j << " val:" << val << std::endl;
          throw std::runtime_error("Could not read covariance matrix");
        }
      }
    }
    return cov;
  }
} // namespace VarPro
