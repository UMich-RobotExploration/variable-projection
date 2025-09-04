#include <VarPro/Utils.h>

#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>

#include <fstream>
#include <iostream>

#include "ILDL/ILDL.h"
#include "Optimization/LinearAlgebra/LOBPCG.h"

namespace VarPro {

using SymmetricLinOp =
    Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>;

Matrix projectToSOd(const Matrix &M) {
  // Compute the SVD of M
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Scalar detU = svd.matrixU().determinant();
  Scalar detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    return svd.matrixU() * svd.matrixV().transpose();
  } else {
    Matrix Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    return Uprime * svd.matrixV().transpose();
  }
}

Matrix getTranslation(const Symbol &sym, const Problem &problem,
                      const Matrix &soln) {
  checkMatrixShape("getTranslation", problem.getDataMatrixSize(), problem.dim(),
                   soln.rows(), soln.cols());
  return soln.row(problem.getTranslationIdx(sym));
}

Matrix getRotation(const Symbol &sym, const Problem &problem,
                   const Matrix &soln) {
  checkMatrixShape("getRotation", problem.getDataMatrixSize(), problem.dim(),
                   soln.rows(), soln.cols());

  // need the dim x dim rotation matrix
  Index start_idx = problem.getRotationIdx(sym);
  Matrix rot =
      soln.block(start_idx * problem.dim(), 0, problem.dim(), problem.dim())
          .transpose();
  // check that the rotation matrix is valid
  if (std::abs(rot.determinant() - 1) > 1e-6) {
    throw std::runtime_error("Rotation matrix determinant is: " +
                             std::to_string(rot.determinant()) + " not 1");
  }
  if ((rot * rot.transpose() - Matrix::Identity(problem.dim(), problem.dim()))
          .norm() > 1e-6) {
    throw std::runtime_error("Rotation matrix is not orthogonal");
  }

  return rot;
}

void saveSolnToG20(const std::vector<Symbol> pose_symbols,
                   const Problem &problem, const Matrix &soln,
                   const std::string &fpath) {
  // we are assuming that the solution is in translation-explicit form. We have
  // helper functions to do this: "getTranslationExplicitSolution" and
  // "alignEstimateToOrigin"
  checkMatrixShape("saveSolnToG20", problem.getDataMatrixSize(), problem.dim(),
                   soln.rows(), soln.cols());

  // open fpath for writing
  std::ofstream output_file(fpath);
  if (!output_file.is_open()) {
    throw std::runtime_error("Could not open file " + fpath);
  }

  // iterate over all the symbols and find the rotation and translation indices
  for (size_t time = 0; time < pose_symbols.size(); time++) {
    //  write the poses to the file in the format:
    //  timestamp x y z qx qy qz qw
    Matrix tran = getTranslation(pose_symbols[time], problem, soln);
    Matrix rot = getRotation(pose_symbols[time], problem, soln);

    // get xyz from tran
    Scalar x = tran(0);
    Scalar y = tran(1);
    Scalar z;
    if (problem.dim() == 2) {
      z = 0;
    } else {
      z = tran(2);
    }

    // get quaternion from rot
    Eigen::Matrix3d rot_padded = Eigen::Matrix3d::Identity();
    rot_padded.block(0, 0, problem.dim(), problem.dim()) = rot;

    if (problem.dim() == 3) {
      // VERTEX_SE3:QUAT 1 0.341895 -0.0416997 0.0330394 -0.00189341 0.00395691
      // 0.0899835 0.995934

      Eigen::Quaternion<Scalar> quat(rot_padded);
      Scalar qw = quat.w();
      Scalar qx = quat.x();
      Scalar qy = quat.y();
      Scalar qz = quat.z();

      // write the line to the file
      output_file << "VERTEX_SE3:QUAT " << time << " " << x << " " << y << " "
                  << z << " " << qx << " " << qy << " " << qz << " " << qw
                  << "\n";
    } else {
      // VERTEX_SE2 1 0.144012 -0.004462 -0.017453
      Scalar theta = std::atan2(rot(1, 0), rot(0, 0));
      output_file << "VERTEX_SE2 " << time << " " << x << " " << y << " "
                  << theta << "\n";
    }
  }

  // close the file
  output_file.close();

  // print that we saved the poses
  // std::cout << "Saved robot poses to " << fpath << std::endl;
}

void saveSolnToTum(const std::vector<Symbol> pose_symbols,
                   const Problem &problem, const Matrix &soln,
                   const std::string &fpath) {
  // we are assuming that the solution is in translation-explicit form. We have
  // helper functions to do this: "getTranslationExplicitSolution" and
  // "alignEstimateToOrigin"
  checkMatrixShape("saveSolnToTum", problem.getDataMatrixSize(), problem.dim(),
                   soln.rows(), soln.cols());

  // open fpath for writing
  std::ofstream output_file(fpath);
  if (!output_file.is_open()) {
    throw std::runtime_error("Could not open file " + fpath);
  }

  // iterate over all the symbols and find the rotation and translation indices
  for (size_t time = 0; time < pose_symbols.size(); time++) {
    //  write the poses to the file in the format:
    //  timestamp x y z qx qy qz qw
    Matrix tran = getTranslation(pose_symbols[time], problem, soln);
    Matrix rot = getRotation(pose_symbols[time], problem, soln);

    // get xyz from tran
    Scalar x = tran(0);
    Scalar y = tran(1);
    Scalar z;
    if (problem.dim() == 2) {
      z = 0;
    } else {
      z = tran(2);
    }

    // get quaternion from rot
    Eigen::Matrix3d rot_padded = Eigen::Matrix3d::Identity();
    rot_padded.block(0, 0, problem.dim(), problem.dim()) = rot;
    Eigen::Quaternion<Scalar> quat(rot_padded);
    Scalar qw = quat.w();
    Scalar qx = quat.x();
    Scalar qy = quat.y();
    Scalar qz = quat.z();

    // write the line to the file
    output_file << time << " " << x << " " << y << " " << z << " " << qx << " "
                << qy << " " << qz << " " << qw << std::endl;
  }

  // close the file
  output_file.close();

  // print that we saved the poses
  // std::cout << "Saved robot poses to " << fpath << std::endl;
}

} // namespace VarPro
