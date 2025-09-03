/** @file
    @brief The main header file for the VARPRO library.

    This file provides the primary interface to the VARPRO library. All
    usage of the library should be done through this file.
*/

#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/PyfgTextParser.h>
#include <string>
#include <utility>
#include <vector>

namespace VarPro {

using TntResult = Optimization::Riemannian::TNTResult<Matrix, Scalar>;
using ProblemResult = std::pair<TntResult, std::vector<Matrix>>;

ProblemResult solveProblem(Problem &problem, const Matrix &x0,
                     int max_relaxation_rank = 20, bool verbose = false,
                     bool log_iterates = false, bool show_iterates = false);
inline ProblemResult solveProblem(std::string filepath) {
  Problem problem = parsePyfgTextToProblem(filepath);
  Matrix x0 = Matrix();
  throw std::runtime_error(
      "Not implemented -- need to decide how to get initialization");
  return solveProblem(problem, x0);
}

Matrix saddleEscape(const Problem &problem, const Matrix &Y, Scalar theta,
                    const Vector &v, Scalar gradient_tolerance,
                    Scalar preconditioned_gradient_tolerance);

Matrix projectSolution(const Problem &problem, const Matrix &Y,
                       bool verbose = false);

} // namespace VarPro
