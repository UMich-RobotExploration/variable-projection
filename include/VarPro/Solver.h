/** @file
    @brief The main header file for the VARPRO library.

    This file provides the primary interface to the VARPRO library. All
    usage of the library should be done through this file.
*/

#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/PyfgTextParser.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace VarPro
{

  using TntResult = Optimization::Riemannian::TNTResult<Matrix, Scalar>;
  using ProblemResult = TntResult;

  ProblemResult solveProblem(Problem &problem, const Matrix &x0, bool verbose = false);
  ProblemResult solveProblem(
      Problem &problem, const Matrix &x0,
      const std::optional<InstrumentationFunction> &user_function,
      bool verbose = false);

} // namespace VarPro
