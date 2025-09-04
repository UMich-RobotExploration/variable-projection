#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Symbol.h>

#include <string>
#include <vector>

namespace VarPro {

Matrix projectToSOd(const Matrix &A);

void saveSolnToG20(const std::vector<Symbol> pose_symbols,
                   const Problem &problem, const Matrix &soln,
                   const std::string &fpath);

void saveSolnToTum(const std::vector<Symbol> pose_symbols,
                   const Problem &problem, const Matrix &soln,
                   const std::string &fpath);

} // namespace VarPro
