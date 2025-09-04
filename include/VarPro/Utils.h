#pragma once

#include <VarPro/Problem.h>
#include <VarPro/Types.h>
#include <VarPro/Symbol.h>

#include <string>
#include <vector>
#include <unsupported/Eigen/SparseExtra>

namespace VarPro
{

    Matrix projectToSOd(const Matrix &A);

    void saveSolnToG20(const std::vector<Symbol> pose_symbols,
                       const Problem &problem, const Matrix &soln,
                       const std::string &fpath);

    void saveSolnToTum(const std::vector<Symbol> pose_symbols,
                       const Problem &problem, const Matrix &soln,
                       const std::string &fpath);

    inline void saveSparseMatrixToFile(const SparseMatrix &A, const std::string &fpath)
    {
        // Write Matrix Market (.mtx)
        if (!Eigen::saveMarket(A, fpath))
        {
            std::cerr << "Failed to write A.mtx\n";
        }
    }

    inline void readSparseMatrixFromFile(SparseMatrix &A, const std::string &fpath)
    {
        if (!Eigen::loadMarket(A, fpath))
        {
            throw std::runtime_error("Failed to read matrix from " + fpath);
        }
    }

} // namespace VarPro
