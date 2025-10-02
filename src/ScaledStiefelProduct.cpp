#include <Eigen/QR>
#include <iostream>

#include "VarPro/ScaledStiefelProduct.h"

namespace VarPro {

Matrix ScaledStiefelProduct::projectRToStiefel(const Matrix &A_R) const {
  // Project each p×k block of A_R onto St(p,k) using a thin-QR with
  // sign-fixing (make diag(R) ≥ 0 and absorb signs into Q).
  Matrix P(p_, k_ * n_);

  // Shape check
  if (A_R.rows() != p_ || A_R.cols() != k_ * n_) {
    throw std::runtime_error(
        "ScaledStiefelProduct::projectRToStiefel: "
        "A_R has shape " + std::to_string(A_R.rows()) + " x " + std::to_string(A_R.cols()) +
        " but expected " + std::to_string(p_) + " x " + std::to_string(k_ * n_));
  }

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Eigen::Index>(i * k_);

    // Block Ai ∈ R^{p×k}
    Matrix Ai = A_R.block(0, c0, p_, k_);

    // Thin QR (Householder). householderQ() gives an implicit p×p Q; take first k columns.
    Eigen::HouseholderQR<Matrix> qr(Ai);
    Matrix Qi = qr.householderQ() * Matrix::Identity(p_, k_);

    // Upper-triangular R (k×k) from the top-left corner.
    Matrix Ri = qr.matrixQR().topLeftCorner(k_, k_)
                  .template triangularView<Eigen::Upper>();

    // Make diag(R) nonnegative; absorb signs into Q’s columns.
    // The same as in Manopt qr_unique function
    Vector s = Vector::Ones(static_cast<Eigen::Index>(k_));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(k_); ++j) {
      if (Ri(j, j) < Scalar(0)) s(j) = Scalar(-1); // treat zeros as +1
    }
    Qi.noalias() = Qi * s.asDiagonal();

    // Write back block
    P.block(0, c0, p_, k_) = Qi;
  }

  return P; // Each block has orthonormal columns with a consistent sign convention.
}

Vector ScaledStiefelProduct::projectSToPositive(const Vector& A_s,
                                                const Vector& V_s) const {
  // Simple positive map for scales: s_new = s .* exp(v ./ s).
  // More details for math can refer to manopt
  if (A_s.rows() != static_cast<Eigen::Index>(n_)) {
    throw std::runtime_error("ScaledStiefelProduct::projectSToPositive: bad size for A_s.");
  }

  Vector P = (A_s.array() * (V_s.array() / A_s.array()).exp()).matrix();
  return P;
}

Matrix ScaledStiefelProduct::SymBlockDiagProduct(const Matrix &A,
                                                 const Matrix &BT,
                                                 const Matrix &C) const {
  // Compute R = A * SymBlockDiag(B^T * C) blockwise, where each block is k×k.
  Matrix R(p_, k_ * n_);
  Matrix Pblk(k_, k_);
  Matrix Sblk(k_, k_);

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Eigen::Index>(i * k_);
    // P_i = B_i^T * C_i
    Pblk = BT.block(c0, 0, k_, p_) * C.block(0, c0, p_, k_);
    // S_i = sym(P_i)
    Sblk = Scalar(0.5) * (Pblk + Pblk.transpose());
    // R_i = A_i * S_i
    R.block(0, c0, p_, k_) = A.block(0, c0, p_, k_) * Sblk;
  }
  return R;
}

ScaledStiefelProduct::Point
ScaledStiefelProduct::random_sample(const std::default_random_engine::result_type &seed) const {
  // Draw s > 0 (log-normal-like via |N(0,1)|) and R ~ N(0,1), then project.
  std::default_random_engine generator(seed);
  std::normal_distribution<Scalar> g;

  ScaledStiefelProduct::Point X;
  X.R.resize(static_cast<Eigen::Index>(p_), static_cast<Eigen::Index>(k_ * n_));
  X.s = Vector::Ones(static_cast<Eigen::Index>(n_));

  for (size_t i = 0; i < n_; ++i) {
    X.s(static_cast<Eigen::Index>(i)) = std::abs(g(generator));
  }
  for (size_t r = 0; r < p_; ++r)
    for (size_t c = 0; c < k_ * n_; ++c)
      X.R(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = g(generator);

  // Project R to Stiefel^n and s to (R_{>0})^n (using V_s = 0 here).
  return projectToManifold(X.R, X.s, Vector::Zero(static_cast<Eigen::Index>(n_)));
}

} // namespace VarPro 88
