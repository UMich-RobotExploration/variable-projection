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

Matrix ScaledStiefelProduct::projectToManifold(const Matrix &A) const {
  // Each p×k block: SVD → s_i = mean(singular values), R_i = U V^T, return s_i * R_i.
  Matrix P(p_, k_ * n_);

  if (A.rows() != static_cast<Eigen::Index>(p_) ||
      A.cols() != static_cast<Eigen::Index>(k_ * n_)) {
    throw std::runtime_error(
        "ScaledStiefelProduct::projectToManifold (packed): bad shape.");
  }

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Eigen::Index>(i * k_);
    Matrix Ai = A.block(0, c0, p_, k_);
    Eigen::JacobiSVD<Matrix> svd(Ai, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Scalar s_i = svd.singularValues().sum() / static_cast<Scalar>(k_);
    P.block(0, c0, p_, k_) = s_i * (svd.matrixU() * svd.matrixV().transpose());
  }
  return P;
}

Matrix ScaledStiefelProduct::projectToTangentSpace(const Matrix &Y,
                                                    const Matrix &V) const {
  // At Y_i = s_i R_i, tangent condition: sym(Y_i^T V_i) = mu_i I_k.
  // proj(V_i) = V_i - Y_i * aniso_sym(Y_i^T V_i) / s_i^2
  Matrix result = V;
  const Eigen::Index ki = static_cast<Eigen::Index>(k_);
  const Matrix I_k = Matrix::Identity(ki, ki);

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Eigen::Index>(i * k_);
    const Matrix Yi = Y.block(0, c0, p_, ki);
    const Matrix Vi = V.block(0, c0, p_, ki);

    Scalar s_i_sq = Yi.squaredNorm() / static_cast<Scalar>(k_);

    Matrix P = Yi.transpose() * Vi;              // k×k
    Matrix S = Scalar(0.5) * (P + P.transpose()); // sym
    S -= (S.trace() / static_cast<Scalar>(k_)) * I_k; // remove isotropic part

    result.block(0, c0, p_, ki) = Vi - Yi * S / s_i_sq;
  }
  return result;
}

Matrix ScaledStiefelProduct::SymBlockDiagProduct_aniso(const Matrix &A,
                                                        const Matrix &BT,
                                                        const Matrix &C) const {
  // A:  p × k*n,  BT: k*n × p (= Y in Problem coords, block i is Y_i = k×p),
  // C:  p × k*n
  // Computes A_i * aniso_sym(BT_i * C_i) / s_i^2 per block.
  // s_i^2 = ||BT_i||_F^2 / k where BT_i = BT.block(i*k, 0, k, p).
  Matrix R(p_, k_ * n_);
  const Eigen::Index ki = static_cast<Eigen::Index>(k_);
  const Matrix I_k = Matrix::Identity(ki, ki);

  for (size_t i = 0; i < n_; ++i) {
    auto c0 = static_cast<Eigen::Index>(i * k_);
    const Matrix BTi = BT.block(c0, 0, ki, p_);      // k×p
    const Matrix Ci  = C.block(0, c0, p_, ki);        // p×k
    const Matrix Ai  = A.block(0, c0, p_, ki);        // p×k

    Scalar s_i_sq = BTi.squaredNorm() / static_cast<Scalar>(k_);

    Matrix P = BTi * Ci;                                // k×k
    Matrix S = Scalar(0.5) * (P + P.transpose());      // sym
    S -= (S.trace() / static_cast<Scalar>(k_)) * I_k;  // aniso_sym
    S /= s_i_sq;

    R.block(0, c0, p_, ki) = Ai * S;
  }
  return R;
}

} // namespace VarPro
