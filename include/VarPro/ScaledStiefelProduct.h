/**
 * ScaledStiefelProduct: (R_{>0})^n × St(p,k)^n
 * - Point X = { s ∈ R_{>0}^n, R ∈ R^{p×(k n)} } where R has n blocks (p×k) with orthonormal columns.
 * - Tangent T = { ds ∈ R^n, Z ∈ T_R St(p,k)^n }.
 * - Provides simple projections/retractions for s (keep >0) and R (blockwise Stiefel projection).
 */

#pragma once

#include <random>
#include <Eigen/Dense>

#include <VarPro/Types.h>
#include <VarPro/MatrixManifold.h>

namespace VarPro
{

  class ScaledStiefelProduct{

    struct Point
    {
      Vector s; // size n_
      Matrix R; // size p_ × (k_ n_), block i is p_×k_
    };

    struct Tangent
    {
      Vector ds; // size n_
      Matrix Z;  // size p_ × (k_ n_)
    };

  private:
    size_t k_{}; // columns per Stiefel block
    size_t p_{}; // ambient rows
    size_t n_{}; // number of blocks

  public:
    ScaledStiefelProduct() = default;
    ScaledStiefelProduct(size_t k, size_t p, size_t n) : k_(k), p_(p), n_(n) {}
    ~ScaledStiefelProduct() = default;

    // Accessors/mutators
    void set_k(size_t k) { k_ = k; }
    void set_p(size_t p) { p_ = p; }
    void set_n(size_t n) { n_ = n; }
    size_t get_k() const { return k_; }
    size_t get_p() const { return p_; }
    size_t get_n() const { return n_; }
    void addNewFrame() { n_++; }

    // ---- Core API (separate s and R) ----

    // Project R to St(p,k)^n (blockwise).
    // The same as what you have already writen in StiefelProduct.h
    Matrix projectRToStiefel(const Matrix &A_R) const;

    // Project s to (R_{>0})^n
    // keep the same as in XM and Manopt, use exp as retraction
    Vector projectSToPositive(const Vector &A_s, const Vector &V_s) const;

    // Project both parts at once.
    Point projectToManifold(const Matrix &A_R, const Vector &A_s, const Vector &V_s) const
    {
      Point X;
      X.R = projectRToStiefel(A_R);
      X.s = projectSToPositive(A_s, V_s);
      return X;
    }

    // R * symblockdiag(B^T C) (blockwise helper).
    // Just copy from SiefelProduct
    Matrix SymBlockDiagProduct(const Matrix &A, const Matrix &BT, const Matrix &C) const;

    // Tangent projections.
    Matrix projectRToStiefel_Tangent(const Matrix &A_R, const Matrix &V_R) const
    {
      return V_R - SymBlockDiagProduct(A_R, A_R.transpose(), V_R);
    }

    // Tangent space is just R^n
    Vector projectSToPositive_Tangent(const Vector & /*A_s*/, const Vector &V_s) const
    {
      return V_s;
    }

    Point projectToTangentSpace(const Point &A, const Point &V) const
    {
      Point T;
      T.R = projectRToStiefel_Tangent(A.R, V.R);
      T.s = projectSToPositive_Tangent(A.s, V.s);
      return T;
    }

    // Random sample: s > 0, R ∈ St(p,k)^n.
    Point random_sample(const std::default_random_engine::result_type &seed =
                            std::default_random_engine::default_seed) const;
  };

} // namespace VarPro
