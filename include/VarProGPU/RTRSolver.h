/**
 * @file RTRSolver.h
 * @brief Shared RTR/pTCG types used by GpuRTRSolver.
 *
 * Defines the algorithm hyperparameters (RTRParams), result struct (RTRResult),
 * inner-CG result (PTCGResult), and the host-side Steihaug-Toint pTCG
 * subproblem solver (PTCGSubproblemSolver). The GPU outer loop and inner CG
 * live in GpuRTRSolver.h.
 *
 * For the CPU production solver, use VarPro::solveProblem (TNT) from
 * VarPro/Solver.h.
 */

#pragma once

#include <VarPro/Types.h>

#include <Eigen/Dense>

#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

namespace VarProGPU {

// ---------------------------------------------------------------------------
// RTRParams — algorithm hyperparameters matching the paper's setup
// ---------------------------------------------------------------------------

struct RTRParams {
  // Trust-region
  double Delta0                = 1.0;   ///< initial trust-region radius
  double eta1                  = 0.05;  ///< min gain ratio for acceptance
  double eta2                  = 0.9;   ///< gain ratio for radius increase
  double alpha1                = 0.25;  ///< radius shrink factor
  double alpha2                = 3.0;   ///< radius grow factor
  double Delta_min             = 1e-5;  ///< minimum allowed radius
  // pTCG
  int    max_inner_iters       = 80;    ///< max pTCG iterations per outer
  double kappa_fgr             = 0.1;   ///< relative residual decrease target
  double theta                 = 0.8;   ///< superlinear target exponent
  // Outer convergence
  int    max_outer_iters       = 250;
  double gradient_tol          = 1e-6;
  double precond_gradient_tol  = 1e-6;
  double relative_decrease_tol = 1e-6;
  double stepsize_tol          = 1e-6;
  double max_time_seconds      = 300.0;
  // Diagnostics
  bool   verbose               = false;
  int    print_precision       = 4;
  bool   log_iterates          = false;
};

// ---------------------------------------------------------------------------
// RTRResult
// ---------------------------------------------------------------------------

struct RTRResult {
  VarPro::Matrix           x;
  VarPro::Scalar           f{0};
  VarPro::Scalar           grad_norm{0};
  int                      outer_iters{0};
  std::vector<int>         inner_iters_per_outer;
  std::vector<VarPro::Scalar> objective_values;
  std::vector<VarPro::Scalar> gradient_norms;
  std::vector<double>      elapsed_times;
  std::string              stop_reason;
};

// ---------------------------------------------------------------------------
// PTCGResult — inner subproblem
// ---------------------------------------------------------------------------

struct PTCGResult {
  VarPro::Matrix step;
  double         step_M_norm{0};
  int            iters{0};
  bool           hit_boundary{false};
};

// ---------------------------------------------------------------------------
// PTCGSubproblemSolver
// Steihaug-Toint preconditioned truncated conjugate gradients for:
//   min_{h ∈ T_x M} m(h) = <g, h> + 0.5 <h, H h>    s.t. ||h||_M <= Delta
//
// H is the Riemannian Hessian operator at x.
// M is the preconditioner (optional, defaults to identity).
// All operations use the supplied inner-product function.
// ---------------------------------------------------------------------------

class PTCGSubproblemSolver {
 public:
  using HessOp    = std::function<VarPro::Matrix(const VarPro::Matrix&,
                                                  const VarPro::Matrix&)>;
  using PrecondOp = std::function<VarPro::Matrix(const VarPro::Matrix&)>;
  using InnerProd = std::function<double(const VarPro::Matrix&,
                                          const VarPro::Matrix&)>;
  using ProjectOp = std::function<VarPro::Matrix(const VarPro::Matrix&)>;

  /**
   * @brief Solve the trust-region subproblem.
   *
   * @param x        current iterate (for Hessian evaluation)
   * @param g        Riemannian gradient at x
   * @param Hess     Riemannian Hessian-vector product operator
   * @param inner    Riemannian metric (inner product)
   * @param project  tangent-space projection (to keep iterates in T_x M)
   * @param Delta    trust-region radius (in M-norm)
   * @param params   algorithm parameters
   * @param precon   optional preconditioner M
   */
  static PTCGResult solve(
      const VarPro::Matrix& x,
      const VarPro::Matrix& g,
      HessOp Hess,
      InnerProd inner,
      ProjectOp project,
      double Delta,
      const RTRParams& params,
      std::optional<PrecondOp> precon = std::nullopt) {

    PTCGResult result;
    result.step = VarPro::Matrix::Zero(g.rows(), g.cols());

    // h = 0, r = g, z = M^{-1} g
    VarPro::Matrix h = VarPro::Matrix::Zero(g.rows(), g.cols());
    VarPro::Matrix r = g;
    VarPro::Matrix z = precon ? (*precon)(r) : r;

    double rz = inner(r, z);
    if (rz <= 0) {
      result.step = h;
      return result;
    }

    double g_norm = std::sqrt(inner(g, g));
    double rz0 = rz;

    VarPro::Matrix p = -z;   // search direction

    double h_M_norm_sq = 0.0;
    const int max_iters = params.max_inner_iters;

    for (int j = 0; j < max_iters; ++j) {
      result.iters = j + 1;

      VarPro::Matrix Hp = Hess(x, p);
      double pHp = inner(p, Hp);

      if (pHp <= 0) {
        // Negative curvature: step to boundary
        double ph_inner = inner(p, h);
        double pp_inner = inner(p, p);
        double hh_inner = inner(h, h);
        double tau = solveQuadratic(pp_inner, 2.0 * ph_inner,
                                    hh_inner - Delta * Delta);
        result.step = h + tau * p;
        result.step_M_norm = Delta;
        result.hit_boundary = true;
        return result;
      }

      double alpha = rz / pHp;

      // Trial step h_new = h + alpha * p
      VarPro::Matrix h_new = h + alpha * p;
      double h_new_M_norm_sq = inner(h_new, h_new);

      if (std::sqrt(h_new_M_norm_sq) >= Delta) {
        // Step to boundary
        double ph_inner = inner(p, h);
        double pp_inner = inner(p, p);
        double hh_inner = h_M_norm_sq;
        double tau = solveQuadratic(pp_inner, 2.0 * ph_inner,
                                    hh_inner - Delta * Delta);
        result.step = h + tau * p;
        result.step_M_norm = Delta;
        result.hit_boundary = true;
        return result;
      }

      h = h_new;
      h_M_norm_sq = h_new_M_norm_sq;

      r = r + alpha * Hp;

      // Check relative residual decrease (Steihaug criterion)
      double r_norm = std::sqrt(inner(r, r));
      double target = std::min(params.kappa_fgr,
                                std::pow(g_norm, params.theta)) * g_norm;
      if (r_norm <= target) {
        result.step = h;
        result.step_M_norm = std::sqrt(h_M_norm_sq);
        return result;
      }

      z = precon ? (*precon)(r) : r;
      double rz_new = inner(r, z);

      double beta = rz_new / rz;
      p = -z + beta * p;
      rz = rz_new;
    }

    result.step = h;
    result.step_M_norm = std::sqrt(h_M_norm_sq);
    return result;
  }

 private:
  // Solve a t^2 * a + t * b + c = 0 for the positive root (> 0)
  static double solveQuadratic(double a, double b, double c) {
    double disc = b * b - 4.0 * a * c;
    if (disc < 0) disc = 0;
    return (-b + std::sqrt(disc)) / (2.0 * a);
  }
};

// ---------------------------------------------------------------------------
// RTRSolverBase — backend-agnostic RTR outer loop
// ---------------------------------------------------------------------------

class RTRSolverBase {
 public:
  using CostFn     = std::function<VarPro::Scalar(const VarPro::Matrix&)>;
  using GradFn     = std::function<VarPro::Matrix(const VarPro::Matrix&)>;
  using HessVecFn  = std::function<VarPro::Matrix(const VarPro::Matrix&,
                                                    const VarPro::Matrix&)>;
  using ProjectFn  = std::function<VarPro::Matrix(const VarPro::Matrix&,
                                                    const VarPro::Matrix&)>;
  using RetractFn  = std::function<VarPro::Matrix(const VarPro::Matrix&,
                                                    const VarPro::Matrix&)>;
  using PrecondFn  = std::function<VarPro::Matrix(const VarPro::Matrix&,
                                                    const VarPro::Matrix&)>;
  using InnerFn    = std::function<VarPro::Scalar(const VarPro::Matrix&,
                                                    const VarPro::Matrix&,
                                                    const VarPro::Matrix&)>;
  using SepUpdateFn = std::function<void(VarPro::Matrix&)>;

  RTRSolverBase() = default;

  RTRResult solve(
      CostFn cost,
      GradFn euclidean_gradient,
      HessVecFn hess_vec,
      ProjectFn project,
      RetractFn retract,
      InnerFn inner,
      const VarPro::Matrix& x0,
      const RTRParams& params = RTRParams{},
      std::optional<PrecondFn> precon = std::nullopt,
      std::optional<SepUpdateFn> sep_update = std::nullopt) {

    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    RTRResult result;
    VarPro::Matrix x = x0;
    double fx = cost(x);

    // Riemannian gradient = tangent projection of Euclidean gradient
    VarPro::Matrix egrad = euclidean_gradient(x);
    VarPro::Matrix grad  = project(x, egrad);

    double Delta = params.Delta0;

    auto elapsed = [&]() -> double {
      return std::chrono::duration<double>(clock::now() - t0).count();
    };

    auto inner_at_x = [&](const VarPro::Matrix& V,
                           const VarPro::Matrix& W) -> double {
      return inner(x, V, W);
    };

    auto precon_at_x = [&](const VarPro::Matrix& V) -> VarPro::Matrix {
      return project(x, (*precon)(x, V));
    };

    double grad_norm = std::sqrt(inner_at_x(grad, grad));
    double precond_grad_norm = grad_norm;
    if (precon) {
      VarPro::Matrix pg = precon_at_x(grad);
      precond_grad_norm = std::sqrt(inner_at_x(pg, pg));
    }

    result.objective_values.push_back(fx);
    result.gradient_norms.push_back(grad_norm);
    result.elapsed_times.push_back(0.0);

    if (params.verbose) {
      std::cout << "RTR solver start: f=" << fx
                << "  |g|=" << grad_norm << "\n";
    }

    for (int iter = 0; iter < params.max_outer_iters; ++iter) {
      double t_iter = elapsed();

      // Stopping criteria
      if (grad_norm < params.gradient_tol) {
        result.stop_reason = "gradient_norm";
        break;
      }
      if (precond_grad_norm < params.precond_gradient_tol) {
        result.stop_reason = "precond_gradient_norm";
        break;
      }
      if (t_iter > params.max_time_seconds) {
        result.stop_reason = "time_limit";
        break;
      }

      // Build Hessian operator at x
      auto Hess = [&](const VarPro::Matrix& X,
                      const VarPro::Matrix& eta) -> VarPro::Matrix {
        return project(X, hess_vec(X, eta));
      };

      // Build project operator (tangent at x)
      auto Proj = [&](const VarPro::Matrix& eta) -> VarPro::Matrix {
        return project(x, eta);
      };

      // Optional preconditioner
      std::optional<PTCGSubproblemSolver::PrecondOp> ptcg_precon = std::nullopt;
      if (precon) {
        ptcg_precon = precon_at_x;
      }

      // Solve trust-region subproblem with pTCG
      auto hess_at_x = [&](const VarPro::Matrix& /*X*/,
                            const VarPro::Matrix& eta) -> VarPro::Matrix {
        return Hess(x, eta);
      };
      PTCGResult ptcg = PTCGSubproblemSolver::solve(
          x, grad, hess_at_x, inner_at_x, Proj,
          Delta, params, ptcg_precon);

      // Evaluate trial point
      VarPro::Matrix x_trial = retract(x, ptcg.step);
      double fx_trial = cost(x_trial);

      // Predicted decrease (quadratic model)
      VarPro::Matrix Hh = project(x, hess_vec(x, ptcg.step));
      double dm = -inner_at_x(grad, ptcg.step) -
                   0.5 * inner_at_x(ptcg.step, Hh);
      double df = fx - fx_trial;
      double rho = (dm > 1e-15) ? (df / dm) : 0.0;

      bool accepted = (!std::isnan(rho) && rho > params.eta1);

      if (accepted) {
        x = std::move(x_trial);
        fx = fx_trial;

        // Separable structure update (variable projection step)
        if (sep_update) {
          (*sep_update)(x);
          fx = cost(x);
        }

        // Check stepsize criterion
        double h_norm = std::sqrt(inner_at_x(ptcg.step, ptcg.step));
        if (h_norm < params.stepsize_tol) {
          result.stop_reason = "stepsize";
          break;
        }

        // Check relative decrease
        double rel_dec = df / (1e-15 + std::abs(fx));
        if (rel_dec < params.relative_decrease_tol) {
          result.stop_reason = "relative_decrease";
          break;
        }

        // Recompute gradient at new x
        egrad = euclidean_gradient(x);
        grad  = project(x, egrad);
        grad_norm = std::sqrt(inner(x, grad, grad));
        if (precon) {
          VarPro::Matrix pg = precon_at_x(grad);
          precond_grad_norm = std::sqrt(inner(x, pg, pg));
        } else {
          precond_grad_norm = grad_norm;
        }
      }

      // Update trust-region radius
      if (!std::isnan(rho) && rho >= params.eta2) {
        Delta = std::max(params.alpha2 * ptcg.step_M_norm, Delta);
      } else if (std::isnan(rho) || rho < params.eta1) {
        Delta = params.alpha1 * ptcg.step_M_norm;
        if (Delta < params.Delta_min) {
          result.stop_reason = "trust_region_radius";
          break;
        }
      }

      result.inner_iters_per_outer.push_back(ptcg.iters);
      result.objective_values.push_back(fx);
      result.gradient_norms.push_back(grad_norm);
      result.elapsed_times.push_back(elapsed());
      result.outer_iters = iter + 1;

      if (params.verbose) {
        std::cout << "  iter " << iter + 1
                  << "  f=" << fx
                  << "  |g|=" << grad_norm
                  << "  Delta=" << Delta
                  << "  inner=" << ptcg.iters
                  << "  rho=" << rho
                  << (accepted ? "  ACCEPT" : "  reject")
                  << "\n";
      }
    }

    if (result.stop_reason.empty()) result.stop_reason = "max_iterations";

    result.x = x;
    result.f = fx;
    result.grad_norm = grad_norm;
    return result;
  }
};

// ---------------------------------------------------------------------------
// CpuRTRSolver — wraps Problem's existing callbacks through RTRSolverBase
// ---------------------------------------------------------------------------

class CpuRTRSolver : public RTRSolverBase {
 public:
  RTRResult solve(VarPro::Problem& prob,
                  const VarPro::Matrix& x0,
                  const RTRParams& params = RTRParams{}) {
    using namespace VarPro;

    // Normalize file-based initializations onto the manifold before the first
    // objective/gradient evaluation so the CPU path matches TNT/GPU behavior.
    Matrix projected_x0 = prob.projectToManifold(x0);

    // Cache Euclidean gradient for reuse (avoids double computation)
    Matrix egrad_cache;

    auto cost_fn = [&](const Matrix& Y) -> Scalar {
      return prob.evaluateObjective(Y);
    };
    auto egrad_fn = [&](const Matrix& Y) -> Matrix {
      egrad_cache = prob.Euclidean_gradient(Y);
      return egrad_cache;
    };
    auto hess_fn = [&](const Matrix& Y, const Matrix& eta) -> Matrix {
      return prob.Riemannian_Hessian_vector_product(Y, egrad_cache, eta);
    };
    auto project_fn = [&](const Matrix& Y, const Matrix& V) -> Matrix {
      return prob.tangent_space_projection(Y, V);
    };
    auto retract_fn = [&](const Matrix& Y, const Matrix& V) -> Matrix {
      return prob.retract(Y, V);
    };
    auto inner_fn = [&](const Matrix& /*Y*/,
                         const Matrix& V1,
                         const Matrix& V2) -> Scalar {
      return (V1.transpose() * V2).trace();
    };
    auto precon_fn = [&](const Matrix& Y, const Matrix& V) -> Matrix {
      return prob.precondition(V);
    };

    std::optional<RTRSolverBase::SepUpdateFn> sep_update = std::nullopt;
    if (prob.getFormulation() == VarPro::Formulation::ExplicitVarPro) {
      sep_update = [&](Matrix& Y) {
        prob.separableStructureUpdate(Y);
      };
    }

    return RTRSolverBase::solve(
        cost_fn, egrad_fn, hess_fn, project_fn, retract_fn, inner_fn,
        projected_x0, params,
        std::make_optional(precon_fn),
        sep_update);
  }
};

}  // namespace VarProGPU
