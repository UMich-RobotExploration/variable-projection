/**
 * @file GpuRTRSolver.h
 * @brief GPU-accelerated RTR+pTCG solver.
 *
 * Extends RTRSolverBase by:
 *   - Keeping iterate Y on device across all iterations
 *   - Using cuBLAS for all vector operations inside pTCG
 *   - Using GpuSchurOperator for Hessian-vector products
 *   - Using GpuJacobiPreconditioner for the pTCG inner loop (fully on-device:
 *     diagonal scaling via cublasDdgmm + Stiefel/Oblique tangent projection
 *     via CUDA kernels)
 *
 * The triangular solve (M^{-1} step inside the Schur operator) remains on
 * CPU in this version; all other hot-path operations are on GPU.
 *
 * The Hessian-vector product uses GPU SpMM for Q_sc*p; the tangent projection
 * of the Hessian result still uses the CPU Problem::tangent_space_projection.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarProGPU/ManifoldKernels.h>
#include <VarProGPU/MatrixFreeSchurOperator.h>
#include <VarProGPU/RTRSolver.h>

#include <VarPro/Problem.h>
#include <VarPro/Types.h>

#include <Eigen/Dense>

#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace VarProGPU {

/**
 * @brief Preconditioner callback: z_dev = M^{-1} r_dev, both on device.
 *
 * Used by gpuPTCGSolve.  Implementations should keep all data on-device.
 * The default implementation (GpuJacobiPreconditioner) applies diagonal
 * Q_sc scaling plus tangent-space projection entirely on GPU.
 */
using GpuPrecondFn =
    std::function<void(const GpuDenseMatrix& r, GpuDenseMatrix& z)>;

/**
 * @brief GPU block-diagonal preconditioner for the Schur complement Q_sc.
 *
 * Computes K×K diagonal blocks of Q_sc^{-1} once at construction (CPU),
 * uploads the block inverses to the device, then applies entirely on GPU:
 *
 *   z = Proj_{T_X M}( blkdiag(Q_sc)^{-1} · r )
 *
 * For the Stiefel block: one K×K block per pose, applied via custom kernel.
 * For the Oblique block: scalar diagonal (same as Jacobi).
 * Tangent projection via GPU Stiefel/Oblique kernels.
 */
struct GpuBlockDiagPreconditioner {
  DeviceBuffer<double> block_inv_dev;   ///< n_poses × (K*K) block inverses
  DeviceBuffer<double> range_diag_dev;  ///< n_range scalar inverses
  int p{0}, n_rot{0}, n_range{0}, K{0}, n_poses{0};

  explicit GpuBlockDiagPreconditioner(const VarProPrecomputeResult& pre) {
    p       = pre.p;
    n_rot   = pre.n_rot;
    n_range = pre.n_range;
    K       = pre.K;
    n_poses = pre.n_poses;

    // Z = M^{-1} * B^T  (m × p)
    VarPro::Matrix B_dense = VarPro::Matrix(pre.TransOffDiagRed);
    VarPro::Matrix Bt      = VarPro::Matrix(pre.TransOffDiagRed.transpose());
    VarPro::Matrix Z       = pre.LtransCholRed->solve(Bt);
    VarPro::Matrix Qm_dense = VarPro::Matrix(pre.Qmain);

    // Compute K×K block inverses for the Stiefel (rotation) block
    // Q_sc_ii = Qmain_ii - B_i * Z_i^T   (K×K for each pose)
    Eigen::VectorXd blocks(n_poses * K * K);
    for (int i = 0; i < n_poses; ++i) {
      int r0 = i * K;
      // Extract K×K block of Qmain
      Eigen::MatrixXd Qblock = Qm_dense.block(r0, r0, K, K);
      // Correction: B_i (K×m) * Z_i^T (m×K) = (B_i .* Z_i^T) summed
      // B_i = B_dense.block(r0, 0, K, m),  Z_i = Z.block(0, r0, m, K)
      // B_i * Z_i^T = B_dense(r0:r0+K, :) * Z(:, r0:r0+K)^T... no.
      // Z is m × p, Z_i^T means cols r0..r0+K-1 of Z transposed
      // correction = B_i * Z_cols_i = B_dense(rows r0..r0+K-1, :) * Z(:, r0..r0+K-1)
      Eigen::MatrixXd corr = B_dense.block(r0, 0, K, B_dense.cols()) *
                              Z.block(0, r0, Z.rows(), K);
      Eigen::MatrixXd Qsc_block = Qblock - corr;

      // Invert the K×K block (regularize if near-singular)
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(Qsc_block);
      Eigen::VectorXd evals = eig.eigenvalues().cwiseMax(1e-10);
      Eigen::MatrixXd inv_block =
          eig.eigenvectors() * evals.cwiseInverse().asDiagonal() *
          eig.eigenvectors().transpose();

      // Store row-major: blocks[i*K*K + j*K + k] = inv_block(j,k)
      for (int j = 0; j < K; ++j)
        for (int k = 0; k < K; ++k)
          blocks[i * K * K + j * K + k] = inv_block(j, k);
    }
    block_inv_dev.upload(blocks.data(), static_cast<std::size_t>(n_poses * K * K));

    // Scalar diagonal inverses for Oblique (range measurement) block
    if (n_range > 0) {
      Eigen::VectorXd diag_corr =
          (B_dense.bottomRows(n_range).cwiseProduct(
               Z.rightCols(n_range).transpose())).rowwise().sum();
      Eigen::VectorXd diag_qsc =
          Eigen::VectorXd(pre.Qmain.diagonal()).tail(n_range) - diag_corr;
      Eigen::VectorXd dinv = diag_qsc.cwiseMax(1e-10).cwiseInverse();
      range_diag_dev.upload(dinv.data(), static_cast<std::size_t>(n_range));
    }
  }

  void apply(GpuContext& ctx,
             const GpuDenseMatrix& X_dev,
             const GpuDenseMatrix& r_in,
             GpuDenseMatrix& z_out) const {
    int cols = r_in.cols;
    if (z_out.rows != r_in.rows || z_out.cols != cols)
      z_out.resize(r_in.rows, cols);

    // Step 1a: Stiefel block — K×K block-diagonal multiply
    if (n_rot > 0 && K >= 2 && K <= 4) {
      blockDiagMultiply(
          block_inv_dev.get(), r_in.data.get(), z_out.data.get(),
          n_poses, K, cols, p, ctx.stream.get());
    }

    // Step 1b: Oblique block — scalar diagonal multiply
    if (n_range > 0) {
      ddiagmm(ctx, range_diag_dev,
              r_in.data.get() + n_rot, z_out.data.get() + n_rot,
              n_range, cols, p);
    }

    // Step 2: tangent-space projection on device
    if (n_rot > 0 && K >= 2 && K <= 4) {
      stiefelProjectTangent(
          X_dev.data.get(), z_out.data.get(),
          n_poses, K, cols, ctx.stream.get(), p);
    }
    if (n_range > 0) {
      obliqueProjectTangent(
          X_dev.data.get() + n_rot, z_out.data.get() + n_rot,
          n_range, cols, ctx.stream.get(), p);
    }
  }
};

/**
 * @brief Hessian-vector callback: Hp_dev = Hess * p_dev, result on device.
 *
 * The hybrid implementation uses GPU SpMM for the bulk multiply and CPU
 * tangent_space_projection for the manifold correction — matching the
 * Riemannian Hessian used by CpuRTRSolver (proj(x, Q_sc eta)).
 */
using GpuHessVecFn =
    std::function<void(const GpuDenseMatrix& p, GpuDenseMatrix& Hp)>;

/**
 * @brief GPU pTCG inner loop operating entirely on device.
 *
 * G_dev (Riemannian gradient), hess_fn (Riemannian Hessian-vector product),
 * precon (optional preconditioner).  All cuBLAS/cuSPARSE ops share ctx.
 */
struct GpuPTCGResult {
  GpuDenseMatrix step_dev;  ///< pTCG step h (device)
  double         step_norm{0};
  int            iters{0};
  bool           hit_boundary{false};
};

GpuPTCGResult gpuPTCGSolve(
    GpuContext& ctx,
    const GpuDenseMatrix& G_dev,    ///< Riemannian gradient (tangent vector)
    const GpuHessVecFn& hess_fn,    ///< Hessian-vector product (projected)
    double Delta,
    const RTRParams& params,
    const std::optional<GpuPrecondFn>& precon = std::nullopt) {

  int p = G_dev.rows;
  int r = G_dev.cols;

  // Apply preconditioner or fall back to identity
  auto applyPrecon = [&](const GpuDenseMatrix& r_in, GpuDenseMatrix& z_out) {
    if (precon) {
      (*precon)(r_in, z_out);
    } else {
      dcopy(ctx, r_in, z_out);
    }
  };

  GpuPTCGResult result;
  result.step_dev.resize(p, r);
  result.step_dev.zero();  // h = 0

  // r_dev = g (residual = gradient)
  GpuDenseMatrix r_dev(p, r);
  dcopy(ctx, G_dev, r_dev);

  // z_dev = M^{-1} r
  GpuDenseMatrix z_dev(p, r);
  applyPrecon(r_dev, z_dev);

  // rz = <r, z>
  double rz = ddot(ctx, r_dev, z_dev);
  if (rz <= 0) return result;

  double g_norm = dnrm2(ctx, G_dev);
  double rz0 = rz;

  // p_dev = -z
  GpuDenseMatrix p_dev(p, r);
  dcopy(ctx, z_dev, p_dev);
  dscal(ctx, -1.0, p_dev);

  double h_norm_sq = 0.0;

  // Scratch buffer for Hess * p
  GpuDenseMatrix Hp_dev(p, r);

  for (int j = 0; j < params.max_inner_iters; ++j) {
    result.iters = j + 1;

    // Hp = Hess * p  (Riemannian Hessian-vector product)
    hess_fn(p_dev, Hp_dev);
    ctx.synchronize();

    double pHp = ddot(ctx, p_dev, Hp_dev);

    if (pHp <= 0) {
      // Negative curvature: step to boundary
      // tau = quadratic root for ||h + tau*p||^2 = Delta^2
      double ph = ddot(ctx, p_dev, result.step_dev);
      double pp = ddot(ctx, p_dev, p_dev);
      double disc = ph * ph - pp * (h_norm_sq - Delta * Delta);
      if (disc < 0) disc = 0;
      double tau = (-ph + std::sqrt(disc)) / pp;
      // result.step = h + tau * p
      daxpy(ctx, tau, p_dev, result.step_dev);
      result.step_norm = Delta;
      result.hit_boundary = true;
      ctx.synchronize();
      return result;
    }

    double alpha = rz / pHp;

    // h_new = h + alpha * p
    daxpy(ctx, alpha, p_dev, result.step_dev);
    double h_new_norm_sq = ddot(ctx, result.step_dev, result.step_dev);

    if (std::sqrt(h_new_norm_sq) >= Delta) {
      // Undo last step and compute step to boundary
      daxpy(ctx, -alpha, p_dev, result.step_dev);
      double ph = ddot(ctx, p_dev, result.step_dev);
      double pp = ddot(ctx, p_dev, p_dev);
      double disc = ph * ph - pp * (h_norm_sq - Delta * Delta);
      if (disc < 0) disc = 0;
      double tau = (-ph + std::sqrt(disc)) / pp;
      daxpy(ctx, tau, p_dev, result.step_dev);
      result.step_norm = Delta;
      result.hit_boundary = true;
      ctx.synchronize();
      return result;
    }
    h_norm_sq = h_new_norm_sq;

    // r = r + alpha * Hp
    daxpy(ctx, alpha, Hp_dev, r_dev);

    // Check convergence
    double r_norm = dnrm2(ctx, r_dev);
    double target = std::min(params.kappa_fgr,
                             std::pow(g_norm, params.theta)) * g_norm;
    if (r_norm <= target) {
      result.step_norm = std::sqrt(h_norm_sq);
      ctx.synchronize();
      return result;
    }

    // z = M^{-1} r
    applyPrecon(r_dev, z_dev);

    double rz_new = ddot(ctx, r_dev, z_dev);
    double beta = rz_new / rz;

    // p = -z + beta * p
    dscal(ctx, beta, p_dev);
    daxpy(ctx, -1.0, z_dev, p_dev);

    rz = rz_new;
  }

  result.step_norm = std::sqrt(h_norm_sq);
  ctx.synchronize();
  return result;
}

// ---------------------------------------------------------------------------
// GpuRTRSolver
// ---------------------------------------------------------------------------

class GpuRTRSolver {
 public:
  explicit GpuRTRSolver(GpuContext& ctx) : ctx_(ctx) {}

  /**
   * @brief Solve using the GPU Schur operator.
   *
   * x0 is on host; the iterate is uploaded once and stays on device.
   * The final result is downloaded at the end.
   *
   * Manifold operations use the CPU-side Problem methods for now (the
   * projection and retraction of the Stiefel/Oblique product), with the
   * iterate downloaded before each manifold op and re-uploaded after.
   * For large r (relaxation rank) or many iterations, these manifold ops
   * are a small fraction of the total work.
   */
  RTRResult solve(
      VarPro::Problem& prob,
      const GpuSchurOperator& op,
      const VarPro::Matrix& x0,
      const RTRParams& params = RTRParams{}) {

    using namespace VarPro;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto elapsed = [&]() -> double {
      return std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
    };

    RTRResult result;
    int p = x0.rows();
    int r = x0.cols();

    // Upload iterate to device
    GpuDenseMatrix X_dev(p, r);
    X_dev.upload(x0);

    // Scratch buffers
    GpuDenseMatrix QX_dev(p, r);     // Schur product (scratch)
    GpuDenseMatrix G_dev(p, r);      // Riemannian gradient (device)
    GpuDenseMatrix egrad_dev(p, r);  // Euclidean gradient on device (for Hessian corrections)
    GpuDenseMatrix Xtrial_dev(p, r);
    const bool use_scaled = prob.isScaledStiefel();

    // Compute initial cost and gradient on host (projecting back to manifold)
    Matrix X_host = X_dev.download();
    X_host = prob.projectToManifold(X_host);
    X_dev.upload(X_host);

    Matrix egrad;
    Matrix X_h = X_host;
    Matrix grad;
    double fx = 0.0;

    if (use_scaled) {
      // Scaled-Stiefel SfM needs the exact objective and Euclidean gradient,
      // including scale regularization, to match the CPU solver.
      egrad = prob.Euclidean_gradient(X_h);
      egrad_dev.upload(egrad);
      grad = prob.tangent_space_projection(X_h, egrad);
      fx = prob.evaluateObjective(X_h);
    } else {
      // Initial gradient: egrad = Q_sc * X, riemannian grad = project(X, egrad)
      op.applyDevice(X_dev, QX_dev);
      dcopy(ctx_, QX_dev, egrad_dev);  // keep egrad on device for Hessian corrections
      ctx_.synchronize();
      egrad = QX_dev.download();
      grad  = prob.tangent_space_projection(X_h, egrad);
      fx = 0.5 * (X_h.transpose() * egrad).trace();
    }
    G_dev.upload(grad);
    double Delta = params.Delta0;
    double grad_norm = std::sqrt((grad.transpose() * grad).trace());

    result.objective_values.push_back(fx);
    result.gradient_norms.push_back(grad_norm);
    result.elapsed_times.push_back(0.0);

    if (params.verbose) {
      std::cout << "GpuRTRSolver start: f=" << fx
                << "  |g|=" << grad_norm << "\n";
    }

    const auto& pre = op.precompute();

    for (int iter = 0; iter < params.max_outer_iters; ++iter) {
      // Stopping criteria
      if (grad_norm < params.gradient_tol) {
        result.stop_reason = "gradient_norm"; break;
      }
      if (elapsed() > params.max_time_seconds) {
        result.stop_reason = "time_limit"; break;
      }

      // GPU pTCG inner loop
      // Full Riemannian Hessian on GPU:
      //   1. Q_sc * eta            (cuSPARSE SpMM)
      //   2. -= sym(Y*egrad^T)*eta (Stiefel curvature correction kernel)
      //   3. -= eta*(egrad·Y)      (Oblique curvature correction kernel)
      //   4. tangent projection    (Stiefel + Oblique projection kernels)
      GpuHessVecFn gpu_hess = [&, use_scaled](const GpuDenseMatrix& p_in,
                                   GpuDenseMatrix& Hp_out) {
        if (use_scaled) {
          ctx_.synchronize();
          Matrix eta_host = p_in.download();
          Matrix Hp_host =
              prob.Riemannian_Hessian_vector_product(X_h, egrad, eta_host);
          Hp_out.uploadAsync(Hp_host, ctx_.stream.get());
          return;
        }
        op.applyDevice(p_in, Hp_out);
        int cols = Hp_out.cols;
        // Stiefel / Scaled Stiefel: curvature correction + tangent projection (all GPU)
        if (pre.n_rot > 0 && pre.K >= 2 && pre.K <= 4) {
          if (use_scaled) {
            scaledStiefelCurvatureCorrection(
                X_dev.data.get(), egrad_dev.data.get(),
                p_in.data.get(), Hp_out.data.get(),
                pre.n_poses, pre.K, cols, pre.p, ctx_.stream.get());
            scaledStiefelProjectTangent(
                X_dev.data.get(), Hp_out.data.get(),
                pre.n_poses, pre.K, cols, ctx_.stream.get(), pre.p);
          } else {
            stiefelCurvatureCorrection(
                X_dev.data.get(), egrad_dev.data.get(),
                p_in.data.get(), Hp_out.data.get(),
                pre.n_poses, pre.K, cols, pre.p, ctx_.stream.get());
            stiefelProjectTangent(
                X_dev.data.get(), Hp_out.data.get(),
                pre.n_poses, pre.K, cols, ctx_.stream.get(), pre.p);
          }
        }
        // Oblique: curvature correction + tangent projection
        if (pre.n_range > 0) {
          obliqueCurvatureCorrection(
              X_dev.data.get() + pre.n_rot, egrad_dev.data.get() + pre.n_rot,
              p_in.data.get() + pre.n_rot, Hp_out.data.get() + pre.n_rot,
              pre.n_range, cols, pre.p, ctx_.stream.get());
          obliqueProjectTangent(
              X_dev.data.get() + pre.n_rot, Hp_out.data.get() + pre.n_rot,
              pre.n_range, cols, ctx_.stream.get(), pre.p);
        }
      };

      // Hybrid block-Cholesky preconditioner: download r, CPU block-Cholesky
      // + tangent projection, upload z.  Matches CPU pTCG convergence rate.
      GpuPrecondFn gpu_precon = [&](const GpuDenseMatrix& r_in,
                                    GpuDenseMatrix& z_out) {
        ctx_.synchronize();
        VarPro::Matrix r_host = r_in.download();
        VarPro::Matrix z_host = prob.tangent_space_projection(
            X_h, prob.precondition(r_host));
        z_out.uploadAsync(z_host, ctx_.stream.get());
      };

      GpuPTCGResult ptcg = gpuPTCGSolve(
          ctx_, G_dev, gpu_hess, Delta, params,
          std::make_optional(gpu_precon));

      // Download step and trial point to host for manifold ops
      Matrix step_host = ptcg.step_dev.download();
      X_h = X_dev.download();
      Matrix X_trial = prob.retract(X_h, step_host);

      double fx_trial = 0.0;
      Matrix egrad_trial;
      if (use_scaled) {
        fx_trial = prob.evaluateObjective(X_trial);
      } else {
        // Upload trial point and evaluate cost
        Xtrial_dev.upload(X_trial);
        op.applyDevice(Xtrial_dev, QX_dev);
        ctx_.synchronize();
        egrad_trial = QX_dev.download();
        fx_trial = 0.5 * (X_trial.transpose() * egrad_trial).trace();
      }

      // Predicted decrease — use the same Riemannian Hessian as pTCG
      Matrix Hstep = prob.Riemannian_Hessian_vector_product(
          X_h, egrad, step_host);

      double inner_g_h  = (grad.transpose() * step_host).trace();
      double inner_h_Hh = (step_host.transpose() * Hstep).trace();
      double dm = -inner_g_h - 0.5 * inner_h_Hh;
      double df = fx - fx_trial;
      double rho = (std::abs(dm) > 1e-15) ? (df / dm) : 0.0;

      bool accepted = (!std::isnan(rho) && rho > params.eta1);

      if (accepted) {
        if (use_scaled) {
          X_h = X_trial;
          X_dev.upload(X_h);
          fx = fx_trial;
          egrad = prob.Euclidean_gradient(X_h);
          egrad_dev.upload(egrad);
        } else {
          // Accept: swap iterate
          X_dev = std::move(Xtrial_dev);
          Xtrial_dev.resize(p, r);
          X_h = X_dev.download();
          fx = fx_trial;
          egrad = egrad_trial;
          egrad_dev.upload(egrad);  // keep egrad on device for Hessian corrections
        }

        double h_norm = std::sqrt((step_host.transpose() * step_host).trace());
        if (h_norm < params.stepsize_tol) {
          result.stop_reason = "stepsize"; break;
        }
        double rel_dec = df / (1e-15 + std::abs(fx));
        if (rel_dec < params.relative_decrease_tol) {
          result.stop_reason = "relative_decrease"; break;
        }

        // Recompute Riemannian gradient
        grad = prob.tangent_space_projection(X_h, egrad);
        G_dev.upload(grad);
        grad_norm = std::sqrt((grad.transpose() * grad).trace());
      }

      // Update trust-region radius
      if (!std::isnan(rho) && rho >= params.eta2) {
        Delta = std::max(params.alpha2 * ptcg.step_norm, Delta);
      } else if (std::isnan(rho) || rho < params.eta1) {
        Delta = params.alpha1 * ptcg.step_norm;
        if (Delta < params.Delta_min) {
          result.stop_reason = "trust_region_radius"; break;
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
    result.x = X_dev.download();
    result.f = fx;
    result.grad_norm = grad_norm;
    return result;
  }

  /**
   * @brief Solve using GpuExplicitOperator (Explicit or ExplicitVarPro).
   *
   * The data matrix product is a single SpMM. For ExplicitVarPro, a
   * separable structure update is applied after each accepted step.
   */
  RTRResult solveExplicit(
      VarPro::Problem& prob,
      const GpuExplicitOperator& op,
      const VarPro::Matrix& x0,
      const RTRParams& params = RTRParams{}) {

    using namespace VarPro;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto elapsed = [&]() -> double {
      return std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
    };

    bool do_sep_update =
        (prob.getFormulation() == Formulation::ExplicitVarPro);

    RTRResult result;
    int p = x0.rows();
    int r = x0.cols();

    GpuDenseMatrix X_dev(p, r);
    GpuDenseMatrix QX_dev(p, r);
    GpuDenseMatrix G_dev(p, r);
    GpuDenseMatrix egrad_dev(p, r);
    GpuDenseMatrix Xtrial_dev(p, r);
    const bool use_scaled = prob.isScaledStiefel();

    Matrix X_host = prob.projectToManifold(x0);
    X_dev.upload(X_host);

    Matrix X_h = X_host;
    Matrix egrad;
    Matrix grad;
    double fx = 0.0;

    if (use_scaled) {
      egrad = prob.Euclidean_gradient(X_h);
      egrad_dev.upload(egrad);
      grad = prob.tangent_space_projection(X_h, egrad);
      fx = prob.evaluateObjective(X_h);
    } else {
      // Initial gradient
      op.applyDevice(X_dev, QX_dev);
      dcopy(ctx_, QX_dev, egrad_dev);
      ctx_.synchronize();
      egrad = QX_dev.download();
      grad = prob.tangent_space_projection(X_h, egrad);
      fx = prob.evaluateObjective(X_h);
    }
    G_dev.upload(grad);
    double Delta = params.Delta0;
    double grad_norm = std::sqrt((grad.transpose() * grad).trace());

    result.objective_values.push_back(fx);
    result.gradient_norms.push_back(grad_norm);
    result.elapsed_times.push_back(0.0);

    if (params.verbose)
      std::cout << "GpuRTR(Explicit) start: f=" << fx
                << "  |g|=" << grad_norm << "\n";

    int n_poses = prob.numPoses();
    int K = prob.dim();
    int n_rot = prob.numPosesDim();
    int n_range = prob.numRangeMeasurements();

    for (int iter = 0; iter < params.max_outer_iters; ++iter) {
      if (grad_norm < params.gradient_tol) {
        result.stop_reason = "gradient_norm"; break;
      }
      if (elapsed() > params.max_time_seconds) {
        result.stop_reason = "time_limit"; break;
      }

      // Hessian: GPU SpMM + Stiefel/Scaled-Stiefel curvature correction + tangent projection
      GpuHessVecFn gpu_hess = [&, use_scaled](const GpuDenseMatrix& p_in,
                                   GpuDenseMatrix& Hp_out) {
        if (use_scaled) {
          ctx_.synchronize();
          Matrix eta_host = p_in.download();
          Matrix Hp_host =
              prob.Riemannian_Hessian_vector_product(X_h, egrad, eta_host);
          Hp_out.uploadAsync(Hp_host, ctx_.stream.get());
          return;
        }
        op.applyDevice(p_in, Hp_out);
        int cols = Hp_out.cols;
        if (n_rot > 0 && K >= 2 && K <= 4) {
          if (use_scaled) {
            scaledStiefelCurvatureCorrection(
                X_dev.data.get(), egrad_dev.data.get(),
                p_in.data.get(), Hp_out.data.get(),
                n_poses, K, cols, p, ctx_.stream.get());
            scaledStiefelProjectTangent(
                X_dev.data.get(), Hp_out.data.get(),
                n_poses, K, cols, ctx_.stream.get(), p);
          } else {
            stiefelCurvatureCorrection(
                X_dev.data.get(), egrad_dev.data.get(),
                p_in.data.get(), Hp_out.data.get(),
                n_poses, K, cols, p, ctx_.stream.get());
            stiefelProjectTangent(
                X_dev.data.get(), Hp_out.data.get(),
                n_poses, K, cols, ctx_.stream.get(), p);
          }
        }
        if (n_range > 0) {
          obliqueCurvatureCorrection(
              X_dev.data.get() + n_rot, egrad_dev.data.get() + n_rot,
              p_in.data.get() + n_rot, Hp_out.data.get() + n_rot,
              n_range, cols, p, ctx_.stream.get());
          obliqueProjectTangent(
              X_dev.data.get() + n_rot, Hp_out.data.get() + n_rot,
              n_range, cols, ctx_.stream.get(), p);
        }
      };

      // Hybrid block-Cholesky preconditioner
      GpuPrecondFn gpu_precon = [&](const GpuDenseMatrix& r_in,
                                    GpuDenseMatrix& z_out) {
        ctx_.synchronize();
        Matrix r_host = r_in.download();
        Matrix z_host = prob.tangent_space_projection(
            X_h, prob.precondition(r_host));
        z_out.uploadAsync(z_host, ctx_.stream.get());
      };

      GpuPTCGResult ptcg = gpuPTCGSolve(
          ctx_, G_dev, gpu_hess, Delta, params,
          std::make_optional(gpu_precon));

      // Download step and evaluate trial point
      Matrix step_host = ptcg.step_dev.download();
      X_h = X_dev.download();
      Matrix X_trial = prob.retract(X_h, step_host);

      // ExplicitVarPro: separable structure update
      if (do_sep_update)
        prob.separableStructureUpdate(X_trial);

      double fx_trial = prob.evaluateObjective(X_trial);

      // Predicted decrease using full Riemannian Hessian
      Matrix Hstep = prob.Riemannian_Hessian_vector_product(
          X_h, egrad, step_host);
      double inner_g_h  = (grad.transpose() * step_host).trace();
      double inner_h_Hh = (step_host.transpose() * Hstep).trace();
      double dm = -inner_g_h - 0.5 * inner_h_Hh;
      double df = fx - fx_trial;
      double rho = (std::abs(dm) > 1e-15) ? (df / dm) : 0.0;

      bool accepted = (!std::isnan(rho) && rho > params.eta1);

      if (accepted) {
        X_h = X_trial;
        X_dev.upload(X_h);
        fx = fx_trial;

        // Recompute Euclidean gradient (includes scale regularization if SfM)
        egrad = prob.Euclidean_gradient(X_h);
        egrad_dev.upload(egrad);

        double h_norm = std::sqrt((step_host.transpose() * step_host).trace());
        if (h_norm < params.stepsize_tol) {
          result.stop_reason = "stepsize"; break;
        }
        double rel_dec = df / (1e-15 + std::abs(fx));
        if (rel_dec < params.relative_decrease_tol) {
          result.stop_reason = "relative_decrease"; break;
        }

        grad = prob.tangent_space_projection(X_h, egrad);
        G_dev.upload(grad);
        grad_norm = std::sqrt((grad.transpose() * grad).trace());
      }

      // Trust-region radius update
      if (!std::isnan(rho) && rho >= params.eta2) {
        Delta = std::max(params.alpha2 * ptcg.step_norm, Delta);
      } else if (std::isnan(rho) || rho < params.eta1) {
        Delta = params.alpha1 * ptcg.step_norm;
        if (Delta < params.Delta_min) {
          result.stop_reason = "trust_region_radius"; break;
        }
      }

      result.inner_iters_per_outer.push_back(ptcg.iters);
      result.objective_values.push_back(fx);
      result.gradient_norms.push_back(grad_norm);
      result.elapsed_times.push_back(elapsed());
      result.outer_iters = iter + 1;

      if (params.verbose) {
        std::cout << "  iter " << iter + 1
                  << "  f=" << fx << "  |g|=" << grad_norm
                  << "  Delta=" << Delta << "  inner=" << ptcg.iters
                  << "  rho=" << rho
                  << (accepted ? "  ACCEPT" : "  reject") << "\n";
      }
    }

    if (result.stop_reason.empty()) result.stop_reason = "max_iterations";
    result.x = X_dev.download();
    result.f = fx;
    result.grad_norm = grad_norm;
    return result;
  }

 private:
  GpuContext& ctx_;
};

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
