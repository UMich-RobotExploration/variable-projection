/**
 * @file GpuDenseSchur.h
 * @brief Device-side formation of the explicit (dense) Schur complement.
 *
 * Formulation::Dense multiplies by the reduced system
 *
 *     Q_sc = Q_c - B M^{-1} B^T,     B = TransOffDiagRed (p x m),
 *                                    M = C^T Omega C     (m x m, SPD)
 *
 * which is p x p and therefore expensive to *form* even though applying it is
 * a single SYMM. Problem::fillDenseSchurMatrix() builds it on the host with a
 * blocked CHOLMOD multi-RHS solve; on the larger problems that host formation
 * costs more than the entire GPU solve that follows it (grid3D: 39 s of
 * formation ahead of 13 s of solve), so it, not the solve, is what caps the
 * end-to-end speedup of the Dense baseline.
 *
 * This builds the same matrix entirely on the device, using the same blocked
 * algorithm: cuSPARSE sparse-to-dense for the right-hand sides, cuDSS for
 * M^{-1}, cuSPARSE SpMM for the update, accumulating in place into Q_sc.
 * Nothing but the CSR arrays ever crosses PCIe.
 */

#pragma once

#ifdef VARPRO_HAVE_CUDA

#include <VarProGPU/GpuLinearAlgebra.h>
#include <VarPro/Types.h>

namespace VarProGPU {

class GpuSparseSolver;

/**
 * @brief Form Q_sc = Qmain - B M^{-1} B^T on the device.
 *
 * @param Qmain  p x p sparse, the unreduced block (Q_c). Whatever triangle
 *               structure it is stored in is reproduced in the output, exactly
 *               as Eigen's sparse->dense conversion does on the host.
 * @param B      p x m sparse off-diagonal block (TransOffDiagRed).
 * @param M      m x m sparse SPD translation block, gauge-pinned.
 * @param ctx    GPU context; all work is enqueued on ctx.stream.
 * @param Qsc    output, allocated here if needed: p*p doubles, column-major.
 * @param col_block  columns of Q_sc formed per batch. Peak scratch is
 *               2 * m * col_block doubles on top of the p^2 result, so this
 *               trades VRAM against the number of cuDSS solves.
 *
 * @throws std::runtime_error if cuDSS is unavailable (there is no device path
 *         for M^{-1} without it) or the dimensions disagree.
 */
void formDenseSchurOnDevice(const VarPro::SparseMatrix& Qmain,
                            const VarPro::SparseMatrix& B,
                            const VarPro::SparseMatrix& M,
                            GpuContext& ctx,
                            DeviceBuffer<double>& Qsc,
                            int col_block = 1024);

/**
 * @brief As above, but applying M^{-1} through a solver the caller already owns.
 *
 * Under IRLS the pattern of M is fixed and only its values change, so the
 * caller keeps one GpuSparseSolver alive and calls refactorize() between outer
 * iterations -- the cuDSS reordering and symbolic analysis then happen once
 * instead of once per iteration (Algorithm 2, lines 2 and 5).
 */
void formDenseSchurOnDevice(const VarPro::SparseMatrix& Qmain,
                            const VarPro::SparseMatrix& B,
                            GpuContext& ctx,
                            GpuSparseSolver& Msolver,
                            DeviceBuffer<double>& Qsc,
                            int col_block = 1024);

}  // namespace VarProGPU

#endif  // VARPRO_HAVE_CUDA
