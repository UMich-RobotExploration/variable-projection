#include <VarPro/Solver.h>
#include <VarPro/Utils.h>

#include <Optimization/Base/Concepts.h>
#include <Optimization/Riemannian/TNT.h>

void printIfVerbose(bool verbose, std::string msg)
{
  if (verbose)
  {
    std::cout << msg << std::endl;
  }
}

VarPro::Scalar thresholdVal(VarPro::Scalar val, VarPro::Scalar lower_bound,
                            VarPro::Scalar upper_bound)
{
  if (val < lower_bound)
  {
    return lower_bound;
  }
  else if (val > upper_bound)
  {
    return upper_bound;
  }
  else
  {
    return val;
  }
}

namespace VarPro
{

  ProblemResult solveProblem(Problem &problem, const Matrix &x0, bool verbose)
  {
    // check that x0 has the right number of rows
    if (problem.getFormulation() == Formulation::Explicit ||
        problem.getFormulation() == Formulation::ExplicitVarPro)
    {
      checkMatrixShape("solveCora::Explicit", problem.getDataMatrixSize(),
                       x0.cols(), x0.rows(), x0.cols());
    }
    else
    {
      std::cout << "Solving problem in translation implicit mode. Make sure that "
                   "the initial guess only contains rotation and range "
                   "variables."
                << std::endl;
      checkMatrixShape("solveCora::Implicit", problem.rotAndRangeMatrixSize(),
                       x0.cols(), x0.rows(), x0.cols());
    }

    // objective function
    Optimization::Objective<Matrix, Scalar, Matrix> f =
        [&problem](const Matrix &Y, const Matrix &NablaF_Y)
    {
      return problem.evaluateObjective(Y);
    };

    // quadratic model
    Optimization::Riemannian::QuadraticModel<Matrix, Matrix, Matrix> QM =
        [&problem](const Matrix &Y, Matrix &grad,
                   Optimization::Riemannian::LinearOperator<Matrix, Matrix,
                                                            Matrix> &HessOp,
                   Matrix &NablaF_Y)
    {
      // Compute and cache Euclidean gradient at the current iterate
      NablaF_Y = problem.Euclidean_gradient(Y);

      // Compute Riemannian gradient from Euclidean gradient
      grad = problem.Riemannian_gradient(Y, NablaF_Y);

      // Define linear operator for computing Riemannian Hessian-vector
      // products (cf. eq. (44) in the SE-Sync tech report)
      HessOp = [&problem](const Matrix &Y, const Matrix &Ydot,
                          const Matrix &NablaF_Y)
      {
        return problem.Riemannian_Hessian_vector_product(Y, NablaF_Y, Ydot);
      };
    };

    // variable projection function
    std::optional<SeparableStructureUpdate> separable_update = std::nullopt;
    if (problem.getFormulation() == Formulation::ExplicitVarPro)
    {
      separable_update = [&problem](Matrix &Y,
                                    Matrix &NablaF_Y)
      {
        problem.separableStructureUpdate(Y);
      };
    }

    // get retraction from problem
    Optimization::Riemannian::Retraction<Matrix, Matrix, Matrix> retract =
        [&problem](const Matrix &Y, const Matrix &V, const Matrix &NablaF_Y)
    {
      return problem.retract(Y, V);
    };

    // Euclidean gradient (is passed by reference to QM for caching purposes)
    Matrix NablaF_Y;

    // get preconditioner from problem
    std::optional<
        Optimization::Riemannian::LinearOperator<Matrix, Matrix, Matrix>>
        precon = [&problem](const Matrix &Y, const Matrix &Ydot,
                            const Matrix &NablaF_Y)
    {
      return problem.tangent_space_projection(Y, problem.precondition(Ydot));
    };

    // default TNT parameters for VARPRO
    Optimization::Riemannian::TNTParams<Scalar> params;
    params.Delta0 = 5;
    params.alpha2 = 3.0;
    params.max_TPCG_iterations = 80;
    params.max_iterations = 250;
    params.preconditioned_gradient_tolerance = 1e-6;
    params.gradient_tolerance = 1e-6;
    params.theta = 0.8;
    params.Delta_tolerance = 1e-5;
    params.verbose = show_iterates;
    params.precision = 2;
    params.max_computation_time = 20;
    params.relative_decrease_tolerance = 1e-6;
    params.stepsize_tolerance = 1e-6;
    params.log_iterates = false;

    // metric over the tangent space is the standard matrix trace inner product
    Optimization::Riemannian::RiemannianMetric<Matrix, Matrix, Scalar, Matrix>
        metric =
            [](const Matrix &Y, const Matrix &V1, const Matrix &V2,
               const Matrix &NablaF_Y)
    { return (V1.transpose() * V2).trace(); };

    // no custom instrumentation function for now
    std::optional<InstrumentationFunction> user_function = std::nullopt;

    TntResult result;
    Matrix X = problem.projectToManifold(x0);

    // solve the problem
    result = Optimization::Riemannian::TNT<Matrix, Matrix, Scalar, Matrix>(
        f, QM, metric, retract, X, NablaF_Y, precon, params, user_function, separable_update);
    printIfVerbose(verbose, "Obtained solution with objective value: " +
                                std::to_string(result.f));

    return std::make_pair(result, iterates);
  }

} // namespace VarPro
