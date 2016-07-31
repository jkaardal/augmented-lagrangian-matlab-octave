# augmented-lagrangian-matlab-octave
<b>Augmented Lagrangian method for equality, inequality, and bounded optimization (MATLAB, Octave)</b>

This package contains an algorithm that solves for the local minima of problems of the form

    minimize f(x) subject to {ce(x) = 0}, {ci(x) >= 0}, and lb <= x <= ub
  
where f(x) is any differentiable linear or nonlinear function, {ce(x)} are a set of differentiable equality constraints, {ci(x)} are a set of differentiable inequality constraints, and lb and ub are constant lower and upper bounds on the variable(s), x. The augmented Lagrangian method is used to find a feasible local minimum of f(x) that satisfies the first order Karush-Kuhn-Tucker conditions. This particular implementation uses only first order minimization techniques and thus does not require computing the Hessian. Specifically, the augmented Lagrangian function is minimized using a projected gradient descent with intermediate updates to the Lagrange multipliers and penalty parameter. For more details on the theoretical background and algorithm, see the description in file almSolve.m 

Implementation details of the method may be found in the file almSolve.m and examples of how to run the method may be found in almTest.m. The script almTest.m demonstrates how to construct the problem struct, construct the options struct, and run the solver function in almSolve.m. The test problems in almTest.m also demonstrate application of the method to linear and nonlinear (quadratic and non-quadratic) programming problems and provide (brief) advice on tuning the optimization parameters for nonlinear minimizations.

The file inventory of this package should include:

Required

    almSolve.m
    almSearch.m
    almProj.m
  
Optional

    almTest.m
  
DISCLAIMER: Use at your own risk. The author is not liable for any loss or damages that may come as a result of interacting with the contents of this document and those referenced in this document. (The usual stuff; don't have unreasonable expectations about free and open source software!)
