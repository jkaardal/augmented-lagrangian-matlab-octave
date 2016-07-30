function [x, fval, lambda, kkt] = almSolve(p, x0, lambda0, options)
    % ALMSOLVE  Find a local minimum of a function subject to equality and inequality constraints
    %           using the Augmented Lagrangian method. This function approximates the solution to
    %           linear, quadratic, and nonlinear programming problems of the form
    %
    %               minimize f(x) with respect to x
    %               subject to {ce(x) = 0}, {ci(x) >= 0}, and lb <= x <= ub
    %
    %           where f(x) is a continuously differentiable function, {ce(x)} are a set of
    %           equality constraint conditions, {ci(x)} are a set of inequality constraint
    %           conditions, and lb and ub are a set of constant lower and upper bounds,
    %           respectively, on the solution x. For details on the method, see bottom.
    %
    %   [x, fval, lambda, kkt] = almSolve(p, x0) solves the problem contained in the problem
    %           struct p with initial values of the parameters, x0. Lagrange multipliers are
    %           are automatically initialized using the Moore-Penrose pseudoinverse.
    %   [x, fval, lambda, kkt] = almSolve(p, x0, lambda0) solves the problem contained in problem
    %           struct p with initial values of the parameters, x0, and Lagrange multipliers,
    %           lambda0. Both x0 and lambda0 must be column vectors.
    %   [x, fval, lambda, kkt] = almSolve(p, x0, lambda, options) solves the problem using custom
    %           values for various optimization parameters contained in the options struct.
    %   [x, fval, lambda, kkt] = almSolve(___) returns x, the optimal solution or the last
    %           solution before exceeding the maximum number of interations; fval, f(x) at
    %           solution x; lambda, the Lagrange multipliers at solution x; kkt, the first
    %           order Karush-Kuhn-Tucker (KKT) conditions. The output kkt is a cell array with
    %           three elements: kkt{1} are the stationarity conditions, kkt{2} are the equality
    %           constraint conditions (if any, otherwise kkt{2}=0), and kkt{3} are the inequality
    %           constraint conditions (if any, otherwise kkt{3}=0). For more details on the KKT
    %           conditions, see bottom.
    %
    %   The problem struct, p, contains the details of the constrained minimization problem to be
    %   solved expressed through the following struct fields. The required fields are 'f' and 'df'
    %   while the rest may be omitted if unneeded. However, if 'ce' is provided, 'dce' must also
    %   be provided. Similarly, if 'ci' is provided, 'dci' must be provided.
    %
    %       'f' is the function to be minimized. The input of this function is a column vector
    %           of the x coordinates and the output is a scalar.
    %       'df' computes derivatives of f(x) with respect to x. The input of this function is
    %           a column vector of the x coordinates and the output is a vector of size(x).
    %       'ce' is a function that computes the equality constraint conditions {ce(x)}. The
    %           input is a column vector of the x coordinates and the output is a row vector
    %           where each element corresponds to the scalar value of each equality constraint
    %           condition.
    %       'dce' is a function that computes the Jacobian matrix of the equality constraints,
    %           {ce(x)}, with respect to the x coordinates. The input of the function is a
    %           column vector of the x coordinates and the output is a matrix that is size(x) by
    %           size(ce). Each column corresponds to an equality constraint while each row
    %           corresponds to a derivative with respect to each of the x coordinates.
    %       'ci' is a function that computes the inequality constraint conditions {ci(x)}. The
    %           input is a column vector of the x coordinates and the output is a row vector
    %           where each element corresponds to the scalar value of each inequality constraint
    %           condition.
    %       'dci' is a function that computes the Jacobian matrix of the inequality constraints,
    %           {ci(x)}, with respect to the x coordinates. The input of the function is a
    %           column vector of the x coordinates and the output is a matrix that is size(x) by
    %           size(ci). Each column corresponds to an inequality constraint while each row
    %           corresponds to a derivative with respect to each of the x coordinates.
    %       'lb' is a column vector that holds constant lower bounds of x. Coordinates that do
    %           not have a lower bound have corresponding elements in lb set to NaN or Inf.
    %           By default, the x coordinates are unbounded.
    %       'ub' is a column vector that holds constant upper bounds of x. Coordinates that do
    %           not have an upper bound have corresponding elements in ub set to NaN or Inf.
    %           By default, the x coordinates are unbounded.
    %
    %   The options struct, options, contains values for optimization parameters that can be
    %   tuned by the user. Any field that is omitted reverts to the default value.
    %
    %       'rho' is the initial value of the penalty parameter. By default, rho = 10.
    %       'gma' is the factor by which the penalty parameter, rho, is incremented when an
    %           iterate breaks feasibility. If a constraint is violated, then rho = gma*rho.
    %           By default, gma = 10. This variable will likely need adjustment for nonlinear
    %           problems.
    %       'eta' is the initial infeasibility tolerance. If there are only equality
    %           constraints and no bounds, the default value is eta = 0.25 and eta is
    %           held constant. Otherwise, the default value of eta = 1/rho^0.1 and the
    %           value is decreased when the iterate is infeasible.
    %       'omega' is a convergence tolerance for the projected gradient descent that is
    %           not used when the problem has only equality constraints and no inequality
    %           constraints or bounds. The default of value is omega = 1/rho.
    %       'miter' is the maximum number of gradient descent steps in the minimization of
    %           f(x) subject to constraints for a given set of Lagrange multipliers and
    %           the above optimization variables. Default miter = 500.
    %       'niter' is the maximum number of minimization steps where the Lagrange
    %           multipliers, rho, eta, and omega are modified, when relevant. Default
    %           niter = 50.
    %       'Xtol' is the minimum convergence tolerance in the gradient descent. Default is
    %           Xtol = 1E-10.
    %       'Ktol' is the minimum convergence tolerance on the KKT conditions. Default is
    %           Ktol = 1E-4.
    %       'alpha' is the maximum step length (alpha * gradient). Default is alpha = 1.0.
    %           Changing this variable is not recommended.
    %
    %   To see implementation examples, refer to almTest.m.
    %
    %   <<< THE BOTTOM >>>
    %
    %   --Brief summary of the optimization--
    %
    %   Constrained optimization can be transformed into an unconstrained optimization by
    %   introducing Lagrange multipliers, lambda, and forming the Lagrangian
    %
    %       L = f - [ce, ci-s]*lambda
    %
    %   where slack variables, s >= 0, are introduced and appended to x (these are added
    %   automatically). The augmented Lagrangian method adds an additional 2-norm
    %   penalty term to the Lagrangian
    %
    %       L_a = L + rho/2*norm([ce, ci-s]).
    %
    %   This algorithm minimizes L_a.
    %
    %   When initial Lagrange multipliers are not provided, the algorithm initializes
    %   lambda0 as the minimum norm solution of the gradient of L using the Moore-Penrose
    %   pseudoinverse.
    %
    %   The first order Karush-Kuhn-Tucker (KKT) conditions are defined as follows for the
    %   augmented Lagrangian:
    %
    %       kkt{1} = df - [dce*, dci*]*lambda + rho*[dce*, dci*]*[ce, ci*]';
    %       kkt{2} = ce;
    %       kkt{3} = ci;
    %
    %   and all should be approximately equal to zero (norm(kkt{1}) <= Ktol,
    %   norm(kkt{2}) <= Ktol, norm(kkt{3}) <= Ktol) at an optimal solution (dce*, dci* and ci*
    %   signify that slack variables have been included in the constraints). Note that if
    %   there are inequality constraints, kkt{1} has the slack variable stationarity
    %   conditions appended at the bottom of the column vector; e.g. if x is 3 dimensional
    %   and there are 2 inequality constraints, length(kkt{1}) = 5.
    %
    %   This algorithm is based on the augmented Lagrangian method from
    %       "Numerical Optimization" by Jorge Nocedal & Stephen J. Wright, 2nd Ed.
    %   For more information, refer to this reference.
    %
    %   Dependencies: almProj.m, almSearch.m
    %   Optional: almTest.m
    %
    %   Written by Joel T. Kaardal, July 7th, 2016

    algorithm = 1;
    if nargin < 2
        error('Not enough arguments (2 required).');
    end
    if ~sum(strcmp(fieldnames(p), 'f')) || ~sum(strcmp(fieldnames(p), 'df'))
        error('Problem struct must contain fields f and df.');
    else
        if isempty(p.f) || isempty(p.df)
            error('Problem struct fields f and df that are non-empty.');
        end
    end

    f = p.f;
    df = p.df;

    if sum(strcmp(fieldnames(p), 'ci')) && sum(strcmp(fieldnames(p), 'dci'))
        if isempty(p.ci) || isempty(p.dci)
            error('Problem struct contains fields ci and dci but one or both are empty.');
        end
        algorithm = 2;
        ci = p.ci;
        dci = p.dci;
    else
        ci = [];
        dci = [];
    end
    if sum(strcmp(fieldnames(p), 'lb'))
        if ~isempty(p.lb)
            algorithm = 2;
        end
        lb = p.lb;
    else
        lb = [];
    end
    if sum(strcmp(fieldnames(p), 'ub'))
        if ~isempty(p.ub)
            algorithm = 2;
        end
        ub = p.ub;
    else
        ub = [];
    end

    if sum(strcmp(fieldnames(p), 'ce')) && sum(strcmp(fieldnames(p), 'dce'))
        if algorithm ~= 2
            algorithm = 3;
        end
        ce = p.ce;
        dce = p.dce;
    else
        ce = [];
        dce = [];
    end

    if nargin < 4
        if algorithm ~= 1
            gma = 10;
            if algorithm == 2
                rho = 10;
                eta = 1/rho^0.1;
                omega = 1/rho;
            else
                rho = 10;
                eta = 0.25;
            end
        end
        alpha = 1.0;
        Xtol = 1E-10;
        Ktol = 1E-4;
        niter = 50;
        miter = 500;
    else
        if sum(strcmp(fieldnames(options), 'gma'))
            gma = options.gma;
        else
            gma = 10;
        end
        if sum(strcmp(fieldnames(options), 'rho'))
            rho = options.rho;
        else
            if algorithm == 2
                rho = 10;
            else
                rho = 10;
            end
        end
        if sum(strcmp(fieldnames(options), 'eta'))
            eta = options.eta;
        else
            if algorithm == 2
                eta = 1/rho^0.1;
            else
                eta = 0.25;
            end
        end
        if sum(strcmp(fieldnames(options), 'omega'))
            omega = options.omega;
        else
            if algorithm == 2
                omega = 1/rho;
            end
        end
        if sum(strcmp(fieldnames(options), 'alpha'))
            alpha = options.alpha;
        else
            alpha = 1.0;
        end
        if sum(strcmp(fieldnames(options), 'Xtol'))
            Xtol = options.Xtol;
        else
            Xtol = 1E-10;
        end
        if sum(strcmp(fieldnames(options), 'Ktol'))
            Ktol = options.Ktol;
        else
            Ktol = 1E-4;
        end
        if sum(strcmp(fieldnames(options), 'niter'))
            niter = options.niter;
        else
            niter = 50;
        end
        if sum(strcmp(fieldnames(options), 'miter'))
            miter = options.miter;
        else
            miter = 500;
        end
    end

    if algorithm == 1
        fprintf('Warning: unconstrained problems are better solved with fminunc.m instead of almSolve.m.\n');
        x = almSearch(f, df, x, [], 0.0, omega, lb, ub, Xtol, alpha, miter);
        fval = f(x);
        lambda = [];
        kkt{1} = df(x);
        kkt{2} = 0;
        kkt{3} = 0;
    elseif algorithm == 2
        nvar = length(x0);
        if ~isempty(ce) && ~isempty(dce)
            neq = length(ce(x0));
        else
            neq = 0;
        end
        if ~isempty(ci) && ~isempty(dci)
            nineq = length(ci(x0));
        else
            nineq = 0;
        end

        F = @(s)(f(s(1:nvar)));
        if nineq
            dF = @(s)([df(s(1:nvar)); zeros(nineq,1)]);
        else
            dF = @(s)(df(s(1:nvar)));
        end
        if neq && nineq
            C = @(s)([ce(s(1:nvar)), ci(s(1:nvar))-s(nvar+1:end)']);
            dC = @(s)([[dce(s(1:nvar)), dci(s(1:nvar))]; [zeros(nineq, neq), -eye(nineq)]]);
        elseif nineq
            C = @(s)(ci(s(1:nvar))-s(nvar+1:end)');
            dC = @(s)([dci(s(1:nvar)); -eye(nineq)]);
        elseif neq
            C = @(s)(ce(s(1:nvar)));
            dC = @(s)(dce(s(1:nvar)));
        else
            C = [];
            dC = [];
        end
        L = @(s, lambda, rho)(F(s) - C(s)*lambda + rho/2*norm(C(s))^2);

        if ~isempty(lb)
            lb = [lb; zeros(nineq,1)];
        else
            lb = [NaN(nvar,1); zeros(nineq,1)];
        end
        if ~isempty(ub)
            ub = [ub; NaN(nineq,1)];
        else
            ub = NaN(nvar+nineq,1);
        end
        DxL = @(s, lambda, rho)(dF(s) - dC(s)*lambda + rho*dC(s)*C(s)');
        DxPr = @(s, lambda, rho)(almProj(s, DxL(s, lambda, rho), lb, ub));

        P = @(s, lambda, rho)deal(L(s, lambda, rho), s-DxPr(s, lambda, rho));
        if neq && nineq
            KKT = @(s, lambda, rho)deal(df(s(1:nvar)) - [dce(s(1:nvar)), dci(s(1:nvar))]*lambda, ce(s(1:nvar)), ci(s(1:nvar)));
        elseif nineq
            KKT = @(s, lambda, rho)deal(s - DxPr(s, lambda, rho), 0.0, ci(s(1:nvar)));
        elseif neq
            KKT = @(s, lambda, rho)deal(df(s(1:nvar)) - dce(s(1:nvar))*lambda, ce(s(1:nvar)), 0.0);
        else
            KKT = @(s, lambda, rho)deal(df(s(1:nvar)), 0.0, 0.0);
        end

        if nineq
            s = ci(x0);
            s = max([s; ones(size(s))*eps])';
            s = [x0; s];
        else
            s = x0;
        end
        if nargin < 3 || isempty(lambda0)
            lambda = pinv(dC(s))*dF(s);
        else
            lambda = lambda0;
        end
        kkt = cell(3,1);

        cverg = false;
        for iter = 1:niter
            fprintf('ITERATION = %d\n', iter);

            s = almSearch(L, DxL, s, lambda, rho, omega, lb, ub, Xtol, alpha, miter);

            [kkt{1}, kkt{2}, kkt{3}] = KKT(s, lambda, rho);

            if norm(C(s)) <= max([eta, Ktol])
                if norm(C(s)) <= Ktol && norm(kkt{1}) <= Ktol
                    cverg = true;
                    break;
                elseif iter < niter
                    lambda = lambda - rho*C(s)';
                    eta = eta/rho^0.9;
                    omega = omega/rho;
                end
            elseif iter < niter
                rho = gma*rho;
                eta = eta/rho^0.1;
                omega = 1/rho;
            end
        end

        fval = L(s, lambda, rho);
        x = s(1:nvar);
    elseif algorithm == 3
        L = @(x, lambda, rho)(f(x) - ce(x)*lambda + rho/2*norm(ce(x))^2);
        DxL = @(x, lambda, rho)(df(x) - dce(x)*lambda + rho*dce(x)*ce(x)');

        KKT = @(x, lambda, rho)deal(df(x) - dce(x)*lambda, ce(x), 0.0);

        x = x0;
        if nargin < 3 || isempty(lambda0)
            lambda = pinv(dce(x))*df(x);
        else
            lambda = lambda0;
        end
        v = norm(ce(x))^2;
        kkt = cell(2,1);

        lb = NaN(size(x));
        ub = NaN(size(x));

        cverg = false;
        for iter = 1:niter
            fprintf('ITERATION = %d\n', iter);

            x = almSearch(L, DxL, x, lambda, rho, Xtol, lb, ub, Xtol, alpha, miter);

            [kkt{1}, kkt{2}, kkt{3}] = KKT(x, lambda, rho);

            if norm(kkt{1}) <= Ktol && norm(kkt{2}) <= Ktol
                cverg = true;
                break;
            end

            v0 = norm(ce(x))^2;
            if v0 < eta*v
                lambda = lambda - rho*ce(x)';
                v = v0;
            else
                rho = gma*rho;
            end
        end

        fval = L(x, lambda, rho);
    end

    if cverg
        fprintf('CONVERGED\n');
    else
        fprintf('ITERATIONS EXCEEDED\n');
    end
end

