function x = almSearch(L, dL, x0, lambda, rho, omega, lb, ub, Xtol, alpha, miter)
    % ALMSEARCH Find a local minimum of a function using projected gradient descent and
    %           and golden section line search.
    %
    %   Dependencies: almProj.m
    %
    %   Written by Joel T. Kaardal, July 7th, 2016

    MAXITS = miter;
    GOLD = (1+sqrt(5))/2;
    ALPHA = alpha;
    TOL = max([Xtol, omega]);

    grad = dL(x0, lambda, rho);
    x = almProj(x0, ALPHA*grad, lb, ub);
    its = 1;
    while norm(x-x0) > TOL
        % projected gradient descent with projected line search
        a = x0;
        c = almProj(x0, ALPHA*grad*(GOLD-1)/GOLD, lb, ub);
        d = almProj(x0, ALPHA*grad/GOLD, lb, ub);
        GAMMA = ALPHA;
        while norm(c-d) > TOL/2
            if L(c, lambda, rho) > L(d, lambda, rho)
                a = c;
            end
            GAMMA = GAMMA/GOLD;

            grad = dL(a, lambda, rho);
            c = almProj(a, GAMMA*grad*(GOLD-1)/GOLD, lb, ub);
            d = almProj(a, GAMMA*grad/GOLD, lb, ub);
        end

        x = (a+almProj(a, GAMMA*grad, lb, ub))/2;
        if its > MAXITS
            fprintf('Maximum iterations exceeded in almSearch.\n');
            break;
        end
        its = its+1;
        if norm(x-x0) > TOL
            x0 = x;
            grad = dL(x0, lambda, rho);
            %BETA = grad'*grad/(grad_old'*grad_old);
            x = almProj(x0, ALPHA*grad, lb, ub);
        end
    end
end
