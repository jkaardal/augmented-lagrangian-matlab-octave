function pr = almProj(x, dL, lb, ub)
    % ALMPROJ   Given a gradient vector of a function, calculate the projected gradient
    %           into the subspace defined by the lower and upper bounds on x (when finite).
    %
    %   Written by Joel T. Kaardal, July 7th, 2016

    pr = x - dL;
    llb = ~(isnan(lb) | isinf(lb)) & (pr <= lb);
    pr(llb) = lb(llb);
    lub = ~(isnan(ub) | isinf(ub)) & (pr >= ub);
    pr(lub) = ub(lub);

end
