function f = eval_series(coeffs, x, b, Ns)
% Evaluate power series with terms
% c(p)*(b/x)^2p
%
    f = 0.0;
    bx2 = b*b/(x*x);
    bx2p = 1;
    for p = 1:Ns
        bx2p = bx2p*bx2;
        f = f + bx2p*coeffs(p);
    end
end