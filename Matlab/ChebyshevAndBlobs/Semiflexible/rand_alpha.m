function [out] = rand_alpha(gamma,Nsamp)
    % gamma = kappa_b/(ds*kT);
    
    P = @(a,x) exp(-a * 2 * sin(x./2).^2).*sin(x);
    Nquad = 1e3;
    alpha = linspace(0,pi,Nquad);
    p = cumtrapz(alpha,P(gamma,alpha));
    p = p./p(end);
    
         
    trunc = find(p >= (1-1e-14));
    alpha_new = alpha;
    p_new = p;
    p_new(trunc) = [];
    alpha_new(trunc) = [];
    CDFinv = @(y) interp1(p_new, alpha_new, y, 'pchip','extrap');

    uni = rand(1,Nsamp);
    out = CDFinv(uni);
    
end

