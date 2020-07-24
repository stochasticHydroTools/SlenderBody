function [w1, w3, w5, specquad_needed,groots] = line3_near_weights_Cheb(tj, wj, xj, yj, zj, X, Y, Z, varargin)
%
% [w1,w3,w5] = line3_near_weights(tj, zj, yj, zj, X, Y, Z, [rho])
%
% Near eval quadrature weights for kernel 1/R^p, p=1,3,5
%
% INPUT:
% tj: Nodes in parametrization space [-1, 1] (type 1 Chebyshev points)
% wj: Quadrature weights accompanying tj
% xj, yj, zj: Points on curve, corresponding to tj
% X, Y, Z: List of target points
% rho: Critical Bernstein radius, weights computed for roots inside it.
%      Defaults to 4^(16/n).
%
% n = length of tj,wj,xj,yj,zj
%
% OUTPUT:
% w1,w3,w5: Vectors of size (n, numel(X)) with quadratures weights,
%           replacing wj if needed
% specquad_needed: List of bool values indicating if special quadrature weights
%                  were computed

% Reshape all inputs
tj = tj(:);
wj = wj(:);
xj = xj(:);
yj = yj(:);
zj = zj(:);

% Chebyshev expansions of discretization - using TYPE 1 points
n = length(xj);
th=flipud(((2*(0:n-1)+1)*pi/(2*n))');
Lmat = (cos((0:n-1).*th));
xhat = Lmat \ xj;
yhat = Lmat \ yj;
zhat = Lmat \ zj;

% Cap expansion at 16 coefficients.
% This seems to be more stable near boundary
if n > 16
    xhat = xhat(1:16);
    yhat = yhat(1:16);
    zhat = zhat(1:16);
end

% Set rho if not passed
if isempty(varargin)
    rho = 4^(16/n); % Sufficient for R5
else
    rho = varargin{1};
end

%% Rootfinding: initial guesses
all_tinits = complex(zeros(size(X)));
% Standard guess
for i=1:numel(X)    
    tinit = rootfinder_initial_guess(tj, xj, yj, zj, X(i), Y(i), Z(i)); % O(n) per point
    all_tinits(i) = tinit;
end

% First filter: Don't check points whose initial guess is far away
cp = (bernstein_radius(all_tinits) < 1.5*rho); % cp: Compute Points

%% Rootfinding: Run
all_roots = deal(complex(zeros(size(X))));
rootfinder_converged = false(size(X));
groots = zeros(numel(cp),1);
for j=find(cp(:)')
    tinit = all_tinits(j);
    [troot, converged] = rootfinderCheb(xhat, yhat, zhat,  X(j), Y(j), Z(j), tinit); % O(n) per point        
    all_roots(j) = troot;
    v=(cos((0:min(n,16)-1).*acos(troot)));
    groots(j) = norm(real([v*xhat v*yhat v*zhat]) -[X(j) Y(j) Z(j)])+...
        1i*norm(imag([v*xhat v*yhat v*zhat]));
    rootfinder_converged(j) = converged;    
end

% Check which points need special attention
all_bernstein_radii = bernstein_radius(all_roots);
specquad_needed = rootfinder_converged & (all_bernstein_radii < rho);

% Default is to return regular weights
[w1, w3, w5] = deal(repmat(wj(:), 1, numel(X)));

%% Compute special weights where needed
for i=1:numel(X)
    if specquad_needed(i)
        [w1(:,i), w3(:,i), w5(:,i)] = ...
            rsqrt_pow_weights(tj, all_roots(i)); 
    end
end