function [w1, w3, w5] = rsqrt_pow_weights(tj, troot, varargin)

n = numel(tj);


use_bjorck_pereyra = false; % Even faster than precomputed LU

[p1, p3, p5] = rsqrt_pow_integrals(troot, n);
%p3
basis = tj; 

% Compute "modified quadrature weights"
tdist = abs(tj-troot);
if n < 33 && use_bjorck_pereyra
    % Using Bjorck-Pereyra to solve Vandermonde system
    w1 = pvand(basis, p1) .* tdist; % O(n^2)
    w3 = pvand(basis, p3) .* tdist.^3;
    w5 = pvand(basis, p5) .* tdist.^5;
else
    % Direct solve with Vandermonde matrix is more stable for n>32
    % Still, n>40 is not a good idea    
    pvec = [p1 p3 p5];
    if isempty(varargin)
        A = ones(n);
        for j=2:n
            A(j,:) = A(j-1,:).*basis.'; % build transpose, O(n^2)
        end
        warning('off', 'MATLAB:nearlySingularMatrix')
        W = A \ pvec; % O(n^3)
        warning('on', 'MATLAB:nearlySingularMatrix')        
    else
        L = varargin{1};
        U = varargin{2};
        luvec = varargin{3};
        W = U\(L\pvec(luvec,:));
    end
    w1 = W(:,1) .* tdist;
    w3 = W(:,2) .* tdist.^3;
    w5 = W(:,3) .* tdist.^5;
end
