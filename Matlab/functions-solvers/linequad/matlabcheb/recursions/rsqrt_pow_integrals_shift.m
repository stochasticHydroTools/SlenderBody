function [I1, I3, I5] = rsqrt_pow_integrals_shift(z,N)
% rsqrt_pow_integrals(z,N)
% Recursively compute values of shifted integrals
% Ip(k) = \int_{-1}^{1} (t-a)^{k-1}/|t-z|^p dt
% for k=0,...,N-1 and z=a+bi not in the interval [-1,1]
%
% - Shift increases accuracy near endpoints and away from the interval
% - I1(1) is stabilized by putting z in left half plane
% - I3(1) and I5(1) are computed using power series in b in cones
%   around real line extending from endpoints.
%
% Based on recursions by Anna-Karin Tornberg & Katarina Gustavsson
% Journal of Computational Physics 215 (2006) 172–196

% Test switch to disable power series eval
NO_POWER_SERIES = false;

% Setup variables
a = real(z);
b = imag(z);
t1 = -1-a;
t2 = 1-a;
u1 = sqrt(t1^2 + b^2);
u2 = sqrt(t2^2 + b^2);

% Compute I1
I1 = zeros(N,1);
% First integral is more stable in left half plane
if a<0
    I1(1) = log(t2+u2)-log(t1+u1);
else    
    % Swap
    I1(1) = log(-t1+u1)-log(-t2+u2);        
end
if N>1
    I1(2) = u2-u1;    
end
t1nm1 = 1; % t^(n-1)
t2nm1 = 1; % t^(n-1)
for n=2:N-1
    t1nm1 = t1nm1*t1;
    t2nm1 = t2nm1*t2;
    I1(n+1) = (t2nm1*u2-t1nm1*u1 - (n-1)*b^2*I1(n-1))/n;
end
if nargout==1
    return
end

% Compute I3
I3 = zeros(N,1);
% Series is needed in cones extending around real axis from
% interval endpoints
w = min(abs(t1), abs(-t2)); % distance interval-singularity
b = abs(b);
in_cone = (b/w < 0.5);
outside_interval = (t2 < 0 || t1 > 0);
use_series = (outside_interval && in_cone);
if ~use_series || NO_POWER_SERIES
    I3(1) = t2/(b^2*u2) - t1/(b^2*u1);    
else
    % Power series
    % pick reasonable number of terms
    if b/w < 0.01
        Ns = 4;
    elseif b/w < 0.1
        Ns = 10;
    elseif b/w < 0.2
        Ns = 15;
    else % zi/w < 0.5
        Ns = 24;
    end
    coeffs = coeffs_I3(Ns);
    F0 = @(x) -1/2*x/abs(x)^3;    
    Fs = @(x) F0(x) + abs(x)/x^3 * eval_series(coeffs, x, b, Ns);
    I3(1) = Fs(t2)-Fs(t1);
end
if N>1
    I3(2) = 1/u1-1/u2;
end
for n=2:N-1            % note seems like I3(3) loses digits, even at a=0
  I3(n+1) = I1(n-1) - b^2*I3(n-1);    % and this loss all due to I1(1).
end

if nargout==2
    return
end

% Compute I5
I5 = zeros(N,1);
% Here too use power series for first integral, in cone around real axis
in_cone = (b/w < 0.7);
use_series = (outside_interval && in_cone);
if ~use_series  || NO_POWER_SERIES
    I5(1) = t2/(3*b^2*u2^3) - t1/(3*b^2*u1^3) + 2/(3*b^2)*I3(1);
else
    % Power series
    if b/w < 0.01
        Ns = 4;
    elseif b/w < 0.2
        Ns = 10;
    elseif b/w < 0.5
        Ns = 24;        
    elseif b/w < 0.6
        Ns = 35;                
    else % zi/w < 0.7
        Ns = 50;
    end    
    coeffs = coeffs_I5(Ns);
    F0 = @(x) -1/4*x/abs(x)^5;    
    Fs = @(x) F0(x) + 1/(x*abs(x)^3) * eval_series(coeffs, x, b, Ns);    
    I5(1) = Fs(t2)-Fs(t1);    
end
if N>1
    I5(2) = 1/(3*u1^3) - 1/(3*u2^3);
end
for n=2:N-1
    I5(n+1) = I3(n-1) - b^2*I5(n-1);
end

end