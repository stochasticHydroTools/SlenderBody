function [I1, I3, I5] = rsqrt_pow_integrals(z,N)
% rsqrt_pow_integrals(z,N)
% Recursively compute values of integrals
% Ip(k) = \int_{-1}^{1} t^{k-1}/|t-z|^p dt
% for k=0,...,N-1 and z not in [-1,1]
%
% Recursions by Anna-Karin Tornberg & Katarina Gustavsson
% Journal of Computational Physics 215 (2006) 172–196
%
% Ludvig af Klinteberg, May 2018

% Test switch to disable power series eval
NO_POWER_SERIES = false;

% Test switch to disable half plane switch for I1(1)
NO_HP_SWITCH = false;

% Disable all tricks if called with VPA argument
if isa(z, 'sym')
    NO_POWER_SERIES = true;
    NO_HP_SWITCH = true;
end


zr = real(z);
zi = imag(z);
% (t-zr)^2+zi^2 = t^2-2*zr*t+zr^2+zi^2 = t^2 + b*t + c
b = -2*zr;
c = zr^2+zi^2;
d = zi^2; % d = c - b^2/4;

%u1 = sqrt(1 - b + c);
%u2 = sqrt(1 + b + c);
% The form below gets *much* better accuracy than ui = sqrt(ti^2 + b*ti + c)
u1 = sqrt((1+zr)^2 + zi^2);
u2 = sqrt((1-zr)^2 + zi^2);


% Compute I1
I1 = zeros(N,1);
if NO_HP_SWITCH
    % Vanilla expression
    I1(1) = log(1-zr+u2)-log(-1-zr+u1);    
else    
    % Evaluate after substitution zr -> -|zr|
    arg2 = 1+abs(zr) + sqrt((1+abs(zr))^2 + zi^2);          
    in_rhomb = 4*abs(zi) < 1-abs(zr);    
    if ~in_rhomb || NO_POWER_SERIES
        arg1 = -1+abs(zr) + sqrt((-1+abs(zr))^2 + zi^2);        
    else
        % Series evaluation needed inside 
        % rhombus [-1, i/4, 1, -i/4, -1].
        % Here arg1 has cancellation due to structure
        % -x + sqrt(x^2+b^2)
        Ns = 11;
        coeffs = coeffs_I1(Ns);
        arg1 = (1-abs(zr))*eval_series(coeffs, 1-abs(zr), zi, Ns);    
    end   
    I1(1) = log(arg2)-log(arg1);
end


if N>1
    I1(2) = u2-u1 - b/2*I1(1);    
end
s = 1;
for n=2:N-1
    s = -s; % (-1)^(n-1)
    I1(n+1) = (u2-s*u1 + (1-2*n)*b/2*I1(n) - (n-1)*c*I1(n-1))/n;
end
if nargout==1
    return
end

% Compute I3
I3 = zeros(N,1);
% Series is needed in cones extending around real axis from
% interval endpoints
w = min(abs(1+zr), abs(1-zr)); % distance interval-singularity
outside_interval = (abs(zr)>1);
% Alternative:
%w = abs(zr)-1;
%outside_interval = (0 < zi/w);
zi = abs(zi);
in_cone = (zi < 0.6*w);
use_series = (outside_interval && in_cone);
if ~use_series || NO_POWER_SERIES
    I3(1) = (b+2)/(2*d*u2) - (b-2)/(2*d*u1);    
else
    % Power series for shifted integral
    % pick reasonable number of terms
    if zi < 0.01*w
        Ns = 4;
    elseif zi < 0.1*w
        Ns = 10;
    elseif zi < 0.2*w
        Ns = 15;
    else % zi/w < 0.6
        Ns = 30;
    end
    coeffs = coeffs_I3(Ns);
    Fs = @(x) abs(x)/x^3 * (-0.5 +  eval_series(coeffs, x, zi, Ns));
    I3(1) = Fs(1-zr)-Fs(-1-zr);
end
if N>1
    I3(2) = 1/u1-1/u2 - b/2*I3(1);
end
for n=2:N-1
    I3(n+1) = I1(n-1) - b*I3(n) - c*I3(n-1);
end

if nargout==2
    return
end

% Compute I5
I5 = zeros(N,1);
% Here too use power series for first integral, in cone around real axis
in_cone = (zi < 0.7*w);
use_series = (outside_interval && in_cone);
if ~use_series  || NO_POWER_SERIES
    I5(1) = (2+b)/(6*d*u2^3) - (-2+b)/(6*d*u1^3) + 2/(3*d)*I3(1);
else
    % Power series for shifted integral
    if zi < 0.01*w
        Ns = 4;
    elseif zi < 0.2*w
        Ns = 10;
    elseif zi < 0.5*w
        Ns = 24;        
    elseif zi < 0.6*w
        Ns = 35;                
    else % zi/w < 0.7
        Ns = 50;
    end    
    coeffs = coeffs_I5(Ns);
    Fs = @(x) 1/(x^3*abs(x)) *(-0.25 + eval_series(coeffs, x, zi, Ns));    
    I5(1) = Fs(1-zr)-Fs(-1-zr);    
end

if N>1
    % Second integral computed using shifted version, and then corrected by first
    % integral (which is correctly evaluated using power series)
    % \int_{-1}^1 \frac{t \dif t}{ \pars{ (t-z_r)^2 + z_i^2) }^{5/2}} =
    % \int_{-1-z_r}^{1-z_r} \frac{t \dif t}{ (t^2 + z_i^2)^{5/2} } 
    % + z_r \int_{-1}^1 \frac{t \dif t}{ \pars{ (t-z_r)^2 + z_i^2) }^{5/2} }
    % This is analogous to the formula for I3(1), but was overlooked by Tornberg & Gustavsson
    I5(2) = 1/(3*u1^3) - 1/(3*u2^3)  - b/2*I5(1);
end


if N==1
    I5 = I5(1);
end
for n=2:N-1
    I5(n+1) = I3(n-1) - b*I5(n) - c*I5(n-1);
end
