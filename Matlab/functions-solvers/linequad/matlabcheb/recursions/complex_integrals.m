function varargout = complex_integrals(z,N)
% [IL, I1, I2, I3, ...] = complex_integrals(z, N)
%
% Recursively compute values integrals
% IL(k) = \int_{-1}^{1} t^{k-1} log(t-z) dt    
% Ip(k) = \int_{-1}^{1} t^{k-1}/(t-z)^p dt
% for k=1,...,N and complex z not in the interval [-1,1]
%
% If z is a vector [z1, z2], then the integrals are computed for kernels
% log(t-z1) + log(t-z2)
% and 
% ((t-z1)(t-z2))^-p
% (Only implemented up to p=2)
%
% Based on recursions by Johan Helsing et al.
%
% Ludvig af Klinteberg, 2018

zinput = z;
    
% One pole
z = zinput(1);    
M = max(1, nargout-1);
p = cell(M, 1);
q = zeros(N, 1, 'like', z);
% Rational
for m=1:M
    if m==1
        p{m} = zeros(N+1, 1, 'like', z); % Need one more element for log recursions
        p{1}(1) = log((1-z) / (-1-z));
        for k=1:N
            p{1}(k+1) = z*p{1}(k) + (1-(-1)^k)/k;
        end        
    else    
        p{m} = zeros(N, 1, 'like', z);
        p{m}(1) = ((1-z)^(1-m) - (-1-z)^(1-m))/(1-m);
        for k=1:N-1
            p{m}(k+1) = z*p{m}(k) + p{m-1}(k);
        end        
    end
end
% Log
for k=1:N
    q(k) = (log(1-z) - (-1)^k*log(-1-z) - p{1}(k+1))/k;
end
p{1} = p{1}(1:N); % Remove extra element

% Exit?
if numel(zinput)==1
    % Single pole, we're done
    varargout{1} = q;
    [varargout{2:nargout}] = p{1:nargout-1};
    return
end
if numel(z) > 2
    error('Not implemented for more than two poles')
end

% Two poles
% TODO: This does not work for z1, z2 in all quadrants, 
% but maybe we can assume Im z1 = Im z2, in which case it seems to work
z1 = zinput(1);
z2 = zinput(2);
p2 = cell(M, 1);
% Log
q2 = q + complex_integrals(z2, N);
% Rational
for m=1:M    
    p2{m} = zeros(N, 1, 'like', z);    
    if m==1
        p2{1}(1) = ( log(1-z1)-log(-1-z1) - (log(1-z2)-log(-1-z2)) )/(z1-z2);
        for k=1:N-1
            p2{1}(k+1) = p{1}(k) + z2*p2{1}(k);
        end        
    else    
        if m==2
            F1 = @(t) ((-z1 + z2)/(t - z1) + (-z1 + z2)/(t - z2) - 2*log(t - z1) + 2*log(t ...
                                                              - z2))/(z1 - z2)^3;
            F2 = @(t) ((z1*(-z1 + z2))/(t - z1) + (z2*(-z1 + z2))/(t - z2) - (z1 + ...
                                                              z2)*log(t - z1) + (z1 + z2)*log(t - z2))/(z1 - z2)^3;
            p2{m}(1) = F1(1) - F1(-1);
            p2{m}(2) = F2(1) - F2(-1);
        elseif m==3
            F1 = @(t) (((z1 - z2)*(-2*t + z1 + z2)*(-6*t^2 + z1^2 - 8*z1*z2 + z2^2 + 6*t*(z1 + z2)))/((t - z1)^2*(t - z2)^2) + 12*log(t - z1) - 12*log(t - z2))/(2.*(z1 - z2)^5);
            F2 = @(t) (-(((z1 - z2)*(-6*t^3*(z1 + z2) + 9*t^2*(z1 + z2)^2 - 2*t*(z1 + z2)*(z1^2 + 7*z1*z2 + z2^2) + z1*z2*(z1^2 + 10*z1*z2 + z2^2)))/((t - z1)^2*(t - z2)^2)) + 6*(z1 + z2)*(log(t - z1) - log(t - z2)))/(2.*(z1 - z2)^5);
            p2{m}(1) = F1(1) - F1(-1);
            p2{m}(2) = F2(1) - F2(-1);
            
        else
            % Need the rest...
        end
        for k=2:N-1
            p2{m}(k+1) = p2{m-1}(k-1) + (z1+z2)*p2{m}(k) - z1*z2*p2{m}(k-1);
        end        
    end
end
% Exit
varargout{1} = q2;
[varargout{2:nargout}] = p2{1:nargout-1};

end