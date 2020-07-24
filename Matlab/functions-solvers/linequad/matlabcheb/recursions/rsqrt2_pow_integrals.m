function [I2] = rsqrt2_pow_integrals(z, N)
% rsqrt_pow_integrals(z,N)
% Recursively compute values of integrals
% Ip(k) = \int_{-1}^{1} t^{k-1}/|t-z|^p dt
% for k=0,...,N-1 and z not in [-1,1]
% and p even 
%
% Ludvig af Klinteberg

I2 = zeros(N,1);
a = real(z);
b = imag(z);

I2(1) = (atan((a+1)/b)-atan((a-1)/b))/b;
I2(2) = log( ((a-1)^2+b^2) / ((a+1)^2+b^2) )/2 + a*I2(1);
for j=2:N-1
    I2(j+1) = 2*a*I2(j) - (a^2+b^2)*I2(j-1) + (1+(-1)^j)/(j-1) ;    
end
