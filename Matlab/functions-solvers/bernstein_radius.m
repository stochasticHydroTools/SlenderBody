function rho = bernstein_radius(z) 
% rho = bernstein_radius(z) 
rho = abs(z + sqrt(z - 1).*sqrt(z+1));