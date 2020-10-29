% Least squares in L^2 to obtain Chebyshev coefficients of a function on a
% uniform grid
close all;

% Build resampling matrices
% Chebyshev coefficients to Chebyshev values
NCheb = 16;
th_Cheb =fliplr(2.0*(0:NCheb-1)'+1)*pi/(2*NCheb);
sCheb = cos(th_Cheb); % this will come in from chebfun in the real solver
% Chebyshev coefficients to uniform values
Nuni = 10000;
ds = 2/(Nuni-1);
sUni = (1:-ds:-1)';
CoeffstoValsUniform = cos(acos(sUni).* (0:NCheb-1));
% Take 16 randomly decaying Chebyshev coefficients and sample that
% Chebyshev series on the uniform grid
rng(0);
chat = rand(NCheb,1).*exp(-0.5*(0:NCheb-1)');
univals = CoeffstoValsUniform*chat; % values on uniform grid
univalsPerturbed = univals+(rand(Nuni,1)-0.5)*0.1; % randomly perturbed
plot(sUni,univalsPerturbed)
hold on
plot(sUni,univals,'LineWidth',2.0)
legend('Perturbed data','Original series')

% Compute coefficients by integration (Simpson's rule for theta)
% Weights for trapezoidal rule
[ChebVals,ChebIntsSimp] = ProjectToChebyshev(sCheb,sUni,univals,NCheb);
[ChebValsPert,ChebIntsSimpPert] = ProjectToChebyshev(sCheb,sUni,univalsPerturbed,NCheb);
% Error 
max(abs(ChebIntsSimp-chat))
% Plot normal and perturbed coefficients
figure;
semilogy(0:NCheb-1,abs(chat),'-o',0:NCheb-1,abs(ChebIntsSimp),'s',0:NCheb-1,abs(ChebIntsSimpPert),'d',...
    'MarkerSize',8,'LineWidth',2.0)
legend('Original','Simpson','Simpson for perturbed')    

