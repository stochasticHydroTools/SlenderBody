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

function [ChebVals,ChebHats] = ProjectToChebyshev(sCheb,sUni,univals,NCheb)
    % Obtain the Chebyshev coefficients (and therefore Chebyshev values)
    % from the uniform values on the grid via an L^2 projection
    % s0 = coordinates of the Chebyshev nodes on [-1,1] 
    % sUni = coordinates of the uniform nodes on [-1,1], Univals =
    % values of a function on the uniform grid, NCheb = # of Chebyshev
    % polynomials
    % Return: values on the (type 1) Chebyshev grid and coefficients of the Chebyshev series 
    CoeffsToValsCheb = cos(acos(sCheb).*(0:NCheb-1));
    CoeffsToValsUniform = cos(acos(sUni).*(0:NCheb-1));
    ChebHats = zeros(NCheb,1);
    for k=1:NCheb % products and integrals in theta to get L^2 projection
        fk = CoeffsToValsUniform(:,k).*univals;
        ChebHats(k) = simpson_nonuniform(acos(sUni),fk);
    end
    ChebHats = ChebHats./[pi; pi/2*ones(NCheb-1,1)];
    ChebVals = CoeffsToValsCheb*ChebHats;
end
    

