close all;
% Chebyshev coefficients to Chebyshev values
NCheb = 16;
th_Cheb =fliplr(2.0*(0:NCheb-1)'+1)*pi/(2*NCheb);
sCheb = cos(th_Cheb);
CoeffsToValsCheb = cos(th_Cheb.*(0:NCheb-1));
% Chebyshev coefficients to uniform values
Nuni = 100;
sUni = (1:-2/(Nuni-1):-1)';
CoeffstoValsUniform = cos(acos(sUni).* (0:NCheb-1));
% Matrix from Cheb values to uniform values
ChebtoUniform = CoeffstoValsUniform*CoeffsToValsCheb^-1;
% Check
% y = sCheb.^5-sCheb.^10+sCheb;
% max(abs(ChebtoUniform*y-(sUni.^5-sUni.^10+sUni)));
% Take 16 randomly decaying Chebyshev coefficients
rng(0);
chat = rand(NCheb,1).*exp(-0.1*(0:NCheb-1)');
univals = CoeffstoValsUniform*chat; % values on uniform grid
% Use backslash to get coefficients
chat2=CoeffstoValsUniform \ univals;
max(abs(chat-chat2))
% Try to get 24 coefficients
CoeffstoValsUniform24 = cos(acos(sUni).* (0:24-1));
chat24 = CoeffstoValsUniform24 \ univals;
max(abs(chat-chat24(1:NCheb))) % answer is the same!
% Now add some random noise on the uniform grid
univalsPerturbed = univals+rand(Nuni,1)*0.1;
plot(sUni,univals,sUni,univalsPerturbed,'LineWidth',2.0)
legend('Original function','With noise')
% Try to get 16 coefficients
chatpert=CoeffstoValsUniform \ univalsPerturbed;
% Try to get 24 coefficients
chatpert24 = CoeffstoValsUniform24 \ univalsPerturbed;
% Plot the spectrum
figure;
semilogy(0:NCheb-1,abs(chat),'-o',0:NCheb-1,abs(chatpert),'--s',0:24-1,abs(chatpert24),'-.d',...
    'LineWidth',2.0,'MarkerSize',8)
legend('Original spectrum','Perturbed spectrum (16 coeffs)','Perturbed spectrum (24 coeffs)')

% Alternative way with polyfit (equivalent to the above)
% p = polyfit(sUni,univalsPerturbed,NCheb-1);
% ChebVals_polyfit = polyval(p,sCheb);
% ChebCoeffs_polyfit = CoeffsToValsCheb \ ChebVals_polyfit;
% max(abs(ChebCoeffs_polyfit-chat))

% Compute coefficients by integration (equally spaced in theta)
% Weights for trapezoidal rule
thetasUni = acos(sUni);
dtheta = thetasUni(2:end)-thetasUni(1:end-1);
wtsTheta = [dtheta(1)/2; (dtheta(1:end-1)+dtheta(2:end))/2; dtheta(end)/2];
% Integrals are function values * wts * matrix of Chebyshev values
ChebInts = CoeffstoValsUniform'*(univals.*wtsTheta);
% Normalize by Chebyshev integrals 
CoeffsFromInt = ChebInts./[pi; pi/2*ones(NCheb-1,1)];
% These are the non perturbed ones
% You need LOTS of points to get the same accuracy because this is a
% second-order scheme
% Perturbed ones
ChebIntsPert = CoeffstoValsUniform'*(univalsPerturbed.*wtsTheta);
CoeffsPertFromInt = ChebIntsPert./[pi; pi/2*ones(NCheb-1,1)];