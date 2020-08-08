% Least squares in L^2 to obtain Chebyshev coefficients of a function on a
% uniform grid

% Build resampling matrices
% Chebyshev coefficients to Chebyshev values
NCheb = 16;
th_Cheb =fliplr(2.0*(0:NCheb-1)'+1)*pi/(2*NCheb);
sCheb = cos(th_Cheb);
CoeffsToValsCheb = cos(th_Cheb.*(0:NCheb-1));
% Chebyshev coefficients to uniform values
Nuni = 10000;
ds = 2/(Nuni-1);
sUni = (1:-ds:-1)';
CoeffstoValsUniform = cos(acos(sUni).* (0:NCheb-1));
% Matrix from Cheb values to uniform values
ChebtoUniform = CoeffstoValsUniform*CoeffsToValsCheb^-1;
% Check
% y = sCheb.^5-sCheb.^10+sCheb;
% max(abs(ChebtoUniform*y-(sUni.^5-sUni.^10+sUni)))
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
thetasUni = acos(sUni);
dthetaMax = acos(1-ds)-acos(1); % for convergence tracking
% Simpson's rule to compute integrals
ChebIntsSimp=zeros(NCheb,1);
ChebIntsSimpPert=zeros(NCheb,1);
for k=1:NCheb
    fk = CoeffstoValsUniform(:,k).*univals;
    fpertk = CoeffstoValsUniform(:,k).*univalsPerturbed;
    ChebIntsSimp(k) = simpson_nonuniform(acos(sUni),fk);
    ChebIntsSimpPert(k) = simpson_nonuniform(acos(sUni),fpertk);
end
% Normalize by Chebyshev integrals 
ChebIntsSimp = ChebIntsSimp./[pi; pi/2*ones(NCheb-1,1)];
ChebIntsSimpPert = ChebIntsSimpPert./[pi; pi/2*ones(NCheb-1,1)];
% Error 
max(abs(ChebIntsSimp-chat))
% Plot normal and perturbed coefficients
figure;
semilogy(0:NCheb-1,abs(chat),'-o',0:NCheb-1,abs(ChebIntsSimp),'s',0:NCheb-1,abs(ChebIntsSimpPert),'d',...
    'MarkerSize',8,'LineWidth',2.0)
legend('Original','Simpson','Simpson for perturbed')

function result = simpson_nonuniform(x, f)
    %Simpson rule for irregularly spaced data.
    N = length(x) - 1;
    h = x(2:end)-x(1:end-1);
    result = 0.0;
    for i=2:2:N
        hph = h(i) + h(i - 1);
        result=result+ f(i) * ( h(i).^3 + h(i - 1).^3+ 3. * h(i) * h(i - 1) * hph )...
                     / ( 6 * h(i) * h(i - 1));
        result =result+ f(i - 1) * ( 2. * h(i - 1).^3 - h(i).^3+ 3. * h(i) * h(i - 1).^2)...
                     / ( 6 * h(i - 1) * hph);
        result =result+ f(i + 1) * ( 2. * h(i).^3 - h(i - 1).^3 + 3. * h(i - 1) * h(i).^2)...
                     / ( 6 * h(i) * hph );
    end
    if (mod((N + 1),2) == 0)
        result =result+ f(N+1) * ( 2 * h(N).^2 + 3. * h(N-1) * h(N))/ ( 6 *( h(N-1) + h(N)));
        result =result+ f(N)*(h(N).^2+ 3*h(N)* h(N-1))/( 6 * h(N-1));
        result =result- f(N-1)* h(N).^3/( 6 * h(N-1) * (h(N-1) + h(N)));
    end
end