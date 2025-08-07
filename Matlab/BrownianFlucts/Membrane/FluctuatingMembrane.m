L=3;
N=24;
dx=L/N;
x=(0:N-1)*dx;
[xg,yg]=meshgrid(x,x);
kvals = [0:N/2 -N/2+1:-1]*2*pi/L;
[kx,ky]=meshgrid(kvals);
ksq=kx.^2+ky.^2;
KSqDiag = diag(ksq(:));
FourierEnergyMat = ksq.^2*L^2/N^4

% Check calculation of energy
h = sin(2*pi*xg/L).*sin(4*pi*yg/L);
h = rand(N);
%h = ones(N);
hhat = fft2(h);
ksqhhat = conj(hhat).*FourierEnergyMat.*hhat;
% Integrate and square
SqEn = sum(ksqhhat(:))

% NUFFT to evaluate anywhere 
% IBpts = [0.378 0.642; 0.672 0.88; 0.547 0.02];
% hReal = sin(8*pi*IBpts(:,1)/L).*cos(12*pi*IBpts(:,2)/L);
% hInterp = InterpolatehNUFFT(h,IBpts,ksq,x);
% return

% Write this as a matrix
% Then you can do Brownian dynamics on h
FMatBase = dftmtx(N);
FMat2 = kron(FMatBase,FMatBase);
EnergyMatrix = real((FMat2'*(KSqDiag'*KSqDiag)*FMat2)/N^4*L^2);
%EnergySqRt = real(EnergyMatrix^(1/2));
SqEnFromMat = h(:)'*EnergyMatrix*h(:)

g =rand(N^2,1);
fg = EnergyMatrix*g;
g2 = reshape(g,N,N);
ghat = fft2(g2);
ghatF = ghat.*ksq.^2*dx^2;
fg2 = ifft2(ghatF);

TrueEn=100*pi^4/L^2

% Brownian dynamics
% Projector for edges
rng(2);
B = zeros(2*N,N^2);
for j = 1:N
    B(j,j)=1;
    B((j-1)*N+1,(j-1)*N+1)=1;
end
P = eye(N^2)-B'*pinv(B*B')*B;
M = P*eye(N^2);
Mhalf = P*eye(N^2);
dt = 0.1;
ImpMat = eye(N^2)/dt + M*EnergyMatrix;
InvImpMat = ImpMat^(-1); % fix this later to Fourier
ImpfacFourier = (1/dt+ksq.^2*dx^2*L^2);
kbT = 4.1e-3;
h = h(:);
Exphht= zeros(N^2);
nT=100000;
for iT=1:nT
    RHS = sqrt(2*kbT/dt)*Mhalf*randn(N^2,1);
    RHSHat = fft2(reshape(h/dt+RHS,N,N));
    hNewHat = RHSHat./ImpfacFourier;
    hnew2 = ifft2(hNewHat);
    hnew = InvImpMat*(h/dt+RHS);
    %hnew = hnew - mean(hnew(:));
    surf(xg,yg,reshape(h,N,N))
    zlim([-0.02 0.02])
   drawnow
    h = hnew(:);
    if (iT > 1000)
        Exphht=Exphht+h*h';
    end
end

