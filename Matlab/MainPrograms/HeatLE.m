addpath '/home/om759/Documents/SLENDER_FIBERS/functions-solvers'
% Poisson in a parallelogram
N=4;
g=-0.25;
L=pi;
Ly=pi;
hx=L/N;
hy=Ly/N;
xE=(0:N-1)*hx;
yE=(0:N-1)*hy;
[x,y]=meshgrid(xE,yE);
kvx=[0:N/2-1 -N/2:-1]*2*pi/L;
kvy=[0:N/2-1 -N/2:-1]*2*pi/L;
[kx,ky]=meshgrid(kvx,kvy);
ksq = -kx.^2-(ky-g*kx).^2;
f = 4*pi^2*((g^2+2)*cos(2*pi*x/L).*sin(2*pi*y/Ly)-2*g*...
    sin(2*pi*x/L).*cos(2*pi*y/Ly))/(L^2);
fhat=fft2(f);
uhat=-fhat./ksq;
uhat(1,1)=0;
u=ifft2(uhat);
utrue = cos(2*pi*x/L).*sin(2*pi*y/Ly); 
max(abs(u-utrue))


% Poisson in periodic parallelogram using Ewald
Lx=2;
Ly=4;
Lz=2;
pts=[0 0 0; 0.1 0.6*pi 0];
charges=[1; -1];
% if (Ly > 2)
%     pts=[pts; 0 2 0; 1 2 0; 0.5 3 0; 1.5 3 0];
%     charges = [charges; -1; 1; -2; 2];
% end
xi=3;
tic
ufar1 = EwaldFarPoisson(pts,charges,Lx,Ly,Lz,g,xi);
toc
tic
ufar = EwaldFarPoissonFI(pts,charges,Lx,Ly,Lz,g,xi);
toc
max(abs(ufar1-ufar))
unear = EwaldNearPoisson(pts,charges,xi,Lx,Ly,Lz,g);
ptot = ufar+unear;
