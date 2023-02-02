% Testing
Lx = 1.5;
Ly = 2.0;
Lz = 2.4;
a=0.012;
mu = 1.5;
xi = 10;
pts = load('PtsEwald.txt');
forces = load('ForcesEwald.txt');
g=0.3;

[Npts,~]=size(pts);
tic
velfar = EwaldFarVel(pts,forces,mu,Lx,Ly,Lz,xi,a,g);
toc
tic
velfar2 = EwaldFarVelFI(pts,forces,mu,Lx,Ly,Lz,xi,a,g)';
toc
velNear = EwaldNearSum(Npts,pts,forces,xi,Lx,Ly,Lz,a,mu,g);
velEwald=velNear+velfar2;
UPy = load('EwaldVelocity.txt');
max(abs(velEwald-UPy))
