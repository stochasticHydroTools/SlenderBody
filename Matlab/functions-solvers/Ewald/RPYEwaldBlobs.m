% Testing a configuration of blobs for both a rectangular domain and
% parallelogram domain (see Appendix B in paper)
rng(1);
% Points and forces
Npts=4;
pts=[0 0 0; 1 0 0; 0.5 1 0; 1.5 1 0];
forces=[1 1 1; -1 -1 -1; 2 2 2; -2 -2 -2];
mu=3;
% Define the grid
Lx=2;
Ly=2; % or 4 for rectangle
if (Ly > 2) % do the rectangular (larger domain)
    pts=[pts; 0 2 0; 1 2 0; 0.5 3 0; 1.5 3 0];
    forces = [forces; -1 -1 -1 ; 1 1 1; -2 -2 -2; 2 2 2];
    Npts = Npts*2;
end
Lz=2;
xi=5;
a=0.0245;
g=0.5+(Ly-2)*(-0.25);

velfar = EwaldFarVel(pts,forces,mu,Lx,Ly,Lz,xi,a,g);
velfar2 = EwaldFarVelGaussian2(pts,forces,mu,Lx,Ly,Lz,xi,a,g);
velNear = EwaldNearSum(Npts,pts,forces,xi,Lx,Ly,Lz,a,mu,g);
velEwald=velNear+velfar;

