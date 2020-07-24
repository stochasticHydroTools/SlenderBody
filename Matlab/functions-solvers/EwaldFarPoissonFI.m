% Same file, but using FINUFFT library
function ufarCH = EwaldFarPoissonFI(ptsxyz,charges,Lx,Ly,Lz,g,xi) 
    % Compute the coordinates in the transformed basis
    L = [1 g 0; 0 1 0; 0 0 1];
    pts = (L \ ptsxyz')';
    % Rescale to [-pi,pi]
    pts = pts./[Lx Ly Lz]*2*pi;
    gw=1/(2*xi);
    h=gw/1.6;
    nx=2^(nextpow2(Lx/h));
    ny=2^(nextpow2(Ly/h));
    nz=2^(nextpow2(Lz/h));
    kvx=[0:nx/2-1 -nx/2:-1]*2*pi/Lx;
    kvy=[0:ny/2-1 -ny/2:-1]*2*pi/Ly;
    kvz=[0:nz/2-1 -nz/2:-1]*2*pi/Lz;
    [ky,kx,kz]=meshgrid(kvy,kvx,kvz);
    opts.modeord=1;
    fhat = finufft3d1(pts(:,1),pts(:,2),pts(:,3),charges,-1,1e-10,nx,ny,nz,opts);
    mksq = -kx.^2-(ky-g*kx).^2-kz.^2;
    uhat = fhat./-mksq.*exp(mksq/(4*xi^2));
    uhat(1,1,1)=0;
    ufarCH = real(finufft3d2(pts(:,1),pts(:,2),pts(:,3),1,1e-10,uhat,opts))/(Lx*Ly*Lz);
end
