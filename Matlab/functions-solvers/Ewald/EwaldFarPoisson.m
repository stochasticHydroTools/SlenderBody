function ufarCH = EwaldFarPoisson(pts,charges,Lx,Ly,Lz,g,xi)
    gw=1/(2*xi);
    h=gw/1.6;
    nx=2^(nextpow2(Lx/h));
    ny=2^(nextpow2(Ly/h));
    nz=2^(nextpow2(Lz/h));
    hx=Lx/nx;
    hy=Ly/ny;
    hz=Lz/nz;
    xE=(0:nx-1)*hx;
    yE=(0:ny-1)*hy;
    zE=(0:nz-1)*hz;
    sup=nx; % Support of the Gaussian, 14 for 4 digits as Ewald changes (PLAY WITH THIS LATER)
    kvx=[0:nx/2-1 -nx/2:-1]*2*pi/Lx;
    kvy=[0:ny/2-1 -ny/2:-1]*2*pi/Ly;
    kvz=[0:nz/2-1 -nz/2:-1]*2*pi/Lz;
    [kx,ky,kz]=meshgrid(kvx,kvy,kvz);
%     tic
    S = SMatLE(xE,yE,zE,pts,sup,gw,g);
%     toc
%     tic
%     S2 = SMatLESlow(xE,yE,zE,pts,sup,gw,g);
%     toc
%     max(abs(S-S2))
    gridf=full(S*charges);
    gridf=permute(reshape(gridf,nx,ny,nz),[2 1 3]);
    fhat = fftn(gridf);
    ksq = -kx.^2-(ky-g*kx).^2-kz.^2;
    uhat = fhat./-ksq;
    uhat(1,1,1)=0;
    ufargrid=real(ifftn(uhat)); % is u on the slanted grid
    ufarCH=hx*hy*hz*S'*reshape(permute(ufargrid,[2 1 3]),nx*ny*nz,1);
end
