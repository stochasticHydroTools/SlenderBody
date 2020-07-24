% Far field velocity with PSE RPY tensor
function velfar = EwaldFarVel(pts,forces,mu,Lx,Ly,Lz,xi,a,g)
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
    sup=14; % Support of the Gaussian, 12 for 3 digits as Ewald changes
    kvx=[0:nx/2-1 -nx/2:-1]*2*pi/Lx;
    kvy=[0:ny/2-1 -ny/2:-1]*2*pi/Ly;
    kvz=[0:nz/2-1 -nz/2:-1]*2*pi/Lz;
    [kx,ky,kz]=meshgrid(kvx,kvy,kvz);
    %k=sqrt(kx.^2+ky.^2+kz.^2);
    k = sqrt(kx.^2+(ky-g*kx).^2+kz.^2);
    %S = SMat(xE,xE,xE,pts,sup,gw);
    S = SMatLE(xE,yE,zE,pts,sup,gw,g);
    [gridfx, gridfy, gridfz] = spread(S,forces,nx,ny,nz);
    fxhat = fftn(gridfx);
    fyhat = fftn(gridfy);
    fzhat = fftn(gridfz);
    % Factor is Eq. (9) in the PSE paper, but removing the Gaussian
    % since we are doing that in the spread and interpolate steps
    % Careful: matlab has an extra pi in the sinc function
    factor = 1/mu*1./k.^2.*(sinc(k*a/pi).^2).*(1+(k.^2)/(4*xi^2));
    factor(1,1,1)=0;
    uxhat=factor.*fxhat;
    uyhat=factor.*fyhat;
    uzhat=factor.*fzhat;
    % Project off to get a divergence free field
    for iX=1:nx
        for iY=1:ny
            for iZ=1:nz
                if (k(iY,iX,iZ)>0)
                    ks=[kx(iY,iX,iZ) ky(iY,iX,iZ)-g*kx(iY,iX,iZ) kz(iY,iX,iZ)];
                    ks=ks/norm(ks);
                    usofar=[uxhat(iY,iX,iZ); uyhat(iY,iX,iZ); ...
                        uzhat(iY,iX,iZ)];
                    u_off = (eye(3)-(ks'*ks))*usofar;
                    uxhat(iY,iX,iZ)=u_off(1);
                    uyhat(iY,iX,iZ)=u_off(2);
                    uzhat(iY,iX,iZ)=u_off(3);
                end
            end
        end
    end
    % IFFT back and interpolate to get the far field
    ux=real(ifftn(uxhat));
    uy=real(ifftn(uyhat));
    uz=real(ifftn(uzhat));
    uxpts=interpolate(hx*hy*hz*S,ux,nx,ny,nz);
    uypts=interpolate(hx*hy*hz*S,uy,nx,ny,nz);
    uzpts=interpolate(hx*hy*hz*S,uz,nx,ny,nz);
    velfar=[uxpts'; uypts'; uzpts'];
end