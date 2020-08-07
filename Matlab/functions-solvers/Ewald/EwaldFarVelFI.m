% Far field velocity with PSE RPY tensor using FINUFFT
function velfar = EwaldFarVelFI(ptsxyz,forces,mu,Lx,Ly,Lz,xi,a,g)
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
    fxhat = finufft3d1(pts(:,1),pts(:,2),pts(:,3),forces(:,1),-1,1e-10,nx,ny,nz,opts);
    fyhat = finufft3d1(pts(:,1),pts(:,2),pts(:,3),forces(:,2),-1,1e-10,nx,ny,nz,opts);
    fzhat = finufft3d1(pts(:,1),pts(:,2),pts(:,3),forces(:,3),-1,1e-10,nx,ny,nz,opts);
    k = sqrt(kx.^2+(ky-g*kx).^2+kz.^2);
    % Factor is Eq. (9) in the paper
    % Careful: matlab has an extra pi in the sinc function
    factor = 1/mu*1./k.^2.*(sinc(k*a/pi).^2).*(1+(k.^2)/(4*xi^2)).*exp(-k.^2/(4*xi^2));
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
    uxpts=real(finufft3d2(pts(:,1),pts(:,2),pts(:,3),1,1e-10,uxhat,opts))/(Lx*Ly*Lz);
    uypts=real(finufft3d2(pts(:,1),pts(:,2),pts(:,3),1,1e-10,uyhat,opts))/(Lx*Ly*Lz);
    uzpts=real(finufft3d2(pts(:,1),pts(:,2),pts(:,3),1,1e-10,uzhat,opts))/(Lx*Ly*Lz);
    velfar=[uxpts'; uypts'; uzpts'];
end
