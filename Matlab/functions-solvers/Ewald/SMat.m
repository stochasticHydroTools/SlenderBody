% Computes the spread and interpolation matrices by fast Gaussian gridding.
% Vectorized and linear in the number of points. 
% Domain has to be periodic on [-L/2, L/2]
function SWeights = SMat(xEpts,yEpts,zEpts,IBpts,sup,gw)
    Nx=length(xEpts);
    Ny=length(yEpts);
    Nz=length(zEpts);
    [NIB,~]=size(IBpts);
    hex=xEpts(2)-xEpts(1);
    hey=yEpts(2)-yEpts(1);
    hez=zEpts(2)-zEpts(1);
    aex=min(xEpts);
    aey=min(yEpts);
    aez=min(zEpts);
    Lx=max(xEpts)-min(xEpts)+hex;
    Ly=max(yEpts)-min(yEpts)+hey;
    Lz=max(zEpts)-min(zEpts)+hez;
    down=sup/2-1;
    up=sup/2;
    if (mod(sup,2)==1)
        down=floor(sup/2);
        up=down;
    end
    mvals=-down:up;
    SWeights=sparse(Nx*Ny*Nz,NIB);
    for ilam=1:NIB
        % Fast Gaussian gridding
        % Compute the z weights
        floorz=mod(floor((IBpts(ilam,3)-aez)/hez+1e-10),Nz)+1;
        zclose=zEpts(floorz)+roundm(IBpts(ilam,3)/Lz)*Lz;
        E1z = exp(-(IBpts(ilam,3)-zclose)^2/(2*gw^2));
        E2z = exp((IBpts(ilam,3)-zclose)*Lz/(Nz*gw^2));
        zwts = E1z.*E2z.^mvals.*exp(-(mvals.*Lz/Nz).^2/(2*gw^2));
        zpts=floorz-down:floorz+up;
        zpts=mod(zpts,Nz);
        zpts(zpts==0)=Nz;
        % Compute the y weights
        floory=mod(floor((IBpts(ilam,2)-aey)/hey+1e-10),Ny)+1;
        yclose=yEpts(floory)+roundm(IBpts(ilam,2)/Ly)*Ly;
        E1y = exp(-(IBpts(ilam,2)-yclose)^2/(2*gw^2));
        E2y = exp((IBpts(ilam,2)-yclose)*Ly/(Ny*gw^2));
        ywts = E1y.*E2y.^mvals.*exp(-(mvals.*Ly/Ny).^2/(2*gw^2));
        ypts=floory-down:floory+up;
        ypts=mod(ypts,Ny);
        ypts(ypts==0)=Ny;
        % Compute the x weights
        floorx=mod(floor((IBpts(ilam,1)-aex)/hex+1e-10),Nx)+1;   
        xclose=xEpts(floorx)+roundm(IBpts(ilam,1)/Lx)*Lx;
        E1x = exp(-(IBpts(ilam,1)-xclose)^2/(2*gw^2));
        E2x = exp((IBpts(ilam,1)-xclose)*Lx/(Nx*gw^2));
        xwts = E1x.*E2x.^mvals.*exp(-(mvals.*Lx/Nx).^2/(2*gw^2));
        xpts=floorx-down:floorx+up;
        xpts=mod(xpts,Nx);
        xpts(xpts==0)=Nx;
        xywts=1/(2*pi*gw^2)*xwts'*ywts;
        xywts=xywts(:);
        xyinds=xpts'+Nx*(ypts-1);
        xyinds=xyinds(:);
        zwts=1/sqrt(2*pi*gw^2)*zwts;
        totwts=xywts*zwts;
        totinds=xyinds+Nx*Ny*(zpts-1);
        SWeights=SWeights+sparse(totinds(:),ilam,totwts(:),...
            Nx*Ny*Nz,NIB);
    end
end

function val = roundm(x)
    val = round(x);
    if (mod(x,1)==0.5)
        if (x > 0)
            val = val-1;
        end
    end
end