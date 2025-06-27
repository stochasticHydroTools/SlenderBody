% Computes the spread and interpolation matrices by fast Gaussian gridding.
% Vectorized and linear in the number of points. 
function InterpVals = InterpFromGrid(xEpts,yEpts,ValsOnGrid,IBpts,gw)
    [nPts,~]=size(IBpts);
    InterpVals=zeros(nPts,1);
    dx=xEpts(2)-xEpts(1);
    dy=yEpts(2)-yEpts(1);
    sup=2*floor(4*gw/dx);
    Nx=length(xEpts);
    Ny=length(yEpts);
    hex=xEpts(2)-xEpts(1);
    hey=yEpts(2)-yEpts(1);
    aex=min(xEpts);
    aey=min(yEpts);
    Lx=max(xEpts)-min(xEpts)+hex;
    Ly=max(yEpts)-min(yEpts)+hey;
    IBpts = IBpts-floor(IBpts./[Lx Ly]).*[Lx Ly]; % Puts them on [0,L]
    down=sup/2-1;
    up=sup/2;
    oddsup = false;
    if (mod(sup,2) == 1)
        down = floor(sup/2);
        up = down;
        oddsup = true;
    end
    mvals=-down:up;
    for ilam=1:nPts
        % Fast Gaussian gridding in x and y
        floory=mod(floor((IBpts(ilam,2)-aey)/hey+1e-10),Ny)+1;
        % correct mvals for when particle is closer to right face of cell i
        if oddsup && abs(yEpts(floory) - IBpts(ilam,2)) > hey/2
            floory = floory + 1;
        end  
        yclose=yEpts(floory);
        % Compute the y weights
        E1y = exp(-(IBpts(ilam,2)-yclose)^2/(2*gw^2));
        E2y = exp((IBpts(ilam,2)-yclose)*Ly/(Ny*gw^2));
        ywts = E1y.*E2y.^mvals.*exp(-(mvals.*Ly/Ny).^2/(2*gw^2));
        ypts=floory-down:floory+up;
        ypts=mod(ypts,Ny);
        ypts(ypts==0)=Ny;
        % Compute the x weights
        floorx=mod(floor((IBpts(ilam,1)-aex)/hex+1e-10),Nx)+1;
        % correct mvals for when particle is closer to right face of cell i
        if oddsup && abs(xEpts(floorx) - IBpts(ilam,1)) > hex/2
            floorx = floorx + 1;
        end      
        xclose=xEpts(floorx);
        E1x = exp(-(IBpts(ilam,1)-xclose)^2/(2*gw^2));
        E2x = exp((IBpts(ilam,1)-xclose)*Lx/(Nx*gw^2));
        xwts = E1x.*E2x.^mvals.*exp(-(mvals.*Lx/Nx).^2/(2*gw^2));
        xpts=floorx-down:floorx+up;
        xpts=mod(xpts,Nx);
        xpts(xpts==0)=Nx;
        xywts=1/(2*pi*gw^2)*(ywts'*xwts).*ValsOnGrid(ypts,xpts)*dx*dy;
        InterpVals(ilam)=sum(xywts(:));
    end
end

