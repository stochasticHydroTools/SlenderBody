% Computes the spread and interpolation matrices without Gaussian gridding.
% Vectorized and linear in the number of points. 
function SWeights = SMatLESlow(xEpts,yEpts,zEpts,IBpts,sup,gw,g)
   % Do the naive thing first
   [nP,~]=size(IBpts);
   hex=xEpts(2)-xEpts(1);
   hey=yEpts(2)-yEpts(1);
   hez=zEpts(2)-zEpts(1);
   Lx=max(xEpts)-min(xEpts)+hex;
   Ly=max(yEpts)-min(yEpts)+hey;
   Lz=max(zEpts)-min(zEpts)+hez;
   Nx = length(xEpts);
   Ny = length(yEpts);
   Nz = length(zEpts);
   SWeights=zeros(Nx*Ny*Nz,nP);
   for iPt=1:nP
       % The point will come in as (x,y,z)
       for iX=1:Nx
           for iY=1:Ny
               for iZ=1:Nz
                   % Find the closest point
                   gridpt = xEpts(iX)*[1 0 0]+yEpts(iY)*[g 1 0]+zEpts(iZ)*[0 0 1];
                   rvec=gridpt-IBpts(iPt,:);
                   rvec = calcShifted(rvec,g,Lx,Ly,Lz);
                   r=norm(rvec);
                   ind = iX+Nx*(iY-1)+Nx*Ny*(iZ-1);
                   SWeights(ind,iPt)=1/sqrt(8*pi^3*gw^6)*exp(-r^2/(2*gw^2));
               end
           end
       end
   end
end