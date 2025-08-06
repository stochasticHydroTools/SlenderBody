function Fgmem = SpreadStericForceToMem(xyForce,stForce,Mem)
    M = Mem.M;
    Fgmem = zeros(M);
    for iP=1:length(stForce) % Just put it on the nearest grid pt
        xpt = 1 + round(xyForce(iP,1)/Mem.dx);
        ypt = 1 + round(xyForce(iP,2)/Mem.dx);
        xpt(xpt>M)=mod(xpt(xpt>M),M);
        xpt(xpt<0)=mod(xpt(xpt<0),M);
        xpt(xpt==0)=xpt(xpt==0)+M;
        ypt(ypt>M)=mod(ypt(ypt>M),M);
        ypt(ypt<0)=mod(ypt(ypt<0),M);
        ypt(ypt==0)=ypt(ypt==0)+M;
        Fgmem(ypt,xpt) = Fgmem(ypt,xpt) + stForce(iP);
    end
    Fgmem=Fgmem(:);
end