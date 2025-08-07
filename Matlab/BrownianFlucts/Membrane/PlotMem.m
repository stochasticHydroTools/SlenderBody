function PlotMem(Mem,xCt,yCt)
    for iX=xCt
    for iY=yCt
    XPl = [Mem.xg Mem.xg(:,1)+Mem.Lm];
    XPl = [XPl; XPl(1,:)];
    YPl = [Mem.yg; Mem.yg(1,:)+Mem.Lm];
    YPl = [YPl YPl(:,1)];
    hmempl = reshape(Mem.hmem,Mem.M,Mem.M);
    hmempl = [hmempl hmempl(:,1)];
    hmempl = [hmempl; hmempl(1,:)];
    surf(XPl+Mem.Lm*iX,YPl+Mem.Lm*iY,hmempl,'FaceColor','interp','FaceAlpha',0.5)
    end
    end
end