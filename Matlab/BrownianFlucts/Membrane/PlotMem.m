function PlotMem(Mem,xCt,yCt)
    for iX=xCt
    for iY=yCt
    XPl = [Mem.xgu Mem.xgu(:,1)+Mem.Lm];
    XPl = [XPl; XPl(1,:)];
    YPl = [Mem.ygu; Mem.ygu(1,:)+Mem.Lm];
    YPl = [YPl YPl(:,1)];
    hmempl = reshape(Mem.UpsamplingMatrix*Mem.h,Mem.Nu,Mem.Nu);
    hmempl = [hmempl hmempl(:,1)];
    hmempl = [hmempl; hmempl(1,:)];
    surf(XPl+Mem.Lm*iX,YPl+Mem.Lm*iY,hmempl,...
        'FaceColor','interp','EdgeColor','interp','FaceAlpha',0.5)
    end
    end
end