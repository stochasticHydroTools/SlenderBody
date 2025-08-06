function hInterp = InterpolatehNUFFT(InterpPts,Mem)
    dx=Mem.x(2)-Mem.x(1);
    gw=Mem.dx;
    N = length(Mem.x);
    L = N*dx;
    hmem=Mem.hmem;
    if (size(hmem,1)~=size(hmem,2))
        hmem=reshape(Mem.hmem,Mem.M,Mem.M);
    end
    hhat = fft2(hmem);
    ksqMult=Mem.ksq;
    %ksqMult(abs(ksqMult)>(3*N/8*2*pi/L)^2)=0;
    hhatConv = hhat.*exp(gw^2*ksqMult/2);
    hConv = ifft2(hhatConv);
    hInterp = InterpFromGrid(Mem.x,Mem.x,hConv,InterpPts,gw);
end
