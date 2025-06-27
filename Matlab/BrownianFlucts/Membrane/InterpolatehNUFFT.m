function hInterp = InterpolatehNUFFT(h,InterpPts,ksq,x)
    dx=x(2)-x(1);
    gw=dx;
    N = length(x);
    L = N*dx;
    hhat = fft2(h);
    ksqMult=ksq;
    ksqMult(abs(ksqMult)>(3*N/8*2*pi/L)^2)=0;
    hhatConv = hhat.*exp(gw^2*ksq/2);
    hConv = ifft2(hhatConv);
    hInterp = InterpFromGrid(x,x,hConv,InterpPts,gw);
end
