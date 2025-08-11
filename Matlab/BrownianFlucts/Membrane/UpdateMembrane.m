function Mem = UpdateMembrane(Mem,Fext,kbT,dt)
    Fh = -Mem.Kh*Mem.h;
    RHSmem = Mem.Mmem*(Fext+Fh)...
        +sqrt(2*kbT/dt)*Mem.Mhalfmem*randn(Mem.M^2,1);
    % hnew = InvImpMatMem*(hmem/dt+RHSmem);
    % FFT way
    RHSHat = fft2(reshape(Mem.h/dt+RHSmem,Mem.M,Mem.M));
    hNewHat = RHSHat./Mem.ImpfacFourier;
    hnew = ifft2(hNewHat);
    Mem.h = hnew(:);
end