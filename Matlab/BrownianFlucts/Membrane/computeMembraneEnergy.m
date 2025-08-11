function En = computeMembraneEnergy(Mem)
    hhat = fft2(reshape(Mem.h,Mem.M,Mem.M));
    ksqhhat = conj(hhat).*Mem.FourierEnergyMat.*hhat;
    % Integrate and square
    En = 1/2*sum(ksqhhat(:));
end