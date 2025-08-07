function En = computeMembraneEnergy(Mem)
    hhat = fft2(reshape(Mem.hmem,Mem.M,Mem.M));
    ksqhhat = conj(hhat).*Mem.FourierEnergyMat.*hhat;
    % Integrate and square
    En = sum(ksqhhat(:));
end