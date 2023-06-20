function M = FilterM(M,thres)
    [V,D]=eig(M);
    D = real(diag(D));
    D(D < thres) = thres;
    %DHalf = sqrt(D);
    %DMinusHalf = 1./DHalf; DMinusHalf(D < 1e-10) = 0;
    %Dinv = 1./D; Dinv(D < 1e-10) = 0;
    Vinv = V^(-1);
    M = V*diag(D)*Vinv;
    %Minv = V*diag(Dinv)*Vinv;
    %MHalf = V*diag(DHalf)*Vinv;
    %MMinusHalf = V*diag(DMinusHalf)*Vinv;
end
