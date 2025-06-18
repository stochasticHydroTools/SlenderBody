function PInvSP = ManualPinv(Mat,maxRank)
    [U,S,V]=svd(Mat);
    S=diag(S);
    S(maxRank+1:end)=0;
    S(S/S(1)<1e-10)=0;
    pinvS = 1./S;
    pinvS(S==0)=0;
    PInvSP = V*diag(pinvS)*U';
end