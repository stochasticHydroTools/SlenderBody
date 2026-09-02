function MWsymq = RPYQuadMob(Xt,rtrue,L,mu,sNp1,bNp1,DNp1,AllbS_Np1,AllbD_Np1,...
    NForSmall,WTilde_Np1_Inverse,eigThres)
    M = TransTransMobilityMatrix(reshape(Xt,3,[])',...
    rtrue,L,mu,sNp1,bNp1,DNp1,AllbS_Np1,AllbD_Np1,NForSmall,0,0,0);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsymq = FilterM(1/2*(MWsym+MWsym'),eigThres);
end
