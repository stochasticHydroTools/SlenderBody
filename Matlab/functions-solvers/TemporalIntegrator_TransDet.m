Xt = XonNp1Mat*[Xst;XMP];
XtPrev = XonNp1Mat*[XstPrev;XMPPrev];
Xs3 = reshape(Xst,3,N)';
Xs3Prev = reshape(XstPrev,3,N)';
OneXMob = Xt;
XsK = Xs3;
if (impcoeff ==1/2)
    OneXMob = 3/2*Xt-1/2*XtPrev;
    XsK = 3/2*Xs3-1/2*Xs3Prev;
end
% Mobility evaluation
if (exactRPY)
    % Trans-trans on the N+1 point grid
    M = computeRPYMobility(N,OneXMob,DNp1,a,L,mu,sNp1,bNp1,AllbS_Np1,AllbD_Np1,NForSmall,0);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsym = FilterM(1/2*(MWsym+MWsym'),eigThres);
    if (upsamp==1)
        Xup =RupsampleHydro*reshape(Xt,3,[])';
        MRPY=getGrandMBlobs(NupsampleHydro,Xup,a,mu);
        MWsym = WTilde_Np1_Inverse*stackMatrix(RupsampleHydro)'...
            *WUp*MRPY*WUp*stackMatrix(RupsampleHydro)*WTilde_Np1_Inverse;
    elseif (upsamp==-1)
        MWsym = getGrandMBlobs(Nx,reshape(Xt,3,Nx)',a,mu);
    end
else % local drag + finite part
    Xs_Np1 = stDNp1*OneXMob;
    Xss_Np1 = stDNp1*Xs_Np1;
    M = getGrandMloc(N+1,Xs_Np1,Xss_Np1,a,L,mu,sNp1,deltaLocal);
    if (includeFPonLHS)
        M = M+StokesletFinitePartMatrix(reshape(OneXMob,3,N+1)',reshape(Xs_Np1,3,N+1)',...
            reshape(Xss_Np1,3,N+1)',DNp1,sNp1,L,N+1,mu,AllbS_TrueFP_Np1);
    else
        MFP = StokesletFinitePartMatrix(reshape(OneXMob,3,N+1)',reshape(Xs_Np1,3,N+1)',...
            reshape(Xss_Np1,3,N+1)',DNp1,sNp1,L,N+1,mu,AllbS_TrueFP_Np1);
        U0 = U0 + MFP*WTilde_Np1_Inverse*ForceNL((iFib-1)*3*Nx+1:iFib*3*Nx);
    end
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsym = FilterM(1/2*(MWsym+MWsym'),eigThres);
end
% Saddle pt solve
K = KonNp1(XsK,XonNp1Mat,I);
B = K-impcoeff*dt*MWsym*BendForceMat*K;
Force = ForceExt+BendForceMat*Xt+MWsym \ U0;
RHS = K'*Force;
Schur = K'*(MWsym \ B);
alphaU = pinv(Schur)*RHS;
Lambda = MWsym \ (B*alphaU)-Force;
Omega = reshape(alphaU(1:3*N),3,N)';
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
XMP_p1 = XMP+dt*alphaU(end-2:end);
Xp1 = XonNp1Mat*[Xsp1;XMP_p1];
Xp1Star = Xt+dt*K*alphaU;