Xt = XonNp1Mat*[Xst;XMP];
Xs3 = reshape(Xst,3,N)';
OneXMob = reshape(Xt,3,[])';
XsK = Xs3;
if (impcoeff ==1/2)
    XtPrev = XonNp1Mat*[XstPrev;XMPPrev];
    OneXMob = reshape(3/2*Xt-1/2*XtPrev,3,[])';
    Xs3Prev = reshape(XstPrev,3,N)';
    XsK = 3/2*Xs3-1/2*Xs3Prev;
end
% Mobility evaluation
if (upsamp==1)
    Xup =RupsampleHydro*reshape(Xt,3,[])';
    MRPY=getGrandMBlobs(NupsampleHydro,Xup,a,mu);
    MWsym = WTilde_Np1_Inverse*stackMatrix(RupsampleHydro)'...
        *WUp*MRPY*WUp*stackMatrix(RupsampleHydro)*WTilde_Np1_Inverse;
elseif (upsamp==-1)
    MWsym = getGrandMBlobs(Nx,reshape(Xt,3,Nx)',a,mu);
else 
    Binput = AllbS_Np1;
    if (~exactRPY)
        Binput = AllbS_TrueFP_Np1;
    end
    M = TransTransMobilityMatrix(OneXMob,a,L,mu,sNp1,bNp1,DNp1,Binput,...
        AllbD_Np1,NForSmall,~exactRPY,deltaLocal,TransTransLDOnly);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsym = FilterM(1/2*(MWsym+MWsym'),eigThres);
end
% Saddle pt solve
if (rigid)
K = KRigidonNp1(XsK,XonNp1Mat,I);
else
K = KonNp1(XsK,XonNp1Mat,I);
end
B = K-impcoeff*dt*MWsym*BendForceMat*K;
Force = ForceExt+BendForceMat*Xt+MWsym \ U0;
RHS = K'*Force;
Schur = K'*(MWsym \ B);
alphaU = pinv(Schur)*RHS;
Lambda = MWsym \ (B*alphaU)-Force;
if (rigid)
Omega = repmat(alphaU(1:3)',N,1);
else
Omega = reshape(alphaU(1:3*N),3,N)';
end
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
XMP_p1 = XMP+dt*alphaU(end-2:end);
Xp1 = XonNp1Mat*[Xsp1;XMP_p1];
Xp1Star = Xt+dt*K*alphaU;