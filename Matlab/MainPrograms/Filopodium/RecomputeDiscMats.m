% Precompute the relevant forces using the correct BCs
Lfac = Lfacs(iFib);
WTilde_Np1_L = Lfac*WTilde_Np1;
WTilde_Psi_L = Lfac*WTilde_Psi;
DNp5_L = 1/Lfac*DNp5;
DNp1_L = 1/Lfac*DNp1;
DPsip2_L = 1/Lfac*DPsip2;
DPsi_L = 1/Lfac*DPsi;
D_L = 1/Lfac*D;
IntDNp1_L = Lfac*IntDNp1;
% s = 0 end: can be either free or clamped
XBCMat_L =  FreeBCMatrix(0,sNp5,bNp5,DNp5_L);
if (clamp0)
    XBCMat_L =  ClampedBCMatrix(0,sNp5,bNp5,DNp5_L);
end
% Assume s = L end is always free
XBCMat_L =  [XBCMat_L; FreeBCMatrix(L,sNp5,bNp5,DNp5_L)];
% Compile rectangular spectral collocation matrices
XBCst_L = stackMatrix(XBCMat_L);
BCShift_L =  [RX;XBCst_L] \ [zeros(3*(N+1),1);BCanswers(12*(iFib-1)+1:12*iFib)];
FE_L = -Eb*RX*stackMatrix(DNp5_L^4);
BCForce_L = FE_L*BCShift_L;
UpsampleXBCMat_L = [RX;XBCst_L] \ [eye(3*(N+1)); zeros(12,3*(N+1))];
BendForceDensityMat_L = FE_L*UpsampleXBCMat_L;
DPsi_L = 1/Lfac*DPsi;
DPsip2_L = 1/Lfac*DPsip2;

ThetaBCMat_L = ClampedThetBCMatrix(0,sPsip2,bPsip2,DPsip2_L,Mrr_Psip2,twmod);
% s=L end: fiber is either clamped with rotational velocity, or torque BC
if (TwistTorq)
    ThetaBCMat_L = [ThetaBCMat_L; ...
        ClampedThetBCMatrixTorq(L,sPsip2,bPsip2,DPsip2_L,Mrr_Psip2,twmod)];
else
    ThetaBCMat_L = [ThetaBCMat_L; ...
        ClampedThetBCMatrix(L,sPsip2,bPsip2,DPsip2_L,Mrr_Psip2,twmod)];
end
UpsampleThetaBCMat_L = [RTheta;ThetaBCMat_L];
ThetaImpPart_L = dt*twmod*RTheta*DPsip2_L*Mrr_Psip2*DPsip2_L*UpsampleThetaBCMat_L^(-1);
ThetaImplicitMatrix_L = eye(Npsi)-(ThetaImpPart_L*[eye(Npsi);zeros(2,Npsi)]);
PsiBC0 = 0;
if (TwistTorq)
    if (Lprime(iFib) > 0)
        PsiBCL= NFromFormin/twmod;
    else
        PsiBCL = 0;
    end
else
    PsiBCL = 0;
end

function BCMat = FreeBCMatrix(sBC,s,b,D)
    BCMat = [barymat(sBC,s,b)*D^2; barymat(sBC,s,b)*D^3];
end

function BCMat = FreeThetBCMatrix(sBC,s,b,D)
    BCMat = barymat(sBC,s,b);
end

function BCMat = ClampedThetBCMatrix(sBC,s,b,D,Mrr,twmod)
    BCMat = barymat(sBC,s,b)*Mrr*twmod*D;
end

function BCMat = ClampedBCMatrix(sBC,s,b,D)
    BCMat = [barymat(sBC,s,b); barymat(sBC,s,b)*D];
end

function BCMat = ClampedThetBCMatrixTorq(sBC,s,b,D,Mrr,twmod)
    BCMat = barymat(sBC,s,b);
end