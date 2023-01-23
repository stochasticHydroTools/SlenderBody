% Temporal integrator for fibers that resist bend and twist
BendMat = -EMat_Np1;
Xs3 = reshape(Xst,3,N)';
Xt = XonNp1Mat*[Xst;XMP];
if (IdForM && count==0)
    M = eye(3*(N+1))/(8*pi*mu);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsymHalf = real(MWsym^(1/2));
elseif (~IdForM)
    M = computeRPYMobility(N,Xt,DNp1,a,L,mu,sNp1,bNp1,AllbS_Np1,AllbD_Np1,NForSmall,0);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsym = FilterM(MWsym,eigThres);
    MWsymHalf = real(MWsym^(1/2));
end
if (PenaltyForceInsteadOfFlow)
    U0 = -MWsym*BendMat*X0; % the explicit part
end
K = KonNp1(Xs3,XonNp1Mat,I);
B = K-impcoeff*dt*MWsym*BendMat*K;
g = randn(3*(N+1),1);
RandomVel = sqrt(2*kbT/dt)*MWsymHalf*g;
if (impcoeff==1 && ModifyBE)
    RandomVel = RandomVel+sqrt(2*kbT/dt)*sqrt(dt/2)*...
        MWsym*BendMatHalf_Np1*randn(3*(N+1),1);
end
RHS = K'*(BendMat*Xt+MWsym \ (RandomVel + U0));
alphaU = lsqminnorm(K'*MWsym^(-1)*B,RHS);
% ut = K*alphaU;
% Omega = cross(reshape(Xst,3,N)',RNp1ToN*DNp1*reshape(ut,3,[])');
% max(abs(Omega-reshape(alphaU(1:3*N),3,N)'))

% Add the RFD part
deltaRFD = 1e-5;
N_RFD = zeros(3*N+3,1);
nSampRFD = 1;
N_og = pinv(K'*MWsym^(-1)*K);
for iSampRFD=1:nSampRFD
    WRFD = randn(3*N+3,1); % This is Delta X on the N+1 grid
    TauPlus = rotateTau(Xs3,deltaRFD*reshape(WRFD(1:3*N),3,N)',1);
    XMPPlus = XMP+deltaRFD*WRFD(3*N+1:end);
    XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XMPPlus];
    if (IdForM)
        MWsymPlus=MWsym;
    else
        MPlus = computeRPYMobility(N,XPlus,DNp1,a,L,mu,sNp1,bNp1,...
            AllbS_Np1,AllbD_Np1,NForSmall,0);
        MWsymPlus = 1/2*(MPlus*WTilde_Np1_Inverse + WTilde_Np1_Inverse*MPlus');
        MWsymPlus = FilterM(MWsymPlus,eigThres);
    end
    KPlus = KonNp1(TauPlus,XonNp1Mat,I);
    N_Plus = pinv(KPlus'*MWsymPlus^(-1)*KPlus);
    N_RFD = N_RFD+1/(deltaRFD*nSampRFD)*(N_Plus-N_og)*WRFD;
end
% Add the RFD term to the saddle point solve
alphaU = alphaU + kbT*N_RFD;
Omega = reshape(alphaU(1:3*N),3,N)';
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
XMP_p1 = XMP+dt*alphaU(end-2:end);
Xp1 = XonNp1Mat*[Xsp1;XMP_p1];