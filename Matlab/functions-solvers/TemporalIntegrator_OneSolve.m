BendMat = -EMat_Np1;
Xt = XonNp1Mat*[Xst;XMP];
Xs3 = reshape(Xst,3,N)';
EndEndDists(count+1) = norm(Xt(1:3)-Xt(end-2:end));
if (IdForM && count==0)
    M = eye(3*(N+1))/(8*pi*mu);
    MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    MWsymHalf = real(MWsym^(1/2));
elseif (~IdForM)
    if (upsamp==1)
        Xup = reshape(ExtRef*Xt,3,[])';
        MRPY=getGrandMBlobs(Nref,Xup,a,mu);
        warning('Update oversampled code to reflect precomputations')
        MWsym = WTilde_Np1_Inverse*ExtRef'*WRef*MRPY*WRef*ExtRef*WTilde_Np1_Inverse;
    elseif (upsamp==-1)
        MWsym = getGrandMBlobs(Nx,reshape(Xt,3,Nx)',a,mu);
    elseif (upsamp==-2)
        M = getMlocStokeslet(Nx,DNp1*reshape(Xt,3,Nx)',a,L,mu,sNp1,1);
        MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
    else
        M = computeRPYMobility(N,Xt,DNp1,a,L,mu,sNp1,...
            bNp1,AllbS_Np1,AllbD_Np1,NForSmall,0);
        MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
        MWsym = FilterM(MWsym,eigThres);
    end
    MWsymHalf = real(MWsym^(1/2));
end
if (min(real(eig(MWsym))) < 0)
   min(real(eig(MWsym)))
   warning('Negative eigs in MW - going to goof everything up')
end
% Obtain Brownian velocity
g = randn(3*(N+1),1);
%g = load(strcat('RandVec1_',num2str(count+1),'.txt'));
%g = g(3*(N+1)*(iFib-1)+1:3*(N+1)*iFib);
RandomVel = sqrt(2*kbT/dt)*MWsymHalf*g;
RandomVelSpl = RandomVel;
OmegaTilde = cross(reshape(Xst,3,N)',RNp1ToN*DNp1*reshape(RandomVelSpl,3,[])');
XMPTilde = XMP+dt/2*BMNp1*RandomVelSpl;
Xstilde = rotateTau(Xs3,OmegaTilde,dt/2);
Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
% Saddle point solve with Ktilde
if (~IdForM)
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XMPTilde];
    if (upsamp==1)
        Xtup = reshape(ExtRef*Xtilde,3,[])';
        MRPYTilde=getGrandMBlobs(Nref,Xtup,a,mu);
        MWsymTilde =  WTilde_Np1_Inverse*ExtRef'*WRef*MRPYTilde*WRef*ExtRef*WTilde_Np1_Inverse;
    elseif (upsamp==-1)
        MWsymTilde = getGrandMBlobs(Nx,reshape(Xtilde,3,Nx)',a,mu);
    elseif (upsamp==-2)
        MTilde = getMlocStokeslet(Nx,DNp1*reshape(Xtilde,3,Nx)',a,L,mu,sNp1,1);
        MWsymTilde = 1/2*(MTilde*WTilde_Np1_Inverse + WTilde_Np1_Inverse*MTilde');
    else
        MTilde = computeRPYMobility(N,Xtilde,DNp1,a,L,mu,sNp1,bNp1,...
            AllbS_Np1,AllbD_Np1,NForSmall,upsamp);
        MWsymTilde = 1/2*(MTilde*WTilde_Np1_Inverse + WTilde_Np1_Inverse*MTilde');
        MWsymTilde = FilterM(MWsymTilde,eigThres);
    end
else
    MWsymTilde = MWsym;
end
if (min(real(eig(MWsym))) < 0)
   min(real(eig(MWsym)))
   warning('Negative eigs in MW - going to goof everything up')
end
if (impcoeff==1 && ModifyBE)
    %g2 = load(strcat('RandVec2_',num2str(count+1),'.txt'));
    %g2 = g2(3*(N+1)*(iFib-1)+1:3*(N+1)*iFib);
    g2 = randn(3*(N+1),1);
    RandomVel = RandomVel+sqrt(2*kbT/dt)*sqrt(dt/2)*...
        MWsymTilde*BendMatHalf_Np1*g2;
end
if (PenaltyForceInsteadOfFlow)
    U0 = -MWsymTilde*BendMat*X0; % the explicit part
end
if (~IdForM)
    % Add the RFD term on M
%     deltaRFD = 1e-5;
%     WRFD = randn(3*N+3,1); % This is Delta X on the N+1 grid
%     KInv = KInvonNp1(Xs3,InvXonNp1Mat,BMNp1);
%     TauRot = reshape(KInv*WRFD,3,N+1)'; 
%     TauPlus = rotateTau(Xs3,deltaRFD*TauRot(1:N,:),1);
%     XMPPlus = XMP+deltaRFD*TauRot(end,:)';
%     XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XMPPlus];
%     MPlus = computeRPYMobility(N,XPlus,DNp1,a,L,mu,sNp1,bNp1,...
%         AllbS_Np1,AllbD_Np1,NForSmall,0);
%     MWsymPlus = 1/2*(MPlus*WTilde_Np1_Inverse + WTilde_Np1_Inverse*MPlus');
%     M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*WRFD;
    M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelSpl);
    U0 = U0 + M_RFD;
end
%fext = load(strcat('ExForceDen_',num2str(count+1),'.txt'));
%fext = fext(3*(N+1)*(iFib-1)+1:3*(N+1)*iFib);
%fext = zeros(3*(N+1),1);
K = Ktilde;
B = K-impcoeff*dt*MWsymTilde*BendMat*K;
RHS = K'*(BendMat*Xt+WTilde_Np1*fext+MWsymTilde \ (RandomVel + U0));
alphaU = lsqminnorm(K'*(MWsymTilde \ B),RHS);
Lambda = MWsymTilde \ (B*alphaU -RandomVel-U0) - BendMat*Xt - WTilde_Np1*fext;
Omega = reshape(alphaU(1:3*N),3,N)';
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
XMP_p1 = XMP+dt*alphaU(end-2:end);
Xp1 = XonNp1Mat*[Xsp1;XMP_p1];