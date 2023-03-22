%% Initialization and precomputations for dynamic fiber simulations
%% Computations on the N+1 point grid
Nx = N+1;
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
if (RectangularCollocation)
    [sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],1);
    DNp1 = diffmat(Nx,[0 L],'chebkind1');
end
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
stDNp1 = stackMatrix(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*(N+1),3);
for iR=1:N+1
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
Xst = zeros(3*N*nFib,1);
Xt = zeros(3*Nx*nFib,1);
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
for iFib=1:nFib
    Xst((iFib-1)*3*N+1:iFib*3*N) = reshape(X_s((iFib-1)*N+1:iFib*N,:)',[],1);
    Xt((iFib-1)*3*Nx+1:iFib*3*Nx) = XonNp1Mat*[Xst((iFib-1)*3*N+1:iFib*3*N);XMP(:,iFib)];
end

% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, b2Np2] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
D2Np2 = diffmat(2*Nx,[0 L],'chebkind2');
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
if (exist('PenaltyForceInsteadOfFlow','var')&& PenaltyForceInsteadOfFlow)
    EMat_Np1 = BendingEnergyMatrix_Np1+gam0*WTilde_Np1;
else
    EMat_Np1 = BendingEnergyMatrix_Np1;
end
BendMatHalf_Np1 = real(EMat_Np1^(1/2));
BendForceMat = -EMat_Np1;
BendForceDenMat = WTilde_Np1_Inverse*BendForceMat;
%% Additional calculations for rectangular spectral colloc
if (RectangularCollocation)
% Rectangular spectral collocation (another option for bending force)
% Precompute the matrices associated with enforcing the weak BCs.
% See Section 4.3 in the paper for details on this
[sNp5, wNp5, bNp5] = chebpts(Nx+4, [0 L], 2);
DNp5 = diffmat(Nx+4,[0 L],'chebkind2');
RNp5ToNp1 = barymat(sNp1,sNp5,bNp5);
RNp5ToN = barymat(s,sNp5,bNp5);
RX = stackMatrix(RNp5ToNp1);

Npsi = N-1;
[sPsi, wPsi, bPsi] = chebpts(Npsi, [0 L], 1);
[sPsi2, wPsi2, bPsi2] = chebpts(2*Npsi, [0 L], 2);
RPsiToN = barymat(s,sPsi,bPsi);
DPsi = diffmat(Npsi,[0 L],'chebkind1');
[sPsip2, wPsip2, bPsip2] = chebpts(Npsi+2, [0 L], 2);
DPsip2 = diffmat(Npsi+2,[0 L],'chebkind2');
RTheta = barymat(sPsi,sPsip2,bPsip2);
RPsiToNp1 = barymat(sNp1,sPsip2,bPsip2);
RPsip2ToN = barymat(s,sPsip2,bPsip2);
RPsiTo2Psi = barymat(sPsi2,sPsi,bPsi);
WTilde_Psi = RPsiTo2Psi'*diag(wPsi2)*RPsiTo2Psi;

RNp1ToPsi = barymat(sPsi,sNp1,bNp1);
RNToPsi = barymat(sPsi,s,b);
RNp1ToPsip2 = barymat(sPsip2,sNp1,bNp1);
RPsiToNp5 = barymat(sNp5,sPsip2,bPsip2);

XForMrrp2 = RNp1ToPsip2*reshape(Xt(1:3*Nx),3,[])';
XForMrr = RNp1ToPsi*reshape(Xt(1:3*Nx),3,[])';

asympRR=1;
Mrr_Psip2 = RotRotMobilityMatrix(XForMrrp2,a,L,mu,sPsip2,bPsip2,DPsip2,[],[],1,deltaLocal);
Mrr = RotRotMobilityMatrix(XForMrr,a,L,mu,sPsi,bPsi,DPsi,[],[],1,deltaLocal);
BCanswers = zeros(12*nFib,1);
% s = 0 end: can be either free or clamped
XBCMat =  FreeBCMatrix(0,sNp5,bNp5,DNp5);
XBCMat_low = FreeBCMatrix(0,sNp1,bNp1,DNp1);
ThetaBCMat = FreeThetBCMatrix(0,sPsip2,bPsip2,DPsip2);
if (clamp0)
    % Modify the K matrix as well
    B0Np1 = stackMatrix(barymat(0,sNp1,bNp1));
    XonNp1Mat = [(eye(3*(N+1))-repmat(B0Np1,N+1,1))*stackMatrix(IntDNp1*RToNp1) I];
    InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); B0Np1];
    for iFib=1:nFib
        Xt((iFib-1)*3*Nx+1:iFib*3*Nx) = XonNp1Mat*[Xst((iFib-1)*3*N+1:iFib*3*N);X0BC(:,iFib)];
    end

    XBCMat_low = ClampedBCMatrix(0,sNp1,bNp1,DNp1);
    XBCMat =  ClampedBCMatrix(0,sNp5,bNp5,DNp5);
    ThetaBCMat = ClampedThetBCMatrix(0,sPsip2,bPsip2,DPsip2,Mrr_Psip2,twmod);
    if (TorqBC)
        ThetaBCMat = ClampedThetBCMatrixTorq(0,sPsip2,bPsip2,DPsip2,Mrr_Psip2,twmod);
    end
    for iFib=1:nFib
        BCanswers((iFib-1)*12+1:(iFib-1)*12+6) = [X0BC(:,iFib); Tau0BC(:,iFib)];
    end
end
% Assume s = L end is always free
XBCMat =  [XBCMat; FreeBCMatrix(L,sNp5,bNp5,DNp5)];
XBCMat_low = [XBCMat_low; FreeBCMatrix(L,sNp1,bNp1,DNp1)];
ThetaBCMat = [ThetaBCMat; FreeThetBCMatrix(L,sPsip2,bPsip2,DPsip2)];

% Compile rectangular spectral collocation matrices
XBCst = stackMatrix(XBCMat);
FE = -Eb*RX*stackMatrix(DNp5^4);
UpsampleXBCMat = [RX;XBCst] \ [eye(3*(N+1)); zeros(12,3*(N+1))];
BendForceDenMat=FE*UpsampleXBCMat;
BendForceMat = WTilde_Np1*BendForceDenMat; % Force not density
BCShift = zeros(3*(N+5)*nFib,1);
for iFib=1:nFib
    BCShift(3*(N+5)*(iFib-1)+1:3*(N+5)*iFib) = [RX;XBCst] \ ...
        [zeros(3*(N+1),1);BCanswers(12*(iFib-1)+1:12*iFib)];
end
UpsampleThetaBCMat = [RTheta;ThetaBCMat];
ThetaImpPart = dt*twmod*RTheta*DPsip2*Mrr_Psip2*DPsip2*UpsampleThetaBCMat^(-1);
ThetaImplicitMatrix = eye(Npsi)-(ThetaImpPart*[eye(Npsi);zeros(2,Npsi)]);
PsiBC0=0;
if (clamp0)
    PsiBC0= TurnFreq;
    if (TorqBC)
        PsiBC0 = -w*diag(Mrr^(-1)).*TurnFreq/twmod;
    end
end

%% Initialize twist angles, material frame and Bishop frame
% Initialize the material frame and Bishop frame. We will keep track of
% D1(L/2) = D1mid by explicitly rotating it, then updating the Bishop frame
% to be consistent with D1(L/2). 
% Fiber and twist angle initialization
if (twmod==0 || initZeroTheta)
    theta0 = zeros(nFib*N,1);
    theta_s = zeros(nFib*Npsi,1);
else
    theta_s = TurnFreq/twmod*pinv(DPsi)*(Mrr \ ones(Npsi,1));
    theta_s = theta_s-barymat(L,sPsi,bPsi)*theta_s;
    theta0 = pinv(D)*RPsiToN*theta_s;
    theta0 = theta0-barymat(L/2,s,b)*theta0;
    theta0 = repmat(theta0,nFib,1);
end
D1mid = zeros(nFib,3);
D1 = zeros(nFib*N,3);
D2 = zeros(nFib*N,3);
bishA = zeros(nFib*N,3);
bishB = zeros(nFib*N,3);
for iFib=1:nFib
    Xsmid = barymat(L/2,s,b)*reshape(Xst((iFib-1)*3*N+1:iFib*3*N),3,N)';
    thisD1mid = [Xsmid(2) -Xsmid(1) 0];
    if (norm(thisD1mid) < 1e-10)
        thisD1mid = [1 0 0];
    end
    D1mid(iFib,:)=thisD1mid/norm(thisD1mid);
    rinds = (iFib-1)*N+1:iFib*N;
    [tbishA,tbishB,tD1,tD2] = computeBishopFrame(N,X_s(rinds,:),...
        D*X_s(rinds,:),pinv(D),barymat(L/2,s,b),theta0(rinds),D1mid(iFib,:)');
    bishA(rinds,:)=tbishA;
    bishB(rinds,:)=tbishB;
    D1(rinds,:)=tD1;
    D2(rinds,:)=tD2;
end
end

%% Precomputations for hydrodynamics (N+1) point grid
% Precompute the integrals necessary for the exact RPY integrals
% These are integrals of T_k sign(s-s') ds' and the like. 
chebForInts=1; % Integrals with Chebyshev grid or analyticaly
AllbS_Np1 = precomputeStokesletInts(sNp1,L,a,N+1,chebForInts);
AllbS_TrueFP_Np1 = precomputeStokesletInts(sNp1,L,0,N+1,chebForInts);
AllbD_Np1 = precomputeDoubletInts(sNp1,L,a,N+1,chebForInts);
% # of pts for R < 2a integrals for exact RPY
warning('NForSmall set in init fiber vars!')
if (a/L*4/exp(3/2)>1e-3+1e-8)
    NForSmall = 8;
else
    NForSmall = 4;
end
gtype=2;
if (RectangularCollocation)
    gtype=1;
end
if (exist('NupsampleHydro','var'))
[sup,wup,~] = chebpts(NupsampleHydro, [0 L],gtype);
RupsampleHydro = barymat(sup,sNp1,bNp1); % From X grid to upsampled
WUp = stackMatrix(diag(wup));
end
if (upsamp==0) % Initialize eigenvalue cutoff
    NForRef = 1/eps;
    [sForRef,wForRef,~] = chebpts(NForRef, [0 L],gtype);
    XStraight = [sForRef zeros(NForRef,2)];
    MRPY=getGrandMBlobs(NForRef,XStraight,a,mu);
    WForRef = stackMatrix(diag(wForRef));
    RForRef = barymat(sForRef,sNp1,bNp1); % From X grid to upsampled
    MWsym = WTilde_Np1_Inverse*stackMatrix(RForRef)'...
        *WForRef*MRPY*WForRef*stackMatrix(RForRef)*WTilde_Np1_Inverse;
    eigThres = min(eig(MWsym));
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

function BCMat = ClampedThetBCMatrixTorq(sBC,s,b,D,Mrr,twmod)
    BCMat = barymat(sBC,s,b);
end

function BCMat = ClampedBCMatrix(sBC,s,b,D)
    BCMat = [barymat(sBC,s,b); barymat(sBC,s,b)*D];
end