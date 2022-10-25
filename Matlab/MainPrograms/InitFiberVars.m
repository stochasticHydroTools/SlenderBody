% Initialization and precomputations for dynamic fiber simulations
% Derivative matrices and various upsampled grids
I=zeros(3*N,3);
wIt=zeros(3,3*N);
for iR=1:N
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
    wIt(1:3,3*(iR-1)+1:3*iR)=w(iR)*eye(3);
end
Lmat = cos(acos(2*s/L-1).*(0:N-1));
Ds = stackMatrix(D);
IntMat = pinv(D);
IntMat_st = stackMatrix(pinv(D));
[sp4, wp4, bp4] = chebpts(N+4, [0 L], 2);
D_sp4 = diffmat(N+4,1,[0 L]);
[sp2, ~, bp2] = chebpts(N+2, [0 L], 2);
D_sp2 = diffmat(N+2,[0 L]);
Oversamp = 2;
[sOversamp, wOversamp, bOversamp] = chebpts(Oversamp*N, [0 L], 1);
DOversamp = diffmat(Oversamp*N,[0 L],'chebkind1');
IntDOversamp = stackMatrix(pinv(DOversamp));
LmatOversamp = cos(acos(2*sOversamp/L-1).*(0:Oversamp*N-1));
R4ToN = barymat(s, sp4, bp4); % Resampling matrix from s+4-> s0
R2ToN = barymat(s, sp2, bp2);
R2To4 = barymat(sp4,sp2,bp2);
R4ToOversamp = barymat(sOversamp,sp4,bp4);
ROversampToN = barymat(s,sOversamp,bOversamp);
RNToOversamp = barymat(sOversamp,s,b);
RNToOversamp_st = stackMatrix(RNToOversamp);
WOversmap = diag(wOversamp);
BM = stackMatrix(barymat(L/2,s,b));
B0 = stackMatrix(barymat(0,s,b));
BendingEnergyMatrix = Eb*stackMatrix((RNToOversamp*D^2)'*WOversmap*RNToOversamp*D^2);
Wtilde = stackMatrix((RNToOversamp'*WOversmap*RNToOversamp));
WtildeHalf = stackMatrix((RNToOversamp'*WOversmap*RNToOversamp)^(1/2));
WtildeInverse = stackMatrix((RNToOversamp'*WOversmap*RNToOversamp)^(-1));
BendingForceMatrix = -WtildeInverse*BendingEnergyMatrix;
BendMat = -BendingEnergyMatrix; % FORCE NOT DENSITY!
if (PenaltyForceInsteadOfFlow)
    BendMat = -BendingEnergyMatrix+-gam0*Wtilde;
end
BendMatHalf = real((-BendMat)^(1/2));
% Energy matrix going through grid of size N+1
[sNp1,wNp1,bNp1]=chebpts(N+1,[0 L],2);
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
DNp1 = diffmat(N+1,[0 L],'chebkind2');
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*stackMatrix(IntDNp1*RToNp1) [I;eye(3)]];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
[s2Np2, w2Np2, b2Np2] = chebpts(2*N+2, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
D2Np2 = diffmat(2*N+2,[0 L],'chebkind2');
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
if (PenaltyForceInsteadOfFlow)
    EMat_Np1 = BendingEnergyMatrix_Np1+gam0*WTilde_Np1;
else
    EMat_Np1 = BendingEnergyMatrix_Np1;
end
BendMatHalf_Np1 = real(EMat_Np1^(1/2));

Rs = stackMatrix(R4ToN);
% Precompute the matrices associated with enforcing the weak BCs.
% See Section 4.3 in the paper for details on this
[~, ~, ~,Mrr_sp2,~] = getGrandMloc(N+2,zeros(3*(N+2),1),zeros(3*(N+2)),a,L,mu,sp2,deltaLocal);
[~, ~, ~,Mrr,~] = getGrandMloc(N,zeros(3*N,1),zeros(3*N),a,L,mu,s,deltaLocal);
BCanswers = zeros(12*nFib,1);
XBCMat_low = FreeBCMatrix(0,s,b,D);
XBCMat =  FreeBCMatrix(0,sp4,bp4,D_sp4);
ThetaBCMat = FreeThetBCMatrix(0,sp2,bp2,D_sp2);
ThetaBCMat_low = FreeThetBCMatrix(0,s,b,D);
if (clamp0)
    XBCMat_low = ClampedBCMatrix(0,s,b,D);
    XBCMat =  ClampedBCMatrix(0,sp4,bp4,D_sp4);
    ThetaBCMat = ClampedThetBCMatrix(0,sp2,bp2,D_sp2,Mrr_sp2,twmod);
    ThetaBCMat_low = ClampedThetBCMatrix(0,s,b,D,Mrr,twmod);
    if (TorqBC)
        ThetaBCMat = ClampedThetBCMatrixTorq(0,sp2,bp2,D_sp2,Mrr_sp2,twmod);
        ThetaBCMat_low = ClampedThetBCMatrixTorq(0,s,b,D,Mrr,twmod);
    end
    for iFib=1:nFib
        BCanswers((iFib-1)*12+1:(iFib-1)*12+6) = ...
            stackMatrix(ClampedBCMatrix(0,s,b,D))*reshape(fibpts(N*(iFib-1)+1:N*iFib,:)',3*N,1);
    end
end
if (clampL)
    XBCMat_low = [XBCMat_low; ClampedBCMatrix(L,s,b,D)];
    XBCMat =  [XBCMat; ClampedBCMatrix(L,sp4,bp4,D_sp4)];
    ThetaBCMat = [ThetaBCMat; ClampedThetBCMatrix(L,sp2,bp2,D_sp2,Mrr_sp2,twmod)];
    ThetaBCMat_low = [ThetaBCMat_low; ClampedThetBCMatrix(L,s,b,D,Mrr,twmod)];
    BCanswers(7:12) = stackMatrix(ClampedBCMatrix(L,s,b,D))*reshape(fibpts',3*N,1);
else
    XBCMat_low = [XBCMat_low; FreeBCMatrix(L,s,b,D)];
    XBCMat =  [XBCMat; FreeBCMatrix(L,sp4,bp4,D_sp4)];
    ThetaBCMat = [ThetaBCMat; FreeThetBCMatrix(L,sp2,bp2,D_sp2)];
    ThetaBCMat_low = [ThetaBCMat_low; FreeThetBCMatrix(L,s,b,D)];
end
XBCst = stackMatrix(XBCMat);
FE = -Eb*Rs*stackMatrix(D_sp4^4);
UpsampleXBCMat = [Rs;XBCst] \ [eye(3*N); zeros(12,3*N)];
[s2Np8,w2Np8,b2Np8] = chebpts(2*(N+4),[0 L]);
RNp4To2Np8 = barymat(s2Np8,sp4,bp4);
W2p8 = diag(w2Np8);
BendingEnergyMatrixBC = stackMatrix((RNp4To2Np8*D_sp4^2*UpsampleXBCMat(1:3:end,1:3:end))'...
    *W2p8*RNp4To2Np8*D_sp4^2*UpsampleXBCMat(1:3:end,1:3:end));
BCShift = zeros(3*(N+4)*nFib,1);
for iFib=1:nFib
    BCShift(3*(N+4)*(iFib-1)+1:3*(N+4)*iFib) = [Rs;XBCst] \ [zeros(3*N,1);BCanswers(12*(iFib-1)+1:12*iFib)];
end
UpsampleThetaBCMat = [R2ToN;ThetaBCMat];
ThetaImplicitMatrix = eye(N)-dt*twmod*R2ToN*D_sp2*Mrr_sp2*D_sp2*(UpsampleThetaBCMat \ [eye(N);zeros(2,N)]);
if (strongthetaBC)
    ThetaImplicitMatrix = eye(N)-dt*D*Mrr*twmod*D;
end
ThetaBC0=0;
if (clamp0)
    ThetaBC0= TurnFreq;
    if (TorqBC)
        ThetaBC0 = -w*diag(Mrr^(-1)).*TurnFreq/twmod;
    end
end
BasisMatrix=Lmat(:,1:N-1);
if (clamp0)
   BasisMatrix = BasisMatrix-repmat(barymat(0,s,b)*BasisMatrix,N,1);
end
BasisMatrix = stackMatrix(BasisMatrix);
Dinvone = pinv(D);
if (makeMovie)
    f=figure;
    movieframes=getframe(f);
end
Xpts=[];
forces=[];
Omegas=[];
links=[];
Thetass = [];
lambdas=zeros(3*N*nFib,1);
Omega=zeros(N*nFib,3);
lambdalast=zeros(3*N*nFib,1);
fext=zeros(3*N*nFib,1);
Xt= reshape(fibpts',3*N*nFib,1);
theta_sp1 = zeros(N*nFib,1);
theta_s_sp2 = zeros((N+2)*nFib,1);
Xtm1 = Xt;
Xst=reshape(X_s',3*N*nFib,1);
Xstm1=Xst;
stopcount=floor(tf/dt+1e-5);
gn=0;
chebForInts=1;
% Precompute the integrals necessary for the exact RPY integrals
% These are integrals of T_k sign(s-s') ds' and the like. 
AllbS = precomputeStokesletInts(s,L,a,N,chebForInts);
Allb_trueFP = precomputeStokesletInts(s,L,0,N,chebForInts);
AllbD = precomputeDoubletInts(s,L,a,N,chebForInts);
AllbS_Np1 = precomputeStokesletInts(sNp1,L,a,N+1,chebForInts);
AllbD_Np1 = precomputeDoubletInts(sNp1,L,a,N+1,chebForInts);
% # of pts for R < 2a integrals for exact RPY
warning('NForSmall set in init fiber vars!')
if (a/L*4/exp(3/2)>1e-3+1e-8)
    NForSmall = 8;
    if (N > 40)
        NForSmall = 12;
    end
else
    NForSmall = 4;
end

[spl,wpl]=chebpts(1000,[0 L]);
Rpl=barymat(spl,s,b);
% Initialize the material frame and Bishop frame. We will keep track of
% D1(L/2) = D1mid by explicitly rotating it, then updating the Bishop frame
% to be consistent with D1(L/2). 
% D1mid = zeros(nFib,3);
% D1 = zeros(nFib*N,3);
% D2 = zeros(nFib*N,3);
% bishA = zeros(nFib*N,3);
% bishB = zeros(nFib*N,3);
% for iFib=1:nFib
%     inds = (iFib-1)*3*N+1:iFib*3*N;
%     np4inds = (iFib-1)*3*(N+4)+1:iFib*3*(N+4);
%     Xsmid = barymat(L/2,s,b)*reshape(Xst(inds),3,N)';
%     thisD1mid = [Xsmid(2) -Xsmid(1) 0];
%     if (norm(thisD1mid) < 1e-10)
%         thisD1mid = [1 0 0];
%     end
%     D1mid(iFib,:)=thisD1mid/norm(thisD1mid);
%     XBC = UpsampleXBCMat*Xt(inds) + BCShift(np4inds);
%     Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
%     Xss3 = reshape(Xss,3,N)';
%     rinds = (iFib-1)*N+1:iFib*N;
%     [tbishA,tbishB,tD1,tD2] = computeBishopFrame(N,X_s(rinds,:),Xss3,s,b,L,theta0(rinds),D1mid(iFib,:)');
%     bishA(rinds,:)=tbishA;
%     bishB(rinds,:)=tbishB;
%     D1(rinds,:)=tD1;
%     D2(rinds,:)=tD2;
% end
Ders=[];
OmegaPar0s=[];
Xp1 = zeros(3*N*nFib,1);
Xsp1 = zeros(3*N*nFib,1);
lambdas = zeros(3*N*nFib,1);
fE = zeros(3*N*nFib,1);
fEprev = zeros(3*N*nFib,1);
fTw = zeros(3*N*nFib,1);
nparTw = zeros(N*nFib,1);
nLvel = zeros(3*N*nFib,1);
dU = zeros(3*N*nFib,1);
UFromTorq = zeros(3*N*nFib,1);
XBCprev = zeros(3*(N+4)*nFib,1);
XBC = XBCprev;
OmegaPar_Euler = zeros(N*nFib,1);


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