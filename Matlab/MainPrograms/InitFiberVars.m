% Initialization and precomputations for dynamic fiber simulations
% Derivative matrices and various upsampled grids
Lmat = cos(acos(2*s/L-1).*(0:N-1));
Ds = stackMatrix(D);
[sp4, ~, bp4] = chebpts(N+4, [0 L], 2);
D_sp4 = diffmat(N+4,1,[0 L]);
[sp2, ~, bp2] = chebpts(N+2, [0 L], 2);
D_sp2 = diffmat(N+2,[0 L]);
[sDouble, ~, bDouble] = chebpts(2*N, [0 L], 2);
LmatDouble = cos(acos(2*sDouble/L-1).*(0:2*N-1));
R4ToN = barymat(s, sp4, bp4); % Resampling matrix from s+4-> s0
R2ToN = barymat(s, sp2, bp2);
R2To4 = barymat(sp4,sp2,bp2);
R4ToDouble = barymat(sDouble,sp4,bp4);
RDoubleToN = barymat(s,sDouble,bDouble);
Rs = stackMatrix(R4ToN);
% Precompute the matrices associated with enforcing the weak BCs.
% See Section 4.3 in the paper for details on this
[~, ~, ~,Mrr_sp2,~] = getGrandMloc(N+2,zeros(3*(N+2),1),zeros(3*(N+2)),a,L,mu,sp2,deltaLocal);
[~, ~, ~,Mrr,~] = getGrandMloc(N,zeros(3*N,1),zeros(3*N),a,L,mu,s,deltaLocal);
BCanswers = zeros(12,1);
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
    BCanswers(1:6) = stackMatrix(ClampedBCMatrix(0,s,b,D))*reshape(fibpts',3*N,1);
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
BCShift = [Rs;XBCst] \ [zeros(3*N,1);BCanswers];
UpsampleThetaBCMat = [R2ToN;ThetaBCMat];
ThetaImplicitMatrix = eye(N)-dt*D*Mrr*twmod*R2ToN*D_sp2*(UpsampleThetaBCMat \ [eye(N);zeros(2,N)]);
if (strongthetaBC)
    ThetaImplicitMatrix = eye(N)-dt*D*Mrr*twmod*D;
end
ThetaBC0=0;
if (clamp0)
    ThetaBC0= TurnFreq;
    if (TorqBC)
        ThetaBC0 = -w*diag(Mrr^(-1))*TurnFreq/twmod;
    end
end
Dinvone = pinv(D);
I=zeros(3*N,3);
wIt=zeros(3,3*N);
for iR=1:N
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
    wIt(1:3,3*(iR-1)+1:3*iR)=w(iR)*eye(3);
end
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
% # of pts for R < 2a integrals for exact RPY
if (a/L > 1e-3)
    NForSmall = 8;
else
    NForSmall = 4;
end
[spl,wpl]=chebpts(1000,[0 L]);
Rpl=barymat(spl,s,b);
% Initialize the material frame and Bishop frame. We will keep track of
% D1(L/2) = D1mid by explicitly rotating it, then updating the Bishop frame
% to be consistent with D1(L/2). 
D1mid = zeros(nFib,3);
D1 = zeros(nFib*N,3);
D2 = zeros(nFib*N,3);
bishA = zeros(nFib*N,3);
bishB = zeros(nFib*N,3);
for iFib=1:nFib
    inds = (iFib-1)*3*N+1:iFib*3*N;
    Xsmid = barymat(L/2,s,b)*reshape(Xst(inds),3,N)';
    thisD1mid = [Xsmid(2) -Xsmid(1) 0];
    if (norm(thisD1mid) < 1e-10)
        thisD1mid = [1 0 0];
    end
    D1mid(iFib,:)=thisD1mid/norm(thisD1mid);
    XBC = UpsampleXBCMat*Xt(inds) + BCShift;
    Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
    Xss3 = reshape(Xss,3,N)';
    rinds = (iFib-1)*N+1:iFib*N;
    [tbishA,tbishB,tD1,tD2] = computeBishopFrame(N,X_s(rinds,:),Xss3,s,b,L,theta0(rinds),D1mid(iFib,:)');
    bishA(rinds,:)=tbishA;
    bishB(rinds,:)=tbishB;
    D1(rinds,:)=tD1;
    D2(rinds,:)=tD2;
end
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