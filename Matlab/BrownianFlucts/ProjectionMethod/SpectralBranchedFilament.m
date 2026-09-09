% Projection method for spectral chain
function SpectralBranchedFilament(seed,Nx,dt)
%if (0)
%dt=1e-3;
%Nx=8;
addpath(genpath('../../'))
nRuns = 1;
%seed=1;
clamp0=0;
nFib=2;

L = 1;
kbT = 4.1e-3; % pN * um
lp = L;
K_b = lp*kbT;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
mu = 0.6;
delta = 1e-5;
%dt=2.5e-4;
implicit=1;
tf = 100;
nSt = (tf/dt);
saveEvery=max(1,floor(1e-2/dt+1e-10));
nSave = floor(1e-6+nSt/saveEvery);
rng(seed);
MaxIts = 25;
tol = 1e-10;
x0=[0;0;0];
tau0=[1;0;0];

rotvec = 70/180*pi*[0 0  1]; % rotation vector for the branch
dotprod = cos(norm(rotvec));
branchpt = 0.8;
taubr = rotateTau(tau0',rotvec,1);

[sX,wX,bX]=chebpts(Nx,[0 L],2);
s = chebpts(Nx-1,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
D = barymat(s,sX,bX)*DX;
D = kron(D,eye(3));
BranchEvalMat = [barymat(branchpt,sX,bX) zeros(1,Nx); ...
    barymat(branchpt,sX,bX)*DX zeros(1,Nx); zeros(1,Nx) barymat(0,sX,bX); ...
    zeros(1,Nx) barymat(0,sX,bX)*DX];
BranchEvalMat=kron(BranchEvalMat,eye(3));
GradMat = @(x) GradMatrix(x,D,BranchEvalMat,clamp0);
constr = @(x) c(x,D,BranchEvalMat,dotprod,clamp0,x0,tau0);

% Energy matrix
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sX,bX);
WTilde_1D = R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx;
WTilde_Inv = kron(WTilde_1D^(-1),eye(3));
WTilde_Nx = stackMatrix(WTilde_1D);
EMat = K_b*stackMatrix(DX^2)'*WTilde_Nx*...
    stackMatrix(DX^2);
EMat = blkdiag(EMat,EMat);

% Hydrodynamics
AllbS_Np1 = precomputeStokesletInts(sX,L,rtrue,Nx,1);
AllbD_Np1 = precomputeDoubletInts(sX,L,rtrue,Nx,1);
NForSmall = 8; % # of pts for R < 2a integrals for exact RPY
eigThres = 1e-3;
MobilityOne = @(x1) RPYQuadMob(x1,rtrue,L,mu,sX,bX,DX,AllbS_Np1,AllbD_Np1,...
    NForSmall,WTilde_Inv,eigThres);
Mobility = @(x) blkdiag(MobilityOne(x(1:3*Nx)),MobilityOne(x(3*Nx+1:end)));

nW = 1;
AllTanVecDots = zeros(nRuns,Nx-1);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,2*nSave);
MDDist = zeros(nRuns,nSave);
Xpts=[];

for iRun=1:nRuns
% Initial state
mother = x0'+sX.*tau0';
branch = x0'+branchpt.*tau0'+sX.*taubr;
x=[mother;branch];
x = reshape(x',[],1);
nC = nFib*(Nx-1)+4; % branchpt 
if (clamp0)
    nC = nC+5;
end
nX = length(x);
c0=constr(x);
% Gradient check
dx = rand(3*Nx*2,1);
gc = GradMat(x)*dx;
for iEps=1:10
    trudiff = constr(x+10^(-iEps)*dx);
    ers(iEps) = norm(trudiff/10^(-iEps)-gc);
end

% Statistics
NumIts = zeros(nSave,1);
ee = zeros(2*nSave,1);
mdrun = zeros(nSave,1);
nFail = 0;
nReallyFail = 0;
opts=optimoptions(@fsolve,'OptimalityTolerance',1e-10,...
    'SpecifyObjectiveGradient',true,'Display','off');

% Unconstrained step
for iT=1:nSt
    [xg,newtits,newtfail,matfail] = ...
        AdvanceProjection(x,dt,kbT,EMat,Mobility,GradMat,constr,...
        MaxIts,tol,nC,implicit,opts);
    if (newtfail)
        nFail=nFail+1;
        if (matfail)
            nReallyFail=nReallyFail+1;
        end
    end
    x = xg;
    if (mod(iT,saveEvery)==0)
        index = floor(1e-10+iT/saveEvery);
        NumIts(index)=newtits;
        ee(2*index-1)=norm(x(1:3)-x(3*Nx-2:3*Nx));
        ee(2*index)=norm(x(3*Nx+1:3*Nx+3)-x(6*Nx-2:6*Nx));
        PtsThisT=reshape(x,3,[])';
        Xpts=[Xpts;PtsThisT];
        MotherEnd = barymat(L,sX,bX)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sX,bX)*PtsThisT(Nx+1:2*Nx,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        mdrun(index)= dispMDT;
    end
end
AllEE(iRun,:)=ee;
FailureRates(iRun) = nFail/nSt;
AllItCounts(iRun,:)=NumIts;
MDDist(iRun,:)=MDDist;
end
save(strcat('BranchRPYProj_Lp',num2str(lp),...
    '_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
end

function cd = c(x,D,BranchEvalMat,dotprod,clamp0,x0,tau0)
    % EvalBPs is a 4 row matrix. Row 1 = mother X @ bp, Row 2 = mother tau
    % @ bp, Row 3 = daugher X @ 0, Row 4 = daughter tau @ 0
    Nx = size(D,2)/3;
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    tau = [reshape(D*x(1:3*Nx),3,[])'; reshape(D*x(3*Nx+1:end),3,[])'];
    cd = sum(tau.*tau,2)-1;
    if (clamp0)
        cd = [cd(2:end); x(1:3)-x0; tau(1,:)'-tau0];
    end
    % Branching constraints
    bpts = BranchEvalMat*x;
    bposer = bpts(1:3)-bpts(7:9);
    tauEr = dot(bpts(10:12),bpts(4:6))-dotprod;
    cd = [cd; bposer; tauEr];
end

function C = GradMatrix(x,D,BranchEvalMat,clamp0)
    Nx = size(D,2)/3;
    nFib = length(x)/(3*Nx);
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    C = zeros(2*Nx-2,3*Nx*nFib);
    for j=1:Nx-1
        DDt = D(3*j-2:3*j,:)'*D(3*j-2:3*j,:);
        for iFib=1:nFib
            finds=(iFib-1)*3*Nx+(1:3*Nx);
            C((Nx-1)*(iFib-1)+j,finds)=(2*DDt*x(finds,:))';
        end
    end
    if (clamp0)
        Ct = zeros(2*Nx+4,3*Nx*nFib);
        Ct(1:2*Nx-3,:)=C(2:end,:);
        Ct(2*Nx-2:2*Nx,1:3)=eye(3);
        Ct(2*Nx+1:2*Nx+3,1:3*Nx)=D(1:3,:);
        C=Ct;
    end
    % Branching constraint
    C(end+1:end+3,:)=[BranchEvalMat(1:3,1:3*Nx) -BranchEvalMat(7:9,3*Nx+1:end)];
    bpts = BranchEvalMat*x;
    C(end+1,:)=[bpts(10:12)'*BranchEvalMat(4:6,1:3*Nx) bpts(4:6)'*BranchEvalMat(10:12,3*Nx+1:end)];
end