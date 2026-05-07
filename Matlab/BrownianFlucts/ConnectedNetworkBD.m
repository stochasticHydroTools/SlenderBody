% Dense matrix operations
%function BranchedFibers(seed,Nx,dt)
% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations

%% Define constants 
%seed=30;
%Nx=13;
%dt=1e-3;
gtype=1;
addpath(genpath('../'))
BranchPts = [0.8]; % on fiber #1
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;

%% Initialization
rng(seed);
nFib = 2;
nBr = length(BranchPts);
N = Nx-1;
impcoeff = 1;
makeMovie = 0;
tf = 25;
Tau0 = [0 1 0];
RotAng = 70/180*pi;
TauBr = Tau0*[cos(RotAng) -sin(RotAng) 0; sin(RotAng) cos(RotAng) 0; 0 0 1]';
Xbar = [0 0 0];
Xs3=[repmat(Tau0,N,1);repmat(TauBr,N,1)];
BranchInds = zeros(1,nBr);

%% Calculation of the X matrix
% Grids for tangent vectors and integration
[s,~,b] = chebpts(N,[0 L],gtype);
sTru_Mother=s;
sTru_Branches=s;
for p=1:nBr
    % This assumes branches are not too close together
    [~,indmin]=min(abs(sTru_Mother-BranchPts(p)));
    sTru_Mother(indmin)=BranchPts(p);
    BranchInds(p) = indmin;
end
sTru_Branches(1)=0;
ChebToConstr_Br = barymat(sTru_Branches,s,b);
ConstrToCheb_Br = ChebToConstr_Br^(-1);
ChebToConstr_Mother = barymat(sTru_Mother,s,b);
ConstrToCheb_Mother = ChebToConstr_Mother^(-1);
[sX,wX,bX]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
BranchPtMat = barymat(BranchPts,sX,bX);
ZeroMat = barymat(0,sX,bX);

% Set up so that the first links are on top of each other
XMat_Mother = pinv(DX)*barymat(sX,s,b)*ConstrToCheb_Mother;
XMat_Br = (eye(Nx)-ones(Nx,1).*ZeroMat)*pinv(DX)*barymat(sX,s,b)*ConstrToCheb_Br;

DOFsToCustomNodes = [XMat_Mother zeros(Nx,N); ones(Nx,1)*BranchPtMat(1,:)*XMat_Mother XMat_Br];

% Only involves the first link
AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
ChebMatZeroMean = SubAvg*DOFsToCustomNodes;
DOFsToChebNodes = [ChebMatZeroMean ones(nFib*Nx,1)];
XMat = stackMatrix(DOFsToChebNodes);
InvXMat = pinv(XMat);%^(-1);
DOFs = [reshape(Xs3',[],1); Xbar'];
Xt = XMat* DOFs;

% Bending energy matrix (2Nx grid)
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sX,bX);
WTilde_1D = R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx;
WTilde_Nx = stackMatrix(WTilde_1D);
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix(DX^2)'*WTilde_Nx*stackMatrix(DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));
BendMatAll = blkdiag(BendForceMat,BendForceMat);
BendMatHalfAll = blkdiag(BendMatHalf,BendMatHalf);
% Pre-computations for mobility
MobConst = -log(eps^2)/(8*pi*mu);

% Need constraint matrix here for the alpha evolution
ConsMat = zeros(nBr,nFib*N);
for p=1:nBr
    ConsMat(p,BranchInds(p))=1;
    ConsMat(p,p*N+1)=-1;
end
ConsMat=stackMatrix(ConsMat);
ConsMat = [ConsMat zeros(nBr*3,3)];

%% Initialize arrays to save 
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-2/dt+1e-10));
Xpts=[];
ee=[];
MDDist=[];
Xbars=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sX,bX);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        DOFs = InvXMat*Xt;
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 1])
            xlim([-1 1])
            PlotAspect
            Locs1=barymat(sTru_Mother,sX,bX)*PtsThisT(1:Nx,:);
            DOF3=reshape(DOFs,3,[])';
            Xs1=DOF3(1:N,:);
            quiver3(Locs1(:,1),Locs1(:,2),Locs1(:,3),Xs1(:,1),Xs1(:,2),Xs1(:,3),'LineWidth',2,'AutoScaleFactor',0.5)
            Locs2=barymat(sTru_Branches,sX,bX)*PtsThisT(Nx+1:2*Nx,:);
            Xs2=DOF3(N+1:2*N,:);
            quiver3(Locs2(:,1),Locs2(:,2),Locs2(:,3),Xs2(:,1),Xs2(:,2),Xs2(:,3),'LineWidth',2,'AutoScaleFactor',0.5)
            movieframes(frameNum)=getframe(f);
        end
        MotherEnd = barymat(L,sX,bX)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sX,bX)*PtsThisT(Nx+1:end,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        MDDist=[MDDist; dispMDT];
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
        Xbars = [Xbars; DOFs(end-2:end,:)'];
    end  

    % Matrices at time step n 
    [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat);
    MWsym = zeros(nFib*3*Nx);
    MWsymHalf = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymOne = LocalDragMob(Xt(finds),DX,MobConst,WTilde_Nx_Inverse);
        MWsymHalfOne = chol(MWsymOne)';
        MWsym(finds,finds)=MWsymOne;
        MWsymHalf(finds,finds)=MWsymHalfOne;
    end
    % Obtain Brownian velocity
    g = randn(3*Nx*nFib,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;

    % Advance to midpoint
    OmegaTilde = KTogetherInv*RandomVelBM;
    Xtilde = updateByRotate(Xt,OmegaTilde,XMat,InvXMat,dt/2);
    Ktilde = KWithLink(Xtilde,XMat,InvXMat);
    MWsymTilde = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = LocalDragMob(Xtilde(finds),DX,MobConst,WTilde_Nx_Inverse);
        MWsymTilde(finds,finds)=MWsymTildeOne;
    end
    
    % Solve at midpoint
    M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalfAll*randn(3*Nx*nFib,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    KWithImp=Ktilde-impcoeff*dt*MWsymTilde*BendMatAll*Ktilde;
    U0 = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    MobC = ConsMat'*((ConsMat*MobK*ConsMat') \ ConsMat);
    alphaU = (MobK - MobK*MobC*MobK)*...
        Ktilde'*(BendMatAll*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Evolve constants by rotating and translating the link
    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt);
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('Branched_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts','MDDist')
end

function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat)
    TausAndXBar = InvXMat*Xt;
    TauVelocity = zeros(length(TausAndXBar));
    InvTauVelocity = zeros(length(TausAndXBar));
    Tau3 = reshape(TausAndXBar(1:end-3),3,[])';
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
        InvTauVelocity(inds,inds) = CMat;
    end
    % The COM
    TauVelocity(end-2:end,end-2:end)=eye(3);
    InvTauVelocity(end-2:end,end-2:end)=eye(3);
    KTogether = XMat*TauVelocity;
    KTogetherInv = InvTauVelocity*InvXMat;
end

function XNew = updateByRotate(Xt,alphaU,XMat,InvXMat,dt)
    TausXBar = InvXMat*Xt;
    TausXBarNew = 0*TausXBar;
    % Update the tangent vectors
    Taus = reshape(TausXBar(1:end-3),3,[])';
    Omega = reshape(alphaU(1:end-3),3,[])';
    newTaus = rotateTau(Taus,Omega,dt);
    TausXBarNew(1:end-3)=reshape(newTaus',[],1);
    % Add the constant
    TausXBarNew(end-2:end)=TausXBar(end-2:end)+dt*alphaU(end-2:end);
    XNew = XMat*TausXBarNew;
end



