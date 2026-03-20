% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
seed=2;
Nx=13;
dt=1e-4;
gtype=1;
addpath(genpath('../'))
LinkLocs = [0.02 0.1; 0.9 0.75; 0.6 0.51];
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
ell = 0.1;

Nlinks = size(LinkLocs,1);
rng(seed);
nFib = 2;
TotalDOFs = nFib*Nx;
NTaus = nFib*Nx - Nlinks -1;
if (mod(NTaus,2)==0)
    N1 = NTaus/2;
    N2 = NTaus/2;
else
    N2 = ceil(NTaus/2);
    N1 = N2 -1;
end
NLink1 = (Nx-1)-N1;
NLink2 = (Nx-1)-N2;
impcoeff = 1;
makeMovie = 1;
tf = 20;
Tau0 = [0 1 0];
Xbar = [0 0 0];
Locs10 = [-ell/2 -L/2 0]+LinkLocs(:,1)*Tau0;
Locs20 = [ell/2 -L/2 0]+LinkLocs(:,2)*Tau0;
Diff=Locs20-Locs10;
LinkLengths = sqrt(sum(Diff.*Diff,2));
LinkHat = Diff./LinkLengths;
if (gtype==2)
    gtypewords='chebkind2';
else
    gtypewords='chebkind1';
end

Xs3=repmat(Tau0,NTaus,1);

% Grids for tangent vectors and integration
[s1,~,b1] = chebpts(N1,[0 L], gtype);
[s2,~,b2] = chebpts(N2,[0 L], gtype);
[s1x,~,b1x] = chebpts(N1+1,[0 L], gtype); % auxillary grid from integrating tangents
DX1 = diffmat(N1+1,[0 L],gtypewords);
[s2x,~,b2x] = chebpts(N2+1,[0 L], gtype);
DX2 = diffmat(N2+1,[0 L],gtypewords);
LinkMat1 = barymat(LinkLocs(:,1),s1x,b1x);
LinkMat2 = barymat(LinkLocs(:,2),s2x,b2x);
% Set up so that the first links are on top of each other
XMat_1 = (eye(N1+1)-ones(N1+1,1).*LinkMat1(1,:))*pinv(DX1)*barymat(s1x,s1,b1);
XMat_2 = (eye(N2+1)-ones(N2+1,1).*LinkMat2(1,:))*pinv(DX2)*barymat(s2x,s2,b2);

[s1star,~,~] = chebpts(N1-NLink2,[0 L], gtype);
[s2star,~,~] = chebpts(N2-NLink1,[0 L], gtype);
AllNodes_1=[s1star;LinkLocs(1:2:end,1);LinkLocs(2:2:end,1)];
AllNodes_2=[s2star;LinkLocs(1,2);LinkLocs(2:2:end,2);LinkLocs(3:2:end,2)];
Np1TosLink_1 = barymat(AllNodes_1(1:N1+1),s1x,b1x);
Np1TosLink_2 = barymat(AllNodes_2(1:N2+1),s2x,b2x);


[sX,wX,bX]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],2);
ChebToNodes_1 = barymat(AllNodes_1,sX,bX);
PosToLinks_1 = barymat(LinkLocs(:,1),sX,bX);
NodesToCheb_1 = ChebToNodes_1^(-1);
ChebToNodes_2 = barymat(AllNodes_2,sX,bX);
NodesToCheb_2 = ChebToNodes_2^(-1);
if (~isempty(null(ChebToNodes_2)) || ~isempty(null(ChebToNodes_1)))
    error('Cross linker is on a grid point exactly')
end
PosToLinks_2 = barymat(LinkLocs(:,2),sX,bX);
altOnes = zeros(NLink1,Nlinks-1);
altTwos = zeros(NLink2,Nlinks-1);
constTwo = zeros(N2+1,Nlinks-1);
for pL=1:NLink1
    altOnes(pL,2*pL-1)=-LinkLengths(2:2:end);
end
for pL=1:NLink2
    altTwos(pL,2*pL)=LinkLengths(3:2:end);
end
% XMat1 and XMat2 are set such that the coordinates of the first link are
% on top of each other
DOFsToCustomNodes = [Np1TosLink_1*XMat_1 zeros(N1+1,N2+Nlinks); ...
    zeros(NLink1,N1) LinkMat2(2:2:end,:)*XMat_2 LinkLengths(1)*ones(NLink1,1) altOnes; ...
    zeros(N2+1,N1) Np1TosLink_2*XMat_2 LinkLengths(1)*ones(N2+1,1) zeros(N2+1,Nlinks-1); ...
    LinkMat1(3:2:end,:)*XMat_1 zeros(NLink2,N2) zeros(NLink2,1) altTwos];
% Fix the constant 
AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;

ChebMatZeroMean = SubAvg*blkdiag(NodesToCheb_1,NodesToCheb_2)*DOFsToCustomNodes;
DOFsToChebNodes = [ChebMatZeroMean ones(nFib*Nx,1)]; % CHECK THIS!
XMat = stackMatrix(DOFsToChebNodes);
InvXMat = XMat^(-1);
DOFs = [reshape(Xs3',[],1); reshape(LinkHat',[],1); Xbar'];
Xt = XMat* DOFs;

% Bending energy matrix (2Nx grid)
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sX,bX);
WTilde_Nx = stackMatrix((R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx));
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix(DX^2)'*WTilde_Nx*stackMatrix(DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));
BendMatAll = blkdiag(BendForceMat,BendForceMat);
BendMatHalfAll = blkdiag(BendMatHalf,BendMatHalf);
% Pre-computations for mobility
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-2/dt+1e-10));
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
mpdist=[];
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
nConstr=1;

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        DOFs = InvXMat*Xt;
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        MPs = blkdiag(barymat(L/2,sX,bX),barymat(L/2,sX,bX))*PtsThisT;
        LinkedPts = [PosToLinks_1 zeros(Nlinks,Nx); zeros(Nlinks,Nx) PosToLinks_2]*PtsThisT;
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            for pL = 1:Nlinks
                plot3(LinkedPts(pL:Nlinks:end,1),LinkedPts(pL:Nlinks:end,2),LinkedPts(pL:Nlinks:end,3),':ko')
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 1])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
        mpdist=[mpdist; norm(MPs(1,:)-MPs(2,:))];
        Xbars = [Xbars; DOFs(end-2:end,:)'];
    end  

    % Matrices at time step n 
    [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat);
    MWsym = zeros(nFib*3*Nx);
    MWsymHalf = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymOne = LocalDragMob(Xt(finds),DX,MobConst,WTilde_Nx_Inverse);
        MWsymHalf = chol(MWsymOne);
        MWsym(finds,finds)=MWsymOne;
        MWsymHalf(finds,finds)=MWsymHalf;
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
    alphaU = MobK*Ktilde'*(BendMatAll*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Evolve constants by rotating and translating the link
    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt);
    UActual = (Xp1-Xt)/dt;
    ULinear = KTogether*alphaU;
    Xt=Xp1;
end
ConstrErs=ConstrErs(1:saveEvery:end);
Totaltime=toc(tStart);
%save(strcat('CDOFType',num2str(gtype),'_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end

function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat)
    nTaus = length(Xt)/3-1;
    TauVelocity = zeros(length(Xt));
    InvTauVelocity = zeros(length(Xt));
    TausAndXBar = InvXMat*Xt;
    Tau3 = reshape(TausAndXBar(1:end-3),3,[])';
    % The matrix for all the taus (incl links) to evolve
    for iR =1:nTaus
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



