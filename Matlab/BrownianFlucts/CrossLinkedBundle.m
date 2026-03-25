function CrossLinkedBundle(seed,Nx,dt)
% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations

%% Define constants 
%seed=30;
%Nx=13;
%dt=1e-5;
gtype=1;
addpath(genpath('../'))
LinkLocs = [0 0; 1 1];
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
ell = 0.1;

%% Initialization
Nlinks = size(LinkLocs,1);
rng(seed);
nFib = 2;
% With every link you lose 1 tangent vector (between the two fibers)
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
makeMovie = 0;
tf = 50;
Tau0 = [0 1 0];
Xbar = [0 0 0];
Locs10 = [-ell/2 -L/2 0]+LinkLocs(:,1)*Tau0;
Locs20 = [ell/2 -L/2 0]+LinkLocs(:,2)*Tau0;
Xs3=repmat(Tau0,NTaus,1);
Diff=Locs20-Locs10;
LinkLengths = sqrt(sum(Diff.*Diff,2));
LinkHat = Diff./LinkLengths;

%% Calculation of the X matrix
% Grids for tangent vectors and integration
[s1,~,b1] = chebpts(N1,[0 L], gtype);
[s2,~,b2] = chebpts(N2,[0 L], gtype);
[sX,wX,bX]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
[sNoLinks,~] = chebpts(Nx-Nlinks,[0 L], gtype);
%dnL = 1/(Nx-Nlinks);
%sNoLinks=(1/2:Nx-Nlinks)'*dnL; % Uniform points are a terrible idea!
% When you put extra nodes (Nlinks>1), you're still going to run into instabilties
% with the polynomials because you are essentially adding a term of degree
% s^(Nx-1) to make all the conditions come out. Fiber shapes might look
% funky because of that. But those oscillations get damped by the bending
% force, so it should work ok. 

% Order of the nodes
% First set = no links and the first link (does not affect tau's)
% Second set = those that are master on this filament
% Third set = those that are slave to the other filament
AllNodes_1=[sNoLinks;LinkLocs(1:2:end,1);LinkLocs(2:2:end,1)];
AllNodes_2=[sNoLinks;LinkLocs(1,2);LinkLocs(2:2:end,2);LinkLocs(3:2:end,2)];
ChebToNodes_1 = barymat(AllNodes_1,sX,bX);
NodesToCheb_1 = ChebToNodes_1^(-1);
ChebToNodes_2 = barymat(AllNodes_2,sX,bX);
NodesToCheb_2 = ChebToNodes_2^(-1);
% Try linear interpolation instead
Id = eye(Nx);
for jC=1:Nx
    NodesToCheb_1(:,jC) = interp1(AllNodes_1,Id(:,jC),sX);
    NodesToCheb_2(:,jC) = interp1(AllNodes_2,Id(:,jC),sX);
end
if (~isempty(null(ChebToNodes_2)) || ~isempty(null(ChebToNodes_1)))
    error('Cross linker is on a grid point exactly')
end
XToLink_1 = barymat(AllNodes_1,sX,bX);
XToLink_2 = barymat(AllNodes_2,sX,bX);

% Set up so that the first links are on top of each other
XMat_1 = (eye(Nx)-ones(Nx,1).*XToLink_1(Nx-Nlinks+1,:))*pinv(DX)*barymat(sX,s1,b1);
XMat_2 = (eye(Nx)-ones(Nx,1).*XToLink_2(Nx-Nlinks+1,:))*pinv(DX)*barymat(sX,s2,b2);

% Matrices to increment the cross linkers
altOnes = zeros(NLink1,Nlinks-1);
altTwos = zeros(NLink2,Nlinks-1);
constTwo = zeros(N2+1,Nlinks-1);
for pL=1:NLink1
    altOnes(pL,2*pL-1)=-LinkLengths(2*pL);
end
for pL=1:NLink2
    altTwos(pL,2*pL)=LinkLengths(2*pL+1);
end
DOFsToCustomNodes = [XToLink_1(1:N1+1,:)*XMat_1 zeros(N1+1,N2+Nlinks); ...
    zeros(NLink1,N1) XToLink_2((Nx-Nlinks+1)+(1:NLink1),:)*XMat_2 LinkLengths(1)*ones(NLink1,1) altOnes; ...
    zeros(N2+1,N1) XToLink_2(1:N2+1,:)*XMat_2 LinkLengths(1)*ones(N2+1,1) zeros(N2+1,Nlinks-1); ...
    XToLink_1((Nx-Nlinks+1)+(1:NLink2),:)*XMat_1 zeros(NLink2,N2) zeros(NLink2,1) altTwos];

% Fix the constant 
AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
ChebMatZeroMean = SubAvg*blkdiag(NodesToCheb_1,NodesToCheb_2)*DOFsToCustomNodes;
DOFsToChebNodes = [ChebMatZeroMean ones(nFib*Nx,1)];
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

%% Initialize arrays to save 
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-2/dt+1e-10));
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

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        DOFs = InvXMat*Xt;
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        MPs = blkdiag(barymat(L/2,sX,bX),barymat(L/2,sX,bX))*PtsThisT;
        LinkedPts = [barymat(LinkLocs(:,1),sX,bX) zeros(Nlinks,Nx); ...
            zeros(Nlinks,Nx) barymat(LinkLocs(:,2),sX,bX)]*PtsThisT;
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
    alphaU = MobK*Ktilde'*(BendMatAll*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Evolve constants by rotating and translating the link
    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt);
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('LinintConstrBundle_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts')
end

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



