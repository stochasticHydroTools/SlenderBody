% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
% TODO: Figure out the inverse of K for multiple links. 
% Understand source of instabilities when links are too close to the ends. 
seed=1;
Nx=13;
dt=1e-4;
gtype=1;
gtypeX=2;
addpath(genpath('../'))
Nlinks = 1;
LinkLocs = [0];
%close all;
rng(seed);
nFib = 2;
N = Nx - Nlinks; % Number of off-grid tangent vector constraints 
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
ell = 0.1;
impcoeff = 1;
makeMovie = 0;
tf = 100;
Tau0 = [0;1;0];
LinkLocs=reshape(LinkLocs,Nlinks,1);
LinkCenters = [ell/2*ones(Nlinks,1) LinkLocs zeros(Nlinks,1)];
LinkVectors = [ones(Nlinks,1) zeros(Nlinks,2)];
LinkDOFs = [reshape(LinkCenters',[],1); reshape(LinkVectors',[],1)];

XFibFromLinkMat = zeros(6*Nlinks);
for iLink=1:Nlinks
    linkInds = 3*(iLink-1)+1:3*iLink;
    XFibFromLinkMat(linkInds,linkInds)= eye(3); % Fiber 1
    XFibFromLinkMat(3*Nlinks+linkInds,linkInds)=eye(3); % Fiber 2
    XFibFromLinkMat(linkInds,3*Nlinks+linkInds)= -ell/2*eye(3); % Fiber 1
    XFibFromLinkMat(3*Nlinks+linkInds,3*Nlinks+linkInds)=ell/2*eye(3); % Fiber 2
end
XDOFFromLink = zeros(6*Nx);
XDOFFromLink(1:3*N,1:3*N)=eye(3*N);
XDOFFromLink(3*N+1:3*Nx,6*N+1:end)=XFibFromLinkMat(1:3*Nlinks,:);
XDOFFromLink(3*Nx+1:3*Nx+3*N,3*N+1:6*N)=eye(3*N);
XDOFFromLink(3*Nx+3*N+1:end,6*N+1:end)=XFibFromLinkMat(3*Nlinks+1:end,:);

Xs3=repmat(Tau0',nFib*N,1);
% Add rows for the constraints 
[s,~,b] = chebpts(N,[0 L], gtype);
[sp1,~,bp1]=chebpts(N+1,[0,L],gtype);
sTot=[sp1;LinkLocs(2:end)];
[sNx,~,bNx]=chebpts(Nx,[0 L],gtypeX);
if (gtype==2)
    Dp1 = diffmat(N+1,[0 L],'chebkind2');
elseif (gtype==1)
    Dp1 = diffmat(N+1,[0 L],'chebkind1');
else
    error('Enter valid grid type for N+1');
end
if (gtypeX==2)
    DX = diffmat(Nx,[0 L],'chebkind2');
elseif (gtypeX==1)
    DX = diffmat(Nx,[0 L],'chebkind1');
else
    error('Enter valid grid type for X');
end
RXToLocs = barymat(LinkLocs,sNx,bNx);
RXToN = barymat(s,sNx,bNx);
RToX = barymat(sNx,s,b);
RNxToDOF = barymat(sTot,sNx,bNx);
if (rank(RNxToDOF) < Nx)
    error('Rank deficient matrix - your links are on the phantom grid points')
end
% Option 1: integrate using the first constant, then integrate again using
% the other constants (this still generates high frequency stuff)
% RDOFToNx = RNxToDOF^(-1);
% RToNp1 = barymat(sp1,s,b);
% 
% XonNp1Mat = [(eye(3*(N+1))-repmat(stackMatrix(BL1),N+1,1))*stackMatrix(pinv(Dp1)*RToNp1) repmat(eye(3),N+1,1)];
% XMat = stackMatrix(RDOFToNx)*[XonNp1Mat zeros(3*(N+1),3*(Nlinks-1)); zeros(3*(Nlinks-1),3*(N+1)) eye(3*(Nlinks-1))];

% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=repmat(eye(3),Nx,1);
BL1 = barymat(LinkLocs(1),sNx,bNx);
XMat = [(eye(3*Nx)-repmat(stackMatrix(BL1),Nx,1))*...
    stackMatrix(pinv(DX)*RToX) I];
XMat=blkdiag(XMat,XMat)*XDOFFromLink;
InvXMat = [stackMatrix(RXToN*DX); stackMatrix(BL1)];
InvXMat = XDOFFromLink \ blkdiag(InvXMat,InvXMat);

if (0)
% Option 2: Subtract the constant associated with the closest link pt! This is the
% most robust. 
SubMat = zeros(Nx);
AddMat = zeros(Nx,Nlinks);
for p=1:Nx
    [~,ind]=min(abs(LinkLocs-sNx(p)));
    SubMat(p,:)=barymat(LinkLocs(ind),sNx,bNx);
    AddMat(p,ind)=1;
end
XMat = stackMatrix([(eye(Nx)-SubMat)*pinv(DX)*barymat(sNx,s,b) AddMat]);
InvXMat = XMat^(-1);

% Option 3: 
% This is a delicate thing - not correct right now!
% The way it is being done now, looking for the Nx degree polynomial that
% goes through the tangent vectors and the points. Creates high order
% instabilities (the bending forces are not "getting through" to the
% tangents in this cases)
InvXonNp1Mat = [RXToN*DX; RXToLocs];
if (~isempty(null(InvXonNp1Mat)))
    error('Polynomial is not unique - switch discretization')
end
XonNp1Mat = InvXonNp1Mat^(-1);
InvXonNp1Mat = stackMatrix(InvXonNp1Mat);
XonNp1Mat = stackMatrix(XonNp1Mat);
end

% Bending energy matrix (2Nx grid)
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sNx,bNx);
WTilde_Nx = stackMatrix((R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx));
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix(DX^2)'*WTilde_Nx*stackMatrix(DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));
BendMatAll = blkdiag(BendForceMat,BendForceMat);
BendMatHalfAll = blkdiag(BendMatHalf,BendMatHalf);

Xt = XMat* [reshape(Xs3',[],1);LinkDOFs];
saveEvery=max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
mpdist=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNx,bNx);
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
        %t
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        MPs = blkdiag(barymat(L/2,sNx,bNx),barymat(L/2,sNx,bNx))*PtsThisT;
        LinkedPts = reshape(XFibFromLinkMat*InvXMat(end-6*Nlinks+1:end,:)*Xt,3,[])';
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
                plot3(LinkedPts(2*pL-1:2*pL,1),LinkedPts(2*pL-1:2*pL,2),LinkedPts(2*pL-1:2*pL,3),':ko')
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 4])
            xlim([-1 2])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
        mpdist=[mpdist; norm(MPs(1,:)-MPs(2,:))];
    end  

    % Matrices at time step n 
    [KTogether,KTogetherInv] = KWithLink(nFib,N,Nx,Xt,XMat,InvXMat);
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
    Xtilde = updateByRotate(nFib,N,Nx,Xt,OmegaTilde,XMat,InvXMat,dt/2);
    Ktilde = KWithLink(nFib,N,Nx,Xtilde,XMat,InvXMat);
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
    Xp1 = updateByRotate(nFib,N,Nx,Xt,alphaU,XMat,InvXMat,dt);
    UActual = (Xp1-Xt)/dt;
    ULinear = KTogether*alphaU;
    Xt=Xp1;
end
ConstrErs=ConstrErs(1:saveEvery:end);
Totaltime=toc(tStart);
%save(strcat('CDOFType',num2str(gtype),'_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end

function [KTogether,KTogetherInv] = KWithLink(nFib,N,Nx,Xt,XMat,InvXMat)
    DOFVelocity = zeros(nFib*3*Nx);
    InvDOFVelocity = zeros(nFib*3*Nx);
    Nlinks = Nx - N;
    XsLinkDOFs = InvXMat*Xt;
    % The matrix for all the taus to evolve
    for iFib=1:nFib
        finds = 3*N*(iFib-1)+1:3*N*iFib;
        Xs3 = reshape(XsLinkDOFs(finds),3,N)';
        TauCross = zeros(3*N);
        for iR=1:N
            inds = (iR-1)*3+1:iR*3;
            TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
        end
        DOFVelocity(finds,finds) =  -TauCross;
        InvDOFVelocity(finds,finds) = TauCross;
    end
    % The matrix for the links to evolve
    for iLink=1:Nlinks
        trinds = (nFib*3*N)+(1:3);
        rotinds = (nFib*3*N+3*Nlinks)+(1:3);
        RLink = XsLinkDOFs(rotinds);
        CPMatR = CPMatrix(RLink);
        % Velocity of each DOF should be = UL + RLink cross Omega
        DOFVelocity(trinds,trinds)=eye(3);
        DOFVelocity(rotinds,rotinds)=-CPMatR;
        InvDOFVelocity(trinds,trinds)=eye(3);
        InvDOFVelocity(rotinds,rotinds)=CPMatR;
    end
    KTogether = XMat*DOFVelocity;
    KTogetherInv = InvDOFVelocity*InvXMat;
end

function XNew = updateByRotate(nFib,N,Nx,Xt,alphaU,XMat,InvXMat,dt)
    Nlinks = Nx - N;
    XsLinkDOFs = InvXMat*Xt;
    XsLinkDOFsNew = zeros(3*Nx*nFib,1);
    % Update the tangent vectors
    for iFib=1:nFib
        finds = 3*N*(iFib-1)+1:3*N*iFib;
        Xs3 = reshape(XsLinkDOFs(finds),3,N)';
        Omega = reshape(alphaU(finds),3,N)';
        newXs = rotateTau(Xs3,Omega,dt);
        XsLinkDOFsNew(finds)=reshape(newXs',[],1);
    end
    % Update the links
    for iLink=1:Nlinks
        trinds = (nFib*3*N)+(1:3);
        rotinds = (nFib*3*N+3*Nlinks)+(1:3);
        XLink = XsLinkDOFs(trinds);
        RLink = XsLinkDOFs(rotinds);
        OmLink = alphaU(rotinds);
        XsLinkDOFsNew(trinds)=XLink+dt*alphaU(trinds);
        XsLinkDOFsNew(rotinds)=rotateTau(RLink',OmLink',dt);
    end
    XNew = XMat*XsLinkDOFsNew;
end



