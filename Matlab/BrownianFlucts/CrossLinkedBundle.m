% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
% TODO: Figure out the inverse of K for multiple links. 
% Understand source of instabilities when links are too close to the ends. 
seed=1;
Nx=12;
dt=1e-3;
gtype=1;
addpath(genpath('../'))
Nlinks = 2;
LinkLocs = [0.25 0.75];
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
impcoeff = 1;
makeMovie = 1;
tf = 5;
Tau0 = [0;1;0];
LinkLocs=reshape(LinkLocs,Nlinks,1);
XLocs = zeros(3*Nlinks,nFib);
XLocs(1:3:end,:)=repmat([0 0.1],Nlinks,1);
XLocs(2:3:end,:)=repmat(LinkLocs,1,2);

Xs3=repmat(Tau0',nFib*N,1);
% Add rows for the constraints 
[s,~,b] = chebpts(N,[0 L], gtype);
[sNx,~,bNx]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
RXToLocs = barymat(LinkLocs,sNx,bNx);
RXToN = barymat(s,sNx,bNx);

InvXonNp1Mat = [RXToN*DX; RXToLocs];
if (~isempty(null(InvXonNp1Mat)))
    error('Polynomial is not unique - switch discretization')
end
XonNp1Mat = InvXonNp1Mat^(-1);
InvXonNp1Mat = stackMatrix(InvXonNp1Mat);
XonNp1Mat = stackMatrix(XonNp1Mat);
I=zeros(3*Nx,3);
for iR=1:Nx
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
BMNp1 = stackMatrix(RXToLocs(1,:));


% Matrix that extracts the link velocities
P = zeros(6*Nlinks,nFib*3*Nx);
AvgMat = zeros(3*Nlinks,6*Nlinks);
DiffMat = zeros(3*Nlinks,6*Nlinks);
for d=1:3
    for iLink=1:Nlinks
        AvgMat(3*(iLink-1)+d,3*(iLink-1)+d:3*Nlinks:end)=1/2;
        DiffMat(3*(iLink-1)+d,3*(iLink-1)+d:3*Nlinks:end)=1/2;
        % Fiber 1 
        P(3*(iLink-1)+d,3*N+3*(iLink-1)+d)=1;
        % Fiber 2
        P(3*Nlinks+3*(iLink-1)+d,3*Nx+3*N+3*(iLink-1)+d)=1;
    end
end
DiffMat(:,3*Nlinks+1:end)=-1*DiffMat(:,3*Nlinks+1:end);
LinkInverse = [AvgMat;DiffMat]*P*blkdiag(InvXonNp1Mat,InvXonNp1Mat);

% Bending energy matrix (2Nx grid)
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sNx,bNx);
WTilde_Nx = stackMatrix((R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx));
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix((R_Nx_To_2Nx*DX^2)'*...
    W2Nx*R_Nx_To_2Nx*DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));
BendMatAll = blkdiag(BendForceMat,BendForceMat);
BendMatHalfAll = blkdiag(BendMatHalf,BendMatHalf);

Xt = zeros(3*nFib*Nx,1);
for iFib=1:nFib
    Xt(3*Nx*(iFib-1)+1:3*Nx*iFib) = XonNp1Mat* [reshape(Xs3(N*(iFib-1)+1:N*iFib,:)',[],1);XLocs(:,iFib)];
end
saveEvery=max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
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
        DOFs = blkdiag(InvXonNp1Mat(1:3:end,1:3:end),InvXonNp1Mat(1:3:end,1:3:end))*PtsThisT;
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
                linkPts=[DOFs(N+pL,:); DOFs(Nx+N+pL,:)];
                plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko')
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 2])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
    end  

    % Matrices at time step n 
    [KTogether,KTogetherInv] = KWithLink(nFib,N,Nx,Xt,XonNp1Mat,InvXonNp1Mat);
    MWsym = zeros(nFib*3*Nx);
    MWsymHalf = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        Xs3 = XsXTrk(1:Nx-1,:);
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
    Xtilde = updateByRotate(nFib,N,Nx,Xt,OmegaTilde,XonNp1Mat,dt);
    Ktilde = KWithLink(nFib,N,Nx,Xtilde,XonNp1Mat,InvXonNp1Mat);
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
    Xp1 = updateByRotate(nFib,N,Nx,Xt,alphaU,XonNp1Mat,dt);
    UActual = (Xp1-Xt)/dt;
    ULinear = KTogether*alphaU;
    Xt=Xp1;
end
ConstrErs=ConstrErs(1:saveEvery:end);
Totaltime=toc(tStart);
%save(strcat('CDOFType',num2str(gtype),'_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end

function [KTogether,KTogetherInv] = KWithLink(nFib,N,Nx,Xt,XonNp1Mat,InvXonNp1Mat)
    DOFVelocity = zeros(nFib*3*Nx);
    Nlinks = Nx - N;
    TrkPts = zeros(3*nFib*Nlinks,1);
    % The matrix for all the taus to evolve
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        XTrk = XsXTrk(end-Nlinks+1:end,:)';
        TrkPts(3*Nlinks*(iFib-1)+1:3*Nlinks*iFib)=XTrk(:);
        Xs3 = XsXTrk(1:N,:);
        TauCross = zeros(3*N);
        for iR=1:N
            inds = (iR-1)*3+1:iR*3;
            TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
        end
        FibOmegaInds = 3*N*(iFib-1)+(1:3*N);
        FibTauInds = 3*Nx*(iFib-1)+(1:3*N);
        DOFVelocity(FibTauInds,FibOmegaInds) =  -TauCross;
        %InverseDOFTogether(3*Nx*(iFib-1)+1:3*Nx*(iFib-1)+3*N,3*Nx*(iFib-1)+1:3*Nx*(iFib-1)+3*N) = TauCross;
    end
    % The matrix for the links to evolve
    for iLink=1:Nlinks
        RLink = 1/2*(TrkPts((3*Nlinks+(iLink-1)*3+1:3*(Nlinks+iLink)))'-TrkPts((iLink-1)*3+1:3*iLink)');
        CPMatR = CPMatrix(RLink);
        % Velocity of each DOF should be = UL + RLink cross Omega
        LinkUInds = nFib*3*N+((iLink-1)*3+1:3*iLink);
        LinkOmInds = nFib*3*N+3*Nlinks+((iLink-1)*3+1:3*iLink);
        Fib1Inds = 3*N+((iLink-1)*3+1:3*iLink);
        Fib2Inds = 3*Nx+3*N+((iLink-1)*3+1:3*iLink);
        DOFVelocity(Fib1Inds,LinkUInds)=eye(3);
        DOFVelocity(Fib2Inds,LinkUInds)=eye(3);
        DOFVelocity(Fib1Inds,LinkOmInds)=-CPMatR;
        DOFVelocity(Fib2Inds,LinkOmInds)=CPMatR;
        % Inverse operation Uc = (U1+U2)/2
        % Compute matrix representation you are using to get U and Omega
    end
    KTogether = blkdiag(XonNp1Mat,XonNp1Mat)*DOFVelocity;
    KTogetherInv = pinv(KTogether);
end

function XNew = updateByRotate(nFib,N,Nx,Xt,alphaU,XonNp1Mat,dt)
    Nlinks = Nx - N;
    TrkPts = zeros(3*nFib*Nlinks,1);
    XsXtrk = zeros(3*Nx*nFib,1);
    XNew = zeros(3*Nx*nFib,1);
    % Update the tangent vectors
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk3 = reshape(XonNp1Mat \ Xt(finds),3,Nx)';
        XTrk = XsXTrk3(end-Nlinks+1:end,:)';
        TrkPts(3*Nlinks*(iFib-1)+1:3*Nlinks*iFib)=XTrk(:);
        Xs3 = XsXTrk3(1:N,:);
        Omega = reshape(alphaU((iFib-1)*3*N+1:iFib*3*N),3,N)';
        newXs = rotateTau(Xs3,Omega,dt);
        XsXtrk((iFib-1)*3*Nx+1:(iFib-1)*3*Nx+3*N)=reshape(newXs',[],1);
    end
    % Update the links
    for iLink=1:Nlinks
        LinkInds = 3*iLink-2:3*iLink;
        linkpts = [TrkPts(LinkInds)'; TrkPts(3*Nlinks+LinkInds)']; 
        r = linkpts(2,:)-linkpts(1,:);
        xc = mean(linkpts);
        uc = alphaU(6*N+LinkInds)';
        OmLink = alphaU(6*N+3*Nlinks+LinkInds)';
        xc=xc+dt*uc;
        linkHat = rotateTau(r,-OmLink,dt);
        XsXtrk(3*N+LinkInds)=xc-linkHat/2;
        XsXtrk(3*Nx+3*N+LinkInds)=xc+linkHat/2;
    end
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XNew(finds) = XonNp1Mat*XsXtrk(finds);
    end
end



