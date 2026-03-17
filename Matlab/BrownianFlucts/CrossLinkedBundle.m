% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
seed=2;
Nx=13;
dt=1e-4;
gtype=1;
gtypeX=2;
addpath(genpath('../'))
Nlinks = 1;
LinkLocs = [0 0];
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
makeMovie = 1;
tf = 20;
Tau0 = [0;1;0];
Link0 = [1;0;0];
Xbar = [0;L/2;0];

Xs3=repmat(Tau0',nFib*N,1);
% Add rows for the constraints 
[s,~,b] = chebpts(N,[0 L], gtype);
[sNx,wNx,bNx]=chebpts(Nx,[0 L],gtypeX);
if (gtypeX==2)
    DX = diffmat(Nx,[0 L],'chebkind2');
elseif (gtypeX==1)
    DX = diffmat(Nx,[0 L],'chebkind1');
else
    error('Enter valid grid type for X');
end
IntDX = pinv(DX);
RxToLoc_1 = barymat(LinkLocs(:,1),sNx,bNx);
RxToLoc_2 = barymat(LinkLocs(:,2),sNx,bNx);
RXToN = barymat(s,sNx,bNx);
RToX = barymat(sNx,s,b);

I = repmat(eye(3),Nx,1);
XFromTaus = zeros(3*nFib*Nx,3*nFib*N+3*Nlinks);
% Integrating the tangents
XFromTaus(1:3*Nx,1:3*N) = stackMatrix(IntDX*RToX);
XFromTaus(3*Nx+(1:3*Nx),3*N+(1:3*N)) = (eye(3*Nx)-I*stackMatrix(RxToLoc_2))*stackMatrix(IntDX*RToX);
XFromTaus(3*Nx+(1:3*Nx),1:3*N) = I*stackMatrix(RxToLoc_1)*stackMatrix(IntDX*RToX);
XFromTaus(3*Nx+(1:3*Nx),3*N*nFib+(1:3))=ell*I;
% Subtract a constant and add
AvgMat = 1/(nFib*L)*repmat(stackMatrix(wNx),1,nFib);
SubAvg = eye(3*Nx*nFib)-repmat(I,nFib,1)*AvgMat;
XMat = [SubAvg*XFromTaus repmat(I,nFib,1)];
InvXMat = XMat^(-1);

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

Xt = XMat* [reshape(Xs3',[],1);Link0;Xbar];
saveEvery=max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
mpdist=[];
Xbars=[];
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
        DOFs = InvXMat*Xt;
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        MPs = blkdiag(barymat(L/2,sNx,bNx),barymat(L/2,sNx,bNx))*PtsThisT;
        LinkedPts = [RxToLoc_1 zeros(1,Nx); zeros(1,Nx) RxToLoc_2]*PtsThisT;
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
    Nlinks = Nx - N;
    TauVelocity = zeros(3*Nx*nFib);
    InvTauVelocity = zeros(3*Nx*nFib);
    TausAndXBar = InvXMat*Xt;
    % The matrix for all the taus to evolve
    for iFib=1:nFib
        finds = 3*N*(iFib-1)+1:3*N*iFib;
        Xs3 = reshape(TausAndXBar(finds),3,N)';
        TauCross = zeros(3*N);
        for iR=1:N
            inds = (iR-1)*3+1:iR*3;
            TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
        end
        TauVelocity(finds,finds) =  -TauCross;
        InvTauVelocity(finds,finds) = TauCross;
    end
    % The matrix for the links to evolve
    for iLink=1:Nlinks
        ellInds = 3*N*nFib+(3*iLink-2:3*iLink);
        RLink = TausAndXBar(ellInds);
        CPMatR = CPMatrix(RLink);
        % Velocity of each DOF should be = UL + RLink cross Omega
        TauVelocity(ellInds,ellInds)=-CPMatR;
        InvTauVelocity(ellInds,ellInds)=CPMatR;
    end
    TauVelocity(end-2:end,end-2:end)=eye(3);
    InvTauVelocity(end-2:end,end-2:end)=eye(3);
    KTogether = XMat*TauVelocity;
    KTogetherInv = InvTauVelocity*InvXMat;
end

function XNew = updateByRotate(nFib,N,Nx,Xt,alphaU,XMat,InvXMat,dt)
    Nlinks = Nx - N;
    TausXBar = InvXMat*Xt;
    TausXBarNew = zeros(3*Nx*nFib,1);
    % Update the tangent vectors
    for iFib=1:nFib
        finds = 3*N*(iFib-1)+1:3*N*iFib;
        Xs3 = reshape(TausXBar(finds),3,N)';
        Omega = reshape(alphaU(finds),3,N)';
        newXs = rotateTau(Xs3,Omega,dt);
        TausXBarNew(finds)=reshape(newXs',[],1);
    end
    % Update the links
    for iLink=1:Nlinks
        ellInds = 3*N*nFib+(3*iLink-2:3*iLink);
        RLink = TausXBar(ellInds);
        OmLink = alphaU(ellInds);
        TausXBarNew(ellInds)=rotateTau(RLink',OmLink',dt);
    end
    % Add the constant
    TausXBarNew(end-2:end)=TausXBar(end-2:end)+dt*alphaU(end-2:end);
    XNew = XMat*TausXBarNew;
end



