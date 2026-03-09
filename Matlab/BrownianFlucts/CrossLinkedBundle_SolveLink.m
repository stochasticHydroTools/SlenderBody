%function FluctClamped_AddDOFs(seed,ForceRt,N,dt,gtype)
% Single fluctuating clamped filament
seed=1;
N=16;
dt=1e-3;
gtype=1;
addpath(genpath('../'))
%close all;
rng(seed);
nFib = 2;
Nc_offgrid = 0; % Number of off-grid tangent vector constraints 
Nog = N - Nc_offgrid;
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
Tau0BC = [0;1;0];
TrkLoc = L/2;
XTrk=[0 0.1;TrkLoc TrkLoc;0 0];
[sog,~,~] = chebpts(Nog, [0 L], gtype);
Xs3=repmat(Tau0BC',nFib*N,1);
% Add rows for the constraints 
[s,~,b] = chebpts(N,[0 L], gtype);
sC=s;
ChebToConstr = eye(N);
ConstrToCheb = eye(N);
Nx = N + 1;
[sNp1,~,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*Nx,3);
for iR=1:Nx
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
XonNp1Mat = [(eye(3*Nx)-repmat(BMNp1,Nx,1))*...
    stackMatrix(IntDNp1*RToNp1*ConstrToCheb) I];
InvXonNp1Mat = [stackMatrix(ConstrToCheb \ RNp1ToN*DNp1); BMNp1];

% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, ~] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
BendMatHalf_Np1 = real(BendingEnergyMatrix_Np1^(1/2));

Xt = zeros(3*nFib*Nx,1);
for iFib=1:nFib
    Xt(3*Nx*(iFib-1)+1:3*Nx*iFib) = XonNp1Mat* [reshape(Xs3(N*(iFib-1)+1:N*iFib,:)',[],1);XTrk(:,iFib)];
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
RplNp1 = barymat(spl,sNp1,bNp1);
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
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            linkPts=[PtsThisT(9,:); PtsThisT(26,:)];
            plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko')
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 2])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(end,:))];
    end
    % Evolve system
    MWsymTilde = zeros(nFib*3*Nx);
    MWsym = zeros(nFib*3*Nx);
    BendMatAll = zeros(nFib*3*Nx);
    BendMatHalfAll = zeros(nFib*3*Nx);
    % Replace K with U and Omega
    KTogether = KWithLink(nFib,N,Nx,Xt,XonNp1Mat,I);
    KTogetherInv = pinv(KTogether);
    RandomVelBM = zeros(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        Xs3 = XsXTrk(1:Nx-1,:);
        MWsymOne = LocalDragMob(Xt(finds),DNp1,MobConst,WTilde_Np1_Inverse);
        MWsymHalf = real(MWsymOne^(1/2));
        % Obtain Brownian velocity
        g = randn(3*Nx,1);
        RandomVelBM(finds) = sqrt(2*kbT/dt)*MWsymHalf*g;
        MWsym(finds,finds)=MWsymOne;
        BendMatAll(finds,finds)=BendForceMat;
        BendMatHalfAll(finds,finds)=BendMatHalf_Np1;
    end
    % Advance to midpoint
    OmegaTilde = KTogetherInv*RandomVelBM;
    Xtilde = updateByRotate(nFib,N,Nx,Xt,OmegaTilde,XonNp1Mat,dt);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = LocalDragMob(Xtilde(finds),DNp1,MobConst,WTilde_Np1_Inverse);
        MWsymTilde(finds,finds)=MWsymTildeOne;
    end

    % Solve at midpoint
    Ktilde = KWithLink(nFib,N,Nx,Xtilde,XonNp1Mat,I);
    M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalfAll*randn(3*Nx*nFib,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    KWithImp=Ktilde-impcoeff*dt*MWsymTilde*BendMatAll*Ktilde;
    BendForce=BendMatAll*Xt;
    U0 = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    alphaU = MobK*Ktilde'*(BendForce+ Fext + MWsymTilde \ (RandomVel + U0));
    % Evolve constants by rotating and translating the link
    Xp1 = updateByRotate(nFib,N,Nx,Xt,alphaU,XonNp1Mat,dt);
    Xt=Xp1;
end
ConstrErs=ConstrErs(1:saveEvery:end);
Totaltime=toc(tStart);
%save(strcat('CDOFType',num2str(gtype),'_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end

function KTogether = KWithLink(nFib,N,Nx,Xt,XonNp1Mat,I)
    KTogether = zeros(nFib*3*Nx);
    TrkPts = zeros(3*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(XonNp1Mat \ Xt(finds),3,Nx)';
        XTrk = XsXTrk(end,:)';
        TrkPts(3*iFib-2:3*iFib)=XTrk;
        Xs3 = XsXTrk(1:Nx-1,:);
        KOne=KonNp1(Xs3,XonNp1Mat,I);
        KTogether(3*Nx*(iFib-1)+1:3*Nx*iFib,3*N*(iFib-1)+1:3*N*iFib) = KOne(:,1:3*N);
    end
    % Computing velocity of the midpoint from Omega and Tau of the Link
    RLink = 1/2*(TrkPts(4:6)'-TrkPts(1:3)');
    CPMatR = CPMatrix(RLink);
    % MP velocity = ULink +/- R x Omega
    KTogether(:,end-5:end-3)=[I;I];
    KTogether(:,end-2:end)=[-repmat(CPMatR,Nx,1); repmat(CPMatR,Nx,1)];
end

function XNew = updateByRotate(nFib,N,Nx,Xt,alphaU,XonNp1Mat,dt)
    TrkPts = zeros(3*nFib,1);
    XsXtrk = zeros(3*Nx*nFib,1);
    XNew = zeros(3*Nx*nFib,1);
    % Update the tangent vectors
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk3 = reshape(XonNp1Mat \ Xt(finds),3,Nx)';
        XTrk = XsXTrk3(end,:)';
        TrkPts(3*iFib-2:3*iFib)=XTrk;
        Xs3 = XsXTrk3(1:Nx-1,:);
        Omega = reshape(alphaU((iFib-1)*3*N+1:iFib*3*N),3,N)';
        newXs = rotateTau(Xs3,Omega,dt);
        XsXtrk((iFib-1)*3*Nx+1:iFib*3*Nx-3)=reshape(newXs',[],1);
    end
    % Update the link
    linkpts = [TrkPts(1:3)'; TrkPts(4:6)'];
    r = linkpts(2,:)-linkpts(1,:);
    xc = mean(linkpts);
    uc = alphaU(end-5:end-3)';
    xc=xc+dt*uc;
    OmLink = alphaU(end-2:end)';
    linkHat = rotateTau(r,-OmLink,dt);
    XsXtrk(3*Nx-2:3*Nx)=xc-linkHat/2;
    XsXtrk(end-2:end)=xc+linkHat/2;
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XNew(finds) = XonNp1Mat*XsXtrk(finds);
    end
end



