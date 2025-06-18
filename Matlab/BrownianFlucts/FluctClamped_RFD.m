function FluctClamped_RFD(seed)
% Single fluctuating clamped filament
rng(str2num(seed))
addpath(genpath('../'))
%close all;
nFib = 1;
L = 1;   % microns
N = 12;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
Eb = L*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie =0;
dt = 1e-4;
tf = 30;
Tau0BC = [0;1;0];
TrkLoc=0;
XTrk=[0;TrkLoc;0];
clamp=1;
X_s=repmat(Tau0BC',N,1);
[s,w,b] = chebpts(N, [0 L], 2); 
InitializationNoTwist;
saveEvery=floor(1e-2/dt+1e-10);
ee=[];
MidPointSolve = 1;
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
Xpts=[];
AllXs = Xst;
AllX = Xt;
if (makeMovie)
    f=figure;
    frameNum=0;
end

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        PtsThisT = reshape(Xt,3,Nx)';
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            plot3(RplNp1*PtsThisT(:,1),RplNp1*PtsThisT(:,2),...
                RplNp1*PtsThisT(:,3));
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt))
            view(2)
            ylim([0 2])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
    end
    % Evolve system
    Xs3 = reshape(Xst,3,N)';
    Xt = XonNp1Mat*[Xst;XTrk];
    MWsym = LocalDragMob(Xt,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = MWsym^(1/2);
    if (max(abs(imag(MWsymHalf(:))))>0)
        error('Imag eigs!')
    end
    % Obtain Brownian velocity
    g = randn(3*Nx,1);
    RandomVel = sqrt(2*kbT/dt)*MWsymHalf*g;
    RandomVelBE = sqrt(kbT)*MWsym*BendMatHalf_Np1*randn(3*Nx,1);
    % Solve for fiber evolution
    K = KonNp1(Xs3,XonNp1Mat,I);
    % For clamping (B matrix)
    if (clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = K*ProjectClamp;
    else
        Kaug = K;
    end
    B = Kaug-impcoeff*dt*MWsym*BendForceMat*Kaug;
    U0 = zeros(3*Nx,1);
    %U0(1:3:end)=1;
    RHS = Kaug'*(BendForceMat*Xt+MWsym \ (RandomVel + U0));
    % Form psuedo-inverse manually
    maxRank = 2*N+3;
    if (clamp)
        maxRank = 2*N-2;
    end
    alphaU = ManualPinv(Kaug'*(MWsym \ B),maxRank)*RHS;
    % Add the RFD part
    deltaRFD = 1e-5;
    N_og = ManualPinv(Kaug'*(MWsym \ Kaug),maxRank);
    WRFD = randn(3*N+3,1); % This is Delta X on the N+1 grid
    if (clamp)
        WRFD = ProjectClamp*WRFD;
    end
    TauPlus = rotateTau(Xs3,deltaRFD*reshape(WRFD(1:3*N),3,N)',1);
    XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XTrk];
    MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    KPlus = KonNp1(TauPlus,XonNp1Mat,I);
    if (clamp)
        KaugPlus = KPlus*ProjectClamp;
    else
        KaugPlus = KPlus;
    end
    N_Plus = ManualPinv(KaugPlus'*(MWsymPlus \ KaugPlus),maxRank);
    N_RFD = 1/(deltaRFD)*(N_Plus-N_og)*WRFD;
    % Add the RFD term to the saddle point solve
    alphaU = alphaU + kbT*N_RFD;
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Xsp1 = reshape(newXs',[],1);
    XTrk_p1 = XTrk+dt*alphaU(end-2:end);
    Xp1 = XonNp1Mat*[Xsp1;XTrk_p1];
    Xt=Xp1;
    Xst = Xsp1;
    XTrk = XTrk_p1;
    ee=[ee;norm(Xt(1:3)-Xt(end-2:end))];
end
save(strcat('ClampedRFDSim_',seed,'Dt',num2str(dt),'.mat'))
end