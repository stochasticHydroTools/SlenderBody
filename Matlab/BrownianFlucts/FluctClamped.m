function FluctClamped(seed)
% Single fluctuating clamped filament
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
%dt = 2.5e-3;
tf = 30;
Tau0BC = [0;1;0];
TrkLoc=0;
XTrk=[0;TrkLoc;0];
clamp=0;
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
        keyboard
    end
    % Obtain Brownian velocity
    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    % Advance to midpoint
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    if (clamp)
        OmegaTilde(1,:)=0; % fix later
    end
    Xstilde = rotateTau(Xs3,OmegaTilde,dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XTrk];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    if (clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*Ktilde;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = Ktilde*ProjectClamp;
    else
        Kaug = Ktilde;
    end
    deltaRFD = 1e-5;
    WRFD = randn(3*Nx,1); % This is Delta X on the N+1 grid
    OmegaPlus = cross(Xs3,RNp1ToN*DNp1*reshape(WRFD,3,[])');
    if (clamp)
        OmegaPlus(1,:)=0;
    end
    TauPlus = rotateTau(Xs3,deltaRFD*OmegaPlus(1:N,:),1);
    XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XTrk];
    MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*WRFD;
    %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    RandomVelBE = sqrt(kbT)*...
        MWsymTilde*BendMatHalf_Np1*randn(3*Nx,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    B = Kaug-impcoeff*dt*MWsymTilde*BendForceMat*Kaug;
    U0 = zeros(3*Nx,1);
    %U0(1:3:end)=1;
    RHS = Kaug'*(BendForceMat*Xt+MWsymTilde \ (RandomVel + U0));
    % Form psuedo-inverse manually
    maxRank = 2*N+3;
    if (clamp)
        maxRank = 2*N-2;
    end
    alphaU = ManualPinv(Kaug'*(MWsymTilde \ B),maxRank)*RHS;
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
save(strcat('ClampedSim_',seed,'.mat'))
end