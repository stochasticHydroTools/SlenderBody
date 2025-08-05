% TODO: Redo this with multiple fibers to make it modular!!
% Especially when they all have different lengths
% Try not to make code a mess!
close all;
addpath(genpath('../'))
deltaP = 0.003; 
Fmems = -deltaP*(-6:2:10); % This is the mean displacement of the membane
FluctFil = 1;
FluctMem = 1;
nTrial=1;
Kster=0;
%for iF=1:length(Fmems)
%for iTrial=1:nTrial
% Polymerization
kPolyOn = 100; % 1/sec
dt = 1e-3;
nPolEvents = 0;
rng(0);
% Membrane geometry
Fmem = 0;%Fmems(iF);
mu = 1;
Lm = 1;
M = 16;
dx=Lm/M;
x=(0:M-1)*dx;
[xg,yg]=meshgrid(x,x);
kvals = [0:M/2 -M/2+1:-1]*2*pi/Lm;
[kx,ky]=meshgrid(kvals);
ksq=kx.^2+ky.^2;
KSqDiag = diag(ksq(:));
FMatBase = dftmtx(M);
FMat2 = kron(FMatBase,FMatBase);
Kcmem = 0.2;
%EnergyMatrixMem = Kcmem*real((FMat2'*(KSqDiag'*KSqDiag)*FMat2)/M^4*Lm^2);
Mmem = eye(M^2)/(8*pi*mu);
Mhalfmem = eye(M^2)/sqrt(8*pi*mu);
%ImpMatMem = eye(M^2)/dt + Mmem*EnergyMatrixMem;
%InvImpMatMem = ImpMatMem^(-1); % fix this later to Fourier
hmem = zeros(M^2,1);
Kh = 1;
FourierEnergyMat = Kcmem*ksq.^2*dx^2;
ImpfacFourier = (1/dt+Mmem(1,1)*FourierEnergyMat);

% Single fluctuating clamped filament
%FluctFil=0;
nFib = 3;
L = 1;   % microns
N = 12;
rtrue = 4e-3; % 4 nm radius
kbT = 4.1e-3;
lp = 10;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
impcoeff = 1;
makeMovie = 1;
tf = 100;
Tau0BC = [0;0;1];
TrkLoc=0;
XTrk0=[Lm/2;Lm/2;-L-deltaP];
XTrk=zeros(3,nFib);
rCirc = 0.02;
clamp=1;
for j = 1:nFib
    t=2*pi/nFib*(j-1);
    XTrk(:,j) = XTrk0+rCirc*[cos(t); sin(t); 0];
    Discr(j) = InitializeDiscretization(repmat(Tau0BC',N,1),...
        XTrk(:,j),TrkLoc,L,Eb,rtrue,mu,clamp);
end
saveEvery=floor(1e-1/dt+1e-10);

stopcount=floor(tf/dt+1e-5);
Xpts=[];
AllMemPts=[];
if (makeMovie)
    f=figure;
    frameNum=0;
end

% Equilibrate membrane
minh=[];
stcteq = floor(10/dt+1e-5);
for count=0:stcteq
    Fh = -Kh*hmem;
    Fpush = Fmem*ones(M^2,1);
    RHSmem = Mmem*(Fpush+Fh)...
        +FluctMem*sqrt(2*kbT/dt)*Mhalfmem*randn(M^2,1);
    % hnew = InvImpMatMem*(hmem/dt+RHSmem);
    % FFT way
    RHSHat = fft2(reshape(hmem/dt+RHSmem,M,M));
    hNewHat = RHSHat./ImpfacFourier;
    hnew = ifft2(hNewHat);
    hmem = hnew(:);
    minh=[minh;min(hmem)];
end
mempts=zeros(stopcount,1);
filH = zeros(stopcount,1);


%% Computations
for count=0:stopcount
    t=count*dt;
    Xt = [];
    for iFib=1:nFib
        Xt = [Xt;Discr(iFib).Xt];
    end
    if (mod(count,saveEvery)==0)
        %t
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                PtsThisFib = reshape(Discr(iFib).Xt,3,[])';
                Rpl = Discr(iFib).RplNp1;
                plot3(Rpl*PtsThisFib(:,1),Rpl*PtsThisFib(:,2),...
                    Rpl*PtsThisFib(:,3));
                hold on
            end
            title(sprintf('$t=$ %1.2f',(frameNum-1)*saveEvery*dt))
            view(2)
            ylim([0 Lm])
            xlim([0 Lm])
            zlim([-1 0.5])
            PlotAspect
            hold on
            XPl = [xg xg(:,1)+Lm];
            XPl = [XPl; XPl(1,:)];
            YPl = [yg; yg(1,:)+Lm];
            YPl = [YPl YPl(:,1)];
            hmempl = reshape(hmem,M,M);
            hmempl = [hmempl hmempl(:,1)];
            hmempl = [hmempl; hmempl(1,:)];
            surf(XPl,YPl,hmempl,'FaceColor','interp','FaceAlpha',0.5)
            view([-45 0])
            movieframes(frameNum)=getframe(f);
        end
        for iFib=1:nFib
            PtsThisFib = reshape(Discr(iFib).Xt,3,[])';
            Xpts=[Xpts;PtsThisFib];
        end
        AllMemPts =[AllMemPts hmem];
    end
    MemPtsAtX = InterpolatehNUFFT(reshape(hmem,M,M),[Xt(1:3:end) ...
        Xt(2:3:end)],ksq,x);
    % Compute the steric force
    dh = (MemPtsAtX - Xt(3:3:end))/deltaP; % Should be > 0
    stForce = -Kster*dh.*(dh < deltaP);
    Fgmem = zeros(M);
    for iP=1:length(stForce) % Just put it on the nearest grid pt
        xpt = 1 + round(Xt(1:3:end)/dx);
        ypt = 1 + round(Xt(2:3:end)/dx);
        xpt(xpt>M)=mod(xpt(xpt>M),M);
        xpt(xpt<0)=mod(xpt(xpt<0),M);
        xpt(xpt==0)=xpt(xpt==0)+M;
        ypt(ypt>M)=mod(ypt(ypt>M),M);
        ypt(ypt<0)=mod(ypt(ypt<0),M);
        ypt(ypt==0)=ypt(ypt==0)+M;
        Fgmem(ypt,xpt) = Fgmem(ypt,xpt) + stForce(iP);
    end
    Fgmem=Fgmem(:);
    for iFib=1:nFib
    if (kPolyOn*dt > 1) % Temp (this used to be > rand)
        % Possible polymerization event - check if membrane accomodates
        TauLast = barymat(L,s,b)*reshape(Xst,3,[])';
        XLast = barymat(L,sNp1,bNp1)*reshape(Xt,3,[])';
        Xadded = TauLast/norm(TauLast)*deltaP+XLast;
        % Check the membrane position at this (x,y)
        hmempt = InterpolatehNUFFT(reshape(hmem,M,M),Xadded(1:2),ksq,x);
        if (Xadded(3) < hmempt+1e-10)
            nPolEvents=nPolEvents+1;
            RandomPolymerization;
        end
    end
    end
    % Evolve the membrane
    Fh = -Kh*hmem;
    Fpush = Fmem*ones(M^2,1);
    RHSmem = Mmem*(Fpush+Fh+Fgmem)...
        +FluctMem*sqrt(2*kbT/dt)*Mhalfmem*randn(M^2,1);
    RHSHat = fft2(reshape(hmem/dt+RHSmem,M,M));
    hNewHat = RHSHat./ImpfacFourier;
    hnew = ifft2(hNewHat);
    hmem = hnew(:);
    minh=[minh;min(hmem)];

    % Evolve system
    if (FluctFil)
        RandomNums = randn(9*(N+1),1);
        for iFib=1:nFib
            Discr(iFib)=EvolveClampedFil(Discr(iFib),kbT,dt,impcoeff,RandomNums);
        end
    end
end
nEvents=nPolEvents/(tf*kPolyOn)
%end
%end


function Disc=EvolveClampedFil(Disc,kbT,dt,impcoeff,RandomNumbers,Fext)
    MobConst=Disc.MobConst;
    Nx = Disc.Nx;
    DNp1 = Disc.DNp1;
    N = Nx - 1;
    WTilde_Np1_Inverse=Disc.WTilde_Np1_Inverse;
    XonNp1Mat = Disc.XonNp1Mat;
    Xs3 = reshape(Disc.Xst,3,[])';
    Xti = XonNp1Mat*[Disc.Xst;Disc.TrkPt];
    MWsym = LocalDragMob(Xti,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = real(MWsym^(1/2));
    % Obtain Brownian velocity
    XonNp1Mat = Disc.XonNp1Mat;
    I = Disc.I;
    K = KonNp1(Xs3,XonNp1Mat,I);
    if (Disc.clamp)
        sNp1 = Disc.sNp1;
        bNp1 = Disc.bNp1;
        s = Disc.s;
        b = Disc.b;
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
    end
    g = RandomNumbers(1:3*Nx);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    % Advance to midpoint
    RNp1ToN = Disc.RNp1ToN;
    DNp1 = Disc.DNp1;
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    if (Disc.clamp)
        OmegaTilde = ProjectClamp*[reshape(OmegaTilde',[],1);zeros(3,1)];
        OmegaTilde = reshape(OmegaTilde,3,[])';
    end
    Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);Disc.TrkPt];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    if (Disc.clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*Ktilde;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClampTilde = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = Ktilde*ProjectClampTilde;
    else
        Kaug = Ktilde;
    end
    deltaRFD = 1e-5;
    WRFD = RandomNumbers(3*Nx+1:6*Nx);
    OmegaPlus = cross(Xs3,RNp1ToN*DNp1*reshape(WRFD,3,[])');
    if (Disc.clamp)
        OmegaPlus = ProjectClamp*[reshape(OmegaPlus',[],1);zeros(3,1)];
        OmegaPlus = reshape(OmegaPlus,3,[])';
    end
    TauPlus = rotateTau(Xs3,deltaRFD*OmegaPlus(1:N,:),1);
    XPlus = XonNp1Mat*[reshape(TauPlus',[],1);Disc.TrkPt];
    MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*WRFD;
    BERand = RandomNumbers(6*Nx+1:end);
    BendMatHalf_Np1 = Disc.BendMatHalf_Np1;
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf_Np1*BERand;
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    B = Kaug-impcoeff*dt*MWsymTilde*Disc.BendForceMat*Kaug;
    U0 = zeros(3*Nx,1);
    %U0(1:3:end)=1;
    Fext = zeros(3*Nx,1);
    %Fext(end-1) = 100*Eb;
    RHS = Kaug'*(Disc.BendForceMat*Xti+ Fext + ...
        MWsymTilde \ (RandomVel + U0));
    % Form psuedo-inverse manually
    maxRank = 2*N+3;
    if (Disc.clamp)
        maxRank = 2*N-2;
    end
    alphaU = ManualPinv(Kaug'*(MWsymTilde \ B),maxRank)*RHS;
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Disc.Xst = reshape(newXs',[],1);
    Disc.TrkPt = Disc.TrkPt+dt*alphaU(end-2:end);
    Disc.Xt = Disc.XonNp1Mat*[Disc.Xst;Disc.TrkPt];
end