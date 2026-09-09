function FluctClamped(seed,Nx,dt)
% Single fluctuating clamped filament
%for seed=1:30
ForceRt=0;
clampL=0;
%seed=1;
%Nx=8;
%dt=1e-2;
N = Nx-1;
gtype=2;
ConfineZ=0;
addpath(genpath('../../'))
%close all;
rng(seed);
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 1;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;
impcoeff = 1;
makeMovie = 0;
%clampL=1;
tf = 100 ;
Tau0BC = [1;0;0];
%Tau0BC=rotate(Tau0BC',-70/180*pi*[0 0 1])';
TrkLoc = 0;
try
    [s,~,b] = chebpts(N, [0 L], gtype);
catch
    [s,~,b] = chebpts(N, [0 L], 1);
end
Xs3=repmat(Tau0BC',N,1);
% Add rows for the constraints 
sC=s;
if (gtype==1)
    % Replace first and last entry with L
    sC(1)=0;
    if (clampL)
        sC(end)=L;
    end
    ChebToConstr = barymat(sC,s,b);
    ConstrToCheb = ChebToConstr^(-1);
elseif (gtype=='u')
    sC=(0:N-1)'/(N-1)*L;
    ChebToConstr = barymat(sC,s,b);
    ConstrToCheb = ChebToConstr^(-1);
elseif (gtype==2)
    ChebToConstr = eye(N);
    ConstrToCheb = eye(N);
end
[sNp1,~,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from X_s
XonNp1Mat = (eye(3*Nx)-repmat(BMNp1,Nx,1))*stackMatrix(IntDNp1*RToNp1*ConstrToCheb);
InvXonNp1Mat = stackMatrix(ConstrToCheb \ RNp1ToN*DNp1);

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

% Hydrodynamics
AllbS_Np1 = precomputeStokesletInts(sNp1,L,rtrue,Nx,1);
AllbD_Np1 = precomputeDoubletInts(sNp1,L,rtrue,Nx,1);
NForSmall = 8; % # of pts for R < 2a integrals for exact RPY
eigThres = 1e-3;
Xt = XonNp1Mat*reshape(Xs3',[],1);
MobConst = -log(eps^2)/(8*pi*mu);
Mobility = @(Xt) RPYQuadMob(Xt,rtrue,L,mu,sNp1,bNp1,DNp1,AllbS_Np1,AllbD_Np1,...
    NForSmall,WTilde_Np1_Inverse,eigThres);
%Mobility = @(Xt) LocalDragMob(Xt,DNp1,MobConst,WTilde_Np1_Inverse);
        
saveEvery=max(1,floor(1e-2/dt+1e-10));

%% Initialization 
stopcount=floor(tf/dt+1e-5);
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
%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        %t
        PtsThisT = reshape(Xt,3,Nx)';
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            plot3(RplNp1*PtsThisT(:,1),RplNp1*PtsThisT(:,2),...
                RplNp1*PtsThisT(:,3));
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 1])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(1,:)-PtsThisT(Nx,:))];
    end

    % Evolve system
    Xs3 = reshape(InvXonNp1Mat*Xt,3,[])';
    MWsym = Mobility(Xt);
    MWsymHalf = chol(MWsym)';
    TauVelocity = zeros(3*N);
    % The matrix for all the taus to evolve
    for iR =1:size(Xs3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Xs3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
    end
    % The COM
    KInv = -TauVelocity*InvXonNp1Mat;
    if (clampL)
        KInv([1:3;3*N-2:3*N],:)=[];
    else
        KInv(1:3,:)=[];
    end

    % Obtain Brownian velocity
    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;

    % Advance to midpoint
    OmegaTilde = cross(Xs3,ChebToConstr*RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    % Fix constrained variables
    if (clampL)
        OmegaTilde([1;N],:)=0;
    else
        OmegaTilde(1,:)=0;
    end
    Xstilde = rotateTau(Xs3,OmegaTilde,dt/2);
    Xtilde = XonNp1Mat*reshape(Xstilde',[],1);
    MWsymTilde = Mobility(Xtilde);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,[]);
    if (clampL)
        Ktilde(:,[1:3;3*N-2:3*N])=[];
    else
        Ktilde(:,1:3)=[];
    end

    % Set up and solve system
    deltaRFD = 1e-5;
    if (clampL)
        WRFD = randn(3*(N-2),1); % This is Delta X on the N+1 grid
        WRFDom= [zeros(3,1); WRFD; zeros(3,1)];
    else
        WRFD = randn(3*(N-1),1); % This is Delta X on the N+1 grid
        WRFDom= [zeros(3,1); WRFD];
    end
    TauPlus = rotateTau(Xs3,reshape(WRFDom(1:3*N),3,[])',deltaRFD);
    XPlus = XonNp1Mat*reshape(TauPlus',[],1);
    MWsymPlus = Mobility(XPlus);
    M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*KInv'*WRFD;

    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf_Np1*randn(3*Nx,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    U0 = zeros(3*Nx,1);
    Fext = zeros(3*Nx,1);
    Fext(end-2) = ForceRt^2*Eb;
    if (ConfineZ)
        Fext(3:3:end)=-Xt(3:3:end);
    end
    KWithImp = Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    alphaU = MobK* Ktilde'*(BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    Omega = reshape(alphaU,3,[])';
    if (clampL)
        Omega = [zeros(1,3); Omega; zeros(1,3)];
    else
        Omega = [zeros(1,3); Omega];
    end
    newXs = rotateTau(Xs3,Omega,dt);
    Xsp1 = reshape(newXs',[],1);
    Xp1 = XonNp1Mat*Xsp1;
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('ClmpRPYPar_Lp',num2str(lp),...
    '_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
end