function SingleFreeFilament(seed,Nx,dt)
% Single fluctuating clamped filament
%seed=1;
N = Nx-1;
%dt=1e-4;
gtype=1;
addpath(genpath('../../'))
%close all;
rng(seed);
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;
impcoeff = 1;
makeMovie = 0;
tf = 25;
Tau0BC = [0;1;0];
TrkLoc = L/2;
[s,~,b] = chebpts(N, [0 L], gtype);
Xs3=repmat(Tau0BC',N,1);
% Add rows for the constraints 
[sNp1,~,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from X_s
XonNp1Mat = [(eye(3*Nx)-repmat(BMNp1,Nx,1))*stackMatrix(IntDNp1*RToNp1) repmat(eye(3),N+1,1)];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];

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

Xt = XonNp1Mat*[reshape(Xs3',[],1); 0; 0; 0];
saveEvery=10;%max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);

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
    Xs3 = Xs3(1:N,:);
    MWsym = LocalDragMob(Xt,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = real(MWsym^(1/2));

    % Obtain Brownian velocity
    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    TauVelocity = zeros(3*N+3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Xs3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Xs3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
    end
    TauVelocity(end-2:end,end-2:end)=eye(3);
    % The COM
    KInv = -TauVelocity*InvXonNp1Mat;
    K = KonNp1(Xs3,XonNp1Mat,repmat(eye(3),N+1,1));

    % Advance to midpoint
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    % Fix constrained variables
    Xstilde = rotateTau(Xs3,OmegaTilde,dt/2);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1); zeros(3,1)];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,repmat(eye(3),N+1,1));

    % Set up and solve system
    %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    g3 = randn(3*N+3,1);
    OmRFD = g3;
    delta = 1e-5;
    XsPlus = rotateTau(Xs3,reshape(OmRFD(1:3*N),3,[])',delta);
    XPlus = XonNp1Mat*[reshape(XsPlus',[],1); zeros(3,1)];
    KPlus = KonNp1(XsPlus,XonNp1Mat,repmat(eye(3),N+1,1));
    TauVelocity = zeros(3*N+3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(XsPlus,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(XsPlus(iR,:));
        TauVelocity(inds,inds) =  -CMat;
    end
    % The COM
    TauVelocity(end-2:end,end-2:end)=eye(3);
    KPlusInv = -TauVelocity*InvXonNp1Mat;
    MWSymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    M_RFD = kbT/delta*(MWSymPlus-MWsym)*KInv'*g3;

    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf_Np1*randn(3*Nx,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    U0 = zeros(3*Nx,1);
    Fext = zeros(3*Nx,1);
    KWithImp = Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    alphaU = MobK* Ktilde'*(BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    Omega = reshape(alphaU,3,[])';
    newXs = rotateTau(Xs3,Omega(1:N,:),dt);
    Xsp1 = reshape(newXs',[],1);
    Xp1 = XonNp1Mat*[Xsp1; zeros(3,1)];
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('Free_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts','ee')
end