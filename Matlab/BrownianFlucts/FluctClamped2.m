function FluctClamped2(seed,ForceRt,N,dt)
%seed='1';
%N='12';
%dt='1e-3';
%ForceRt='5';
% Single fluctuating clamped filament
addpath(genpath('../'))
%close all;
rng(str2num(seed));
nFib = 1;
L = 1;   % microns
N = str2num(N);
ForceRt=str2num(ForceRt);
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 3*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie = 0;
dt = str2double(dt);
tf = 40;
Tau0BC = [0;1;0];
TrkLoc = 0;
XTrk=[0;TrkLoc;0];
%q=3; 
X_s=repmat(Tau0BC',N,1);
[s,w,b] = chebpts(N, [0 L], 2);
%X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
InitializationNoTwist;
saveEvery=floor(1e-1/dt+1e-10);
ee=[];
MeanOmTurn=[];
MobConst = -log(eps^2)/(8*pi*mu);
ConsMat = [stackMatrix(barymat(0,sNp1,bNp1)); ...
    stackMatrix([barymat(0,s,b) 0])*InvXonNp1Mat;...
    stackMatrix([barymat(L,s,b) 0])*InvXonNp1Mat];
Constr=[0;0;0;Tau0BC;Tau0BC];
Correct = 0;
nConstr=length(Constr);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
nNewtonIts=zeros(stopcount,1);
FinalNorms = zeros(stopcount,1);
Xpts=[];
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;
NewtonTime=0;

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
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt))
            view(2)
            ylim([-1 1])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
    end
    % Evolve system
    XsXTrk = reshape(InvXonNp1Mat*Xt,3,Nx)';
    XTrk = XsXTrk(end,:)';
    Xs3 = XsXTrk(1:N,:);
    MWsym = LocalDragMob(Xt,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = real(MWsym^(1/2));
    % Obtain Brownian velocity
    K = KonNp1(Xs3,XonNp1Mat,I);
    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    % Advance to midpoint
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XTrk];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    % deltaRFD = 1e-5;
    % WRFD = randn(3*Nx,1); % This is Delta X on the N+1 grid
    % OmegaPlus = cross(Xs3,RNp1ToN*DNp1*reshape(WRFD,3,[])');
    % TauPlus = rotateTau(Xs3,deltaRFD*OmegaPlus(1:N,:),1);
    % XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XTrk];
    % MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    % M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*WRFD;
    RandomVelBE = sqrt(kbT)*...
         MWsymTilde*BendMatHalf_Np1*randn(3*Nx,1);
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    % B = Kaug-impcoeff*dt*MWsymTilde*BendForceMat*Kaug;
    U0 = zeros(3*Nx,1);
    U0(1:3:end)=0;
    Fext = zeros(3*Nx,1);
    Fext(end-1) = ForceRt^2*Eb;
    %RHS = Kaug'*(BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Form psuedo-inverse manually
    %maxRank = 2*N+3;
    %if (clamp)
    %    maxRank = 2*N-2;
    %end
    %alphaU = ManualPinv(Kaug'*(MWsymTilde \ B),maxRank)*RHS;
    KWithImp = Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
    % tic
    % MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    % MobC = ConsMat'*pinv(ConsMat*Ktilde*MobK*Ktilde'*ConsMat',1e-8)*ConsMat;
    % alphaU = (MobK*Ktilde' - ...
    %     MobK*Ktilde'*MobC*Ktilde*MobK*Ktilde')*...
    %     (BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % toc
    % tic
    Mat2 = [-MWsymTilde KWithImp zeros(3*Nx,nConstr); ...
        Ktilde' zeros(3*Nx) Ktilde'*ConsMat'; ...
        zeros(nConstr,3*Nx) ConsMat*Ktilde zeros(nConstr)];
    RHS2 = [MWsymTilde*(BendForceMat*Xt + Fext) + RandomVel+U0; ...
        zeros(3*Nx+nConstr,1)];
    Sol2 = pinv(Mat2)*RHS2;
    Lambda = Sol2(1:3*Nx);
    alphaU = Sol2(3*Nx+1:6*Nx);
    Gamma = Sol2(6*Nx+1:end);
    % toc
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Xsp1 = reshape(newXs',[],1);
    XTrk_p1 = XTrk+dt*alphaU(end-2:end);
    Xp1 = XonNp1Mat*[Xsp1;XTrk_p1];
    % Solve for minimum Omega s.t. you get back on constraint
    ConstrEr = norm(ConsMat*Xp1-Constr);
    if (Correct && ConstrEr>0)
        % Solve for the "closest" X that satisfies constraints
        ogXp1 = Xp1;
        [Xp1,nNewtonIts(count+1),FinalNorms(count+1)] = ...
            SolveOptimProblem(Xp1,XonNp1Mat,WTilde_Np1,...
            InvXonNp1Mat,ConsMat,Constr);
        SqEr = (Xp1-ogXp1)'*WTilde_Np1*(Xp1-ogXp1);
        MeanOmTurn=[MeanOmTurn;sqrt(SqEr)];
    end
    Xt=Xp1;
    ee=[ee;norm(Xt(1:3)-Xt(end-2:end))];
end
mean(MeanOmTurn)
Totaltime=toc(tStart);
save(strcat('T2Lp',num2str(lp),'_Force',num2str(ForceRt),...
    '_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
end