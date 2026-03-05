%function FluctClamped_ProjBack(seed,ForceRt,N,dt)
% Single fluctuating clamped filament
ForceRt=0;
seed=1;
N=16;
dt=1e-3;
gtype=1;
addpath(genpath('../'))
%close all;
rng(seed);
Nx = N+1;
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = 8.2e-3;%lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie = 0;
tf = 1;
Tau0BC = [0;1;0];
if (gtype==2)
    TrkLoc = 0;
else
    TrkLoc=L/2;
end
XTrk=[0;TrkLoc;0];
X_s=repmat(Tau0BC',N,1);
Xst = reshape(X_s',[],1);

[s,w,b] = chebpts(N, [0 L], gtype);
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
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
    stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
Xt = XonNp1Mat* [Xst;XTrk];


% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, b2Np2] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
BendMatHalf_Np1 = real(BendingEnergyMatrix_Np1^(1/2));


saveEvery=max(1,floor(1e-2/dt+1e-10));
ee=[];
MobConst = -log(eps^2)/(8*pi*mu);
ConsMat = [stackMatrix(barymat(0,sNp1,bNp1)); ...
    stackMatrix([barymat(0,s,b) 0])*InvXonNp1Mat;...
    stackMatrix([barymat(L,s,b) 0])*InvXonNp1Mat];
Constr=[0;0;0;Tau0BC;Tau0BC];
if (gtype==2) % if enforcing extensibility at same place, one of the constraints is redundant
ConsMat([5 8],:)=[];
Constr([5 8])=[];
end
Correct = gtype~=2;
nConstr=length(Constr);


%% Initialization 
stopcount=floor(tf/dt+1e-5);
nNewtonIts=zeros(stopcount,1);
MeanOmTurn = zeros(stopcount,1);
ConstrErs = zeros(stopcount,1);
Xpts=[];
Npl=100;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
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
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
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
    U0 = zeros(3*Nx,1);
    U0(1:3:end)=0;
    Fext = zeros(3*Nx,1);
    Fext(end-1) = ForceRt^2*Eb;
    KWithImp = Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    MobC = ConsMat'*pinv(ConsMat*Ktilde*MobK*Ktilde'*ConsMat')*ConsMat;
    alphaU = (MobK*Ktilde' - ...
        MobK*Ktilde'*MobC*Ktilde*MobK*Ktilde')*...
        (BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Mat2 = [-MWsymTilde KWithImp zeros(3*Nx,nConstr); ...
    %     Ktilde' zeros(3*Nx) Ktilde'*ConsMat'; ...
    %     zeros(nConstr,3*Nx) ConsMat*Ktilde zeros(nConstr)];
    % RHS2 = [MWsymTilde*(BendForceMat*Xt + Fext) + RandomVel+U0; ...
    %     zeros(3*Nx+nConstr,1)];
    % Sol2 = pinv(Mat2)*RHS2;
    % Lambda = Sol2(1:3*Nx);
    % alphaU1 = Sol2(3*Nx+1:6*Nx);
    % Gamma = Sol2(6*Nx+1:end);
    % if (max(abs(alphaU1-alphaU))>0.5)
    %     keyboard
    % end
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Xsp1 = reshape(newXs',[],1);
    XTrk_p1 = XTrk+dt*alphaU(end-2:end);
    Xp1 = XonNp1Mat*[Xsp1;XTrk_p1];
    % Solve for minimum Omega s.t. you get back on constraint
    tNewt = tic;
    ConstrErs(count+1) = norm(ConsMat*Xp1-Constr);
    if (Correct && ConstrErs(count+1)>0)
        % Solve for the "closest" X that satisfies constraints
        ogXp1 = Xp1;
        [Xp1tilde,nNewtonIts(count+1)] = ...
           SolveOptimProblem(Xp1,XonNp1Mat,WTilde_Np1,...
           InvXonNp1Mat,ConsMat,Constr,Xt);
        Xp1 = ogXp1-repmat(ogXp1(1:3),Nx,1);
        sqDiff = (Xp1-ogXp1)'*WTilde_Np1*(Xp1-ogXp1);
        sqDiff1 = (Xp1tilde-ogXp1)'*WTilde_Np1*(Xp1tilde-ogXp1);
        MeanOmTurn(count+1)=sqrt(sqDiff);
    end
    NewtonTime = NewtonTime+toc(tNewt);
    Xt=Xp1;
end
Totaltime=toc(tStart);
%save(strcat('Type2_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end