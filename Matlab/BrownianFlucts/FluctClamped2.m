function Xpts = FluctClamped2(seed,N,dt)
%seed='1';
%N='12';
%dt='1e-2';
% Single fluctuating clamped filament
addpath(genpath('../'))
%close all;
rng(str2num(seed));
nFib = 1;
L = 2;   % microns
N = str2num(N);
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 0;%4.1e-3;
lp = L;
Eb = 0.01;%lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie = 1;
dt = str2double(dt);
tf = 2.5;
Tau0BC = [0;1;0];
TrkLoc = L/2;
XTrk=[0;0.1;0];
clamp=0;
q=3; 
%X_s=repmat(Tau0BC',N,1);
[s,w,b] = chebpts(N, [0 L], 1);
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
InitializationNoTwist;
saveEvery=floor(1e-1/dt+1e-10);
ee=[];
MeanOmTurn=[];
nNewtonIts=[];
MidPointSolve = 1;
MobConst = -log(eps^2)/(8*pi*mu);
ConsMat = [stackMatrix(barymat(0,sNp1,bNp1)); ...
    stackMatrix([barymat(0,s,b) 0])*InvXonNp1Mat];
Constr=[0;0;0;Tau0BC];
Correct=1;
nConstr = length(Constr);
opts = optimoptions(@fsolve,'MaxFunctionEvaluations',1e5,...
    'MaxIterations',1e3);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
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
        Xpts=[Xpts;RplNp1*PtsThisT];
    end
    % Evolve system
    XsXTrk = reshape(InvXonNp1Mat*Xt,3,Nx)';
    XTrk = XsXTrk(end,:)';
    Xs3 = XsXTrk(1:N,:);
    MWsym = LocalDragMob(Xt,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = real(MWsym^(1/2));
    % Obtain Brownian velocity
    K = KonNp1(Xs3,XonNp1Mat,I);
    if (clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
    end
    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    % Advance to midpoint
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    if (clamp)
        OmegaTilde = ProjectClamp*[reshape(OmegaTilde',[],1);zeros(3,1)];
        OmegaTilde = reshape(OmegaTilde,3,[])';
    end
    Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XTrk];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    if (clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*Ktilde;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClampTilde = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = Ktilde*ProjectClampTilde;
    else
        Kaug = Ktilde;
    end
    deltaRFD = 1e-5;
    WRFD = randn(3*Nx,1); % This is Delta X on the N+1 grid
    OmegaPlus = cross(Xs3,RNp1ToN*DNp1*reshape(WRFD,3,[])');
    if (clamp)
        OmegaPlus = ProjectClamp*[reshape(OmegaPlus',[],1);zeros(3,1)];
        OmegaPlus = reshape(OmegaPlus,3,[])';
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
    U0(1:3:end)=Xt(2:3:end);
    Fext = zeros(3*Nx,1);
    %Fext(end-1) = 100*Eb;
    %RHS = Kaug'*(BendForceMat*Xt+ Fext + MWsymTilde \ (RandomVel + U0));
    % Form psuedo-inverse manually
    %maxRank = 2*N+3;
    %if (clamp)
    %    maxRank = 2*N-2;
    %end
    %alphaU = ManualPinv(Kaug'*(MWsymTilde \ B),maxRank)*RHS;
    Mat2 = [-MWsymTilde Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde; ...
        Ktilde' zeros(3*Nx)];
    RHS2 = [MWsymTilde*(BendForceMat*Xt + Fext) + RandomVel+U0; zeros(3*Nx,1)];
    Sol2 = pinv(Mat2)*RHS2;
    Lambda = Sol2(1:3*Nx);
    alphaU = Sol2(3*Nx+1:6*Nx);
    %Gamma = Sol2(6*Nx+1:end);
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Xsp1 = reshape(newXs',[],1);
    XTrk_p1 = XTrk+dt*alphaU(end-2:end);
    Xp1 = XonNp1Mat*[Xsp1;XTrk_p1];
    % Solve for minimum Omega s.t. you get back on constraint
    if (Correct)
        x0 = Sol2;%zeros(6*Nx,1);
        [NonLinAns,nIts] = SolveNonLinEqns(x0,dt,Xt,XonNp1Mat,InvXonNp1Mat,...
            MWsymTilde,BendForceMat,Ktilde,ConsMat,Constr,U0);
        ogXp1 = Xp1;
        NewOm = NonLinAns(3*Nx+1:6*Nx);
        Xp1 = ComputeX(NewOm*dt,Xt,XonNp1Mat,InvXonNp1Mat);
        SqEr = (Xp1-ogXp1)'*WTilde_Np1*(Xp1-ogXp1);
        NewtonTime=NewtonTime+toc;
    end
    Xt=Xp1;
    ee=[ee;norm(Xt(1:3)-Xt(end-2:end))];
    if (Correct)
    MeanOmTurn=[MeanOmTurn;sqrt(SqEr)];
    nNewtonIts = [nNewtonIts;nIts];
    end
end
mean(MeanOmTurn)
Totaltime=toc(tStart);
%save(strcat('MatlabSimsForClamping/T1MBE_Lp',...
%      num2str(lp),'_N',num2str(N),'_Dt',num2str(dt),...
%      '_Seed',num2str(seed),'.mat'))
end

function Xnew = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat)
    XsXMP = InvXonNp1Mat*Xin;
    newXs = rotateTau(reshape(XsXMP(1:end-3),3,[])',...
        reshape(x(1:end-3),3,[])',1);
    Xsp1 = reshape(newXs',[],1);
    NewMP = XsXMP(end-2:end)+x(end-2:end);
    Xnew = XonNp1Mat*[Xsp1;NewMP];
end

function JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat)
    JGrad = zeros(length(Xin));
    XsXMP = InvXonNp1Mat*Xin;
    JGrad(1:end-3,1:end-3) = DrotateTau(reshape(XsXMP(1:end-3),3,[])',...
        reshape(x(1:end-3),3,[])');
    for d=0:2
        JGrad(end-d,end-d)=1;
    end
    JGrad = XonNp1Mat*JGrad;
end

function MotionBasis = calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat)
    Nx = length(Xin)/3;
    Tau = InvXonNp1Mat*Xin;
    Tau = reshape(Tau(1:3*(Nx-1)),3,[])';
    Omega = reshape(OmegaV(1:3*(Nx-1)),3,[])';
    OmegaPerp = cross(Tau,Omega);
    MotionBasis=zeros(3*Nx,2*Nx+1);
    for j=1:Nx-1
        alpha=zeros(3*Nx,1);
        alpha((j-1)*3+1:3*j) = Omega(j,:);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,j)=(NewX-Xin)/dt;
    end
    for j=1:Nx-1
        alpha=zeros(3*Nx,1);
        alpha((j-1)*3+1:3*j) = OmegaPerp(j,:);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,j+Nx-1)=(NewX-Xin)/dt;
    end
    for d=1:3
        alpha=zeros(3*Nx,1);
        alpha(3*(Nx-1)+d)=OmegaV(3*(Nx-1)+d);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,2*(Nx-1)+d)=(NewX-Xin)/dt;
    end
end

function GradMotDotLam = gradMotionBasis(OmegaV,Xin,dt,JGrad,Lambda,...
        XonNp1Mat,InvXonNp1Mat)
    Nx = length(Xin)/3;
    Tau = InvXonNp1Mat*Xin;
    Tau = reshape(Tau(1:3*(Nx-1)),3,[])';
    Omega = reshape(OmegaV(1:3*(Nx-1)),3,[])';
    OmegaPerp = cross(Tau,Omega);
    OmegaVPerp = OmegaV;
    OmegaVPerp(1:3*(Nx-1))=reshape(OmegaPerp',[],1);
    JGradPerp = GradX(OmegaVPerp*dt,Xin,XonNp1Mat,InvXonNp1Mat);
    GradMotDotLam = zeros(2*Nx+1,3*Nx);
    for p = 1:Nx-1
        GradMotDotLam(p,(p-1)*3+1:3*p)=JGrad(:,(p-1)*3+1:3*p)'*Lambda;
        GradMotDotLam(p+Nx-1,(p-1)*3+1:3*p)=...
            (JGradPerp(:,(p-1)*3+1:3*p)*CPMatrix(Tau(p,:)))'*Lambda;
    end
    for d=1:3
        GradMotDotLam(2*(Nx-1)+d,(Nx-1)*3+d)=JGrad(:,(Nx-1)*3+d)'*Lambda;
    end
end

function [x,nIts] = SolveNonLinEqns(x,dt,Xin,XonNp1Mat,InvXonNp1Mat,M,...
    BendForceMat,K,ConsMat,Constr,U0)
    [nConstr,NxThr]=size(ConsMat);
    Nx = NxThr/3;
    SolAll=1;
    nIts=0;
    StepSize=1;
    x0=x;
    % Compute basis of motions
    Lambda = x(1:3*Nx);
    OmegaV = x(3*Nx+1:6*Nx);
    MotionBasis =  calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat);    
    %Gamma = x(6*Nx+1:6*Nx+nConstr);
    NewX = ComputeX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
    Eq1 = (NewX-Xin)/dt - M*(BendForceMat*NewX+Lambda) - U0;
    Eq2 = MotionBasis'*Lambda;%+K'*ConsMat'*Gamma;
    %Eq3 = ConsMat*NewX-Constr;
    SolAll=[Eq1;Eq2];%Eq3];
    while (norm(SolAll)>1e-6 && nIts < 100 && StepSize > 1e-6)
        % Compute gradient
        JGrad = GradX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        dXdOm = JGrad*dt;  
        GradMotDotLam = gradMotionBasis(OmegaV,Xin,dt,JGrad,Lambda,...
            XonNp1Mat,InvXonNp1Mat);
        %GradMat = [-M (eye(3*Nx)/dt-M*BendForceMat)*dXdOm zeros(3*Nx,nConstr); ...
        %    K' zeros(3*Nx) K'*ConsMat'; ...
        %    zeros(nConstr,3*Nx) ConsMat*dXdOm zeros(nConstr)];
        GradMat = [-M (eye(3*Nx)/dt-M*BendForceMat)*dXdOm; ...
            MotionBasis' GradMotDotLam];
        % % Gradient check
        dx = randn(6*Nx,1);
        ExpDiff = GradMat*dx;
        for dp=1:10
            newx = x+10^(-dp)*dx;
            Lambdac = newx(1:3*Nx);
            OmegaVc = newx(3*Nx+1:6*Nx);
            NewXc = ComputeX(OmegaVc*dt,Xin,XonNp1Mat,InvXonNp1Mat);
            NewMotionBasis = calcMotionBasis(OmegaVc,dt,Xin,XonNp1Mat,InvXonNp1Mat);
            Eq1 = (NewXc-Xin)/dt - M*(BendForceMat*NewXc+Lambdac) - U0;
            Eq2 = NewMotionBasis'*Lambdac;
            SolAlld(:,dp)=([Eq1;Eq2]-SolAll)/(10^(-dp));
        end
        % Newton solve
        NewDir = -pinv(GradMat)*SolAll;
        % Line search
        StepSize=1;
        Normnew = inf;
        while (Normnew>norm(SolAll))
            xtry = x+StepSize*NewDir;
            Lambda = xtry(1:3*Nx);
            OmegaV = xtry(3*Nx+1:6*Nx);
            MotionBasis = calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat); 
            %Gamma = xtry(6*Nx+1:6*Nx+nConstr);
            NewX = ComputeX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
            Eq1 = (NewX-Xin)/dt - M*(BendForceMat*NewX+Lambda) - U0;
            Eq2 = MotionBasis'*Lambda;%+K'*ConsMat'*Gamma;
            %Eq3 = ConsMat*NewX-Constr;
            %Normnew=norm([Eq1;Eq2;Eq3]);
            Normnew=norm([Eq1;Eq2]);
            StepSize=StepSize/2;
        end
        SolAll=[Eq1;Eq2];
        nIts=nIts+1;
        x=xtry;
    end
    %max(abs(x(1:3*Nx)-x0(1:3*Nx)))
    %nIts
    %norm(SolAll)
    if (norm(SolAll) > 1e-6)
        keyboard
    end
end

