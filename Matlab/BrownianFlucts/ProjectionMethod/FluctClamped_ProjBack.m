function FluctClamped_ProjBack(seed,Nx,dt)
% Single fluctuating clamped filament
%seed=2;
%dt=1e-5;
%Nx = 8;
addpath(genpath('../../'))
%close all;
rng(seed);
gtype=2;
N = Nx - 1;
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
Tau0BC = [1;0;0];
TrkLoc = L/2;
XTrk=[TrkLoc;0;0];
Xs=repmat(Tau0BC',N,1);
BCs = [0;0;0;Tau0BC];

[s,w,b] = chebpts(N, [0 L], gtype);
[sX,wNp1,bX]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sX,s,b);
RNp1ToN = barymat(s,sX,bX);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sX,bX));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=repmat(eye(3),Nx,1);
XonNp1Mat = [(eye(3*Nx)-repmat(BMNp1,Nx,1))*stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
Xt = XonNp1Mat* [reshape(Xs',[],1);XTrk];
PtEvalMat = kron([barymat(0,sX,bX); barymat(0,sX,bX)*DNp1],eye(3));

% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, ~] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sX,bX);
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
BendMatHalf_Np1 = real(BendingEnergyMatrix_Np1^(1/2));

saveEvery=max(1,floor(1e-2/dt+1e-10));
ee=[];
MobConst = -log(eps^2)/(8*pi*mu);
Mobility = @(x) LocalDragMob(x,DNp1,MobConst,WTilde_Np1_Inverse);
Constr = @(alpha,x) c(alpha,x,XonNp1Mat,InvXonNp1Mat,PtEvalMat,BCs);
JacC = @(alpha,x) GradMat(alpha,x,XonNp1Mat,InvXonNp1Mat,PtEvalMat);
JacCNormals = @(alpha,x,NormalMat) GradMatNormals(alpha,x,NormalMat,XonNp1Mat,InvXonNp1Mat,PtEvalMat);
Hess = @(alpha,x) HessMat(alpha,x,XonNp1Mat,InvXonNp1Mat,PtEvalMat);
HessNormals = @(alpha,x,NormalMat) HessMatNormals(alpha,x,NormalMat,XonNp1Mat,InvXonNp1Mat,PtEvalMat);

% Gradient check (don't evaluate around 0)
[~,NormalsXt] = KNoNullSpace(Xs,XonNp1Mat);
dalpha = randn(2*Nx+1,1);
% Taylor expanding around alpha = 0 here
Grad0 = JacCNormals(zeros(2*Nx+1,1),Xt,NormalsXt);
gc = Grad0*dalpha;
% Hessian check
Hess0=HessNormals(zeros(2*Nx+1,1),Xt,NormalsXt);
Hess0chng = zeros(6,2*Nx+1);
for p=1:6
    Hess0chng(p,:) = dalpha'*Hess0(:,:,p);
end
%hc = HessC(zeros(3*Nx,1),Xt)
for iEps=1:10
    [~,trudiff] = Constr(10^(-iEps)*NormalsXt*dalpha,Xt);
    trugrad = JacCNormals(10^(-iEps)*dalpha,Xt,NormalsXt);
    graddiff = (trugrad-Grad0)/10^(-iEps);
    ers(iEps) = norm(trudiff/10^(-iEps)-gc);
    ersH(iEps) = norm(graddiff-Hess0chng);
end


%% Initialization 
stopcount=floor(tf/dt+1e-5);
nNewtonIts=zeros(stopcount,1);
MeanOmTurn = zeros(stopcount,1);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
Npl=100;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sX,bX);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;
NewtonTime=0;
nFail =0;
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
        ee=[ee;norm(PtsThisT(1,:)-PtsThisT(end,:))];
    end
    % Evolve system
    XsXTrk = reshape(InvXonNp1Mat*Xt,3,Nx)';
    XTrk = XsXTrk(end,:)';
    Xs3 = XsXTrk(1:N,:);
    MWsym = Mobility(Xt);
    MWsymHalf = chol(MWsym)';
    K = KonNp1(Xs3,XonNp1Mat,I);

    g = randn(3*Nx,1);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    g2 = randn(3*Nx,1);
    RandomVelBE = sqrt(kbT)*MWsym*BendMatHalf_Np1*g2;
    RandomVel = RandomVelBM + RandomVelBE;
    % Advance to midpoint
    % OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    % Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
    % Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    % Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XTrk];
    % MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    % 
    % %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    % deltaRFD = 1e-5;
    % WRFD=randn(3*Nx,1);
    % OmRFD =WRFD; % This is Delta X on the N+1 grid
    % TauPlus = rotateTau(Xs3,reshape(OmRFD(1:3*N),3,[])',deltaRFD);
    % XPlus = XonNp1Mat*[reshape(TauPlus',[],1);XTrk+deltaRFD*OmRFD(end-2:end)];
    % MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    % M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*KInv'*WRFD;
    % 
    % RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf_Np1*randn(3*Nx,1);
    %RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    % Solve unconstrained system
    Ktilde = K;
    MWsymTilde = MWsym;
    KWithImp = Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    alphaU = MobK*Ktilde'*(BendForceMat*Xt + MWsymTilde \ RandomVel);

    % Now add the RFD terms
    deltaRFD = 1e-6;
    C = JacC(zeros(3*Nx,1),Xt);
    wRFD = randn(3*Nx,1);
    Xplus = RotateAndIntegrate(deltaRFD*wRFD,Xt,XonNp1Mat,InvXonNp1Mat);
    Xsplus = reshape(InvXonNp1Mat*Xplus,3,Nx)';
    Cplus = JacC(zeros(3*Nx,1),Xplus);
    Kplus = KonNp1(Xsplus(1:N,:),XonNp1Mat,I);
    Mplus = Mobility(Xplus);
    Nplus = pinv(Kplus'*(Mplus \ Kplus));
    Ninv = Ktilde'*(MWsymTilde \ Ktilde);
    Nog = pinv(Ninv);
    N_RFD = 1/deltaRFD*(Nplus-Nog)*wRFD;
    NGC_RFD = 1/deltaRFD*(Nplus*Cplus'-Nog*C')*pinv(C*Nog*C')*(C*Nog*wRFD);

    alphaU_Uncons = dt*(alphaU + kbT*(N_RFD-NGC_RFD));
    alphaCor=alphaU_Uncons;

    [~,NormalsXt] = KNoNullSpace(Xs3,XonNp1Mat);
    alphaUCCoords = NormalsXt'*alphaU_Uncons;

    % Solve minimization problem
    % Explicitly take out the null space here? 
    % Nonlinear system for the projection
    % alpha - alphaU_Uncons + N*C(alpha)'*lambda = 0 
    % c(alpha) = 0
    % Newton solve
    nC = size(PtEvalMat,1);
    ag = 0*alphaUCCoords; % The actual rates
    lam = zeros(nC,1);
    MaxIts=20;
    tol = 1e-8;
    Allresids = zeros(MaxIts,1);
    Ninv = NormalsXt'*Ninv*NormalsXt;
    for it=1:MaxIts
        % Compute the gradient and Hessian at x
        C = JacCNormals(ag,Xt,NormalsXt);
        H = HessNormals(ag,Xt,NormalsXt);
        Htot = sum(H.*reshape(lam,1,1,nC),3);
        J = [Ninv+Htot C'; C zeros(nC)];
        [~,ceqc]=Constr(NormalsXt*ag,Xt);
        resid = [Ninv*(ag-alphaUCCoords) + C'*lam;ceqc ];
        er=norm(resid);
        Allresids(it)=er;
        if (er > tol)
            newsol = [ag;lam] - pinv(J)*resid;
            ag = newsol(1:2*Nx+1);
            lam = newsol(2*Nx+2:end);
        else
            alphaCor=NormalsXt*ag;
            break
        end
    end
    if (it>=MaxIts)
        nFail=nFail+1;
        % Matlab default
        fun = @(xvar) ProjectionObjective(xvar,alphaUCCoords,Ninv);
        eqconstr = @(xvar) Constr(NormalsXt*xvar,Xt);
        opts=optimoptions(@lsqnonlin,'OptimalityTolerance',1e-10,...
            'SpecifyObjectiveGradient',true,'Display','off');
        [ag2,resnorm,~,exitflag,~,~,~] = ...
            lsqnonlin(fun,0*alphaUCCoords,[],[],[],[],[],[],eqconstr,opts);
        alphaCor=NormalsXt*ag2;
    end
    Xt = RotateAndIntegrate(alphaCor,Xt,XonNp1Mat,InvXonNp1Mat);
end
Totaltime=toc(tStart);
save(strcat('ClmpHybridRFD_Lp',num2str(lp),...
    '_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
end

function [val,J] = ProjectionObjective(x,xtilde,Minv)
    val = 1/2*(x-xtilde)'*Minv*(x-xtilde);
    if nargout > 1  
        J = Minv*(x-xtilde);
        J = J';
    end
end

function [cleq,cd] = c(alpha,x,XFromTau,XFromTauInv,PtEvalMat,BCs)
    xplus = RotateAndIntegrate(alpha,x,XFromTau,XFromTauInv);
    cd = PtEvalMat*xplus-BCs;
    cleq=[];
end

function C = GradMat(alpha,x,XFromTau,XFromTauInv,PtEvalMat)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    if (size(alpha,2)==1)
        alpha=reshape(alpha,3,[])';
    end
    Nx = length(x)/3;
    tauMP = reshape(XFromTauInv*x,3,[])';
    % Gradient of rotate wrt tau
    GradRotateAndIntegrate = eye(3*Nx);
    GradRotateAndIntegrate(1:3*Nx-3,1:3*Nx-3) = DrotateTau(tauMP(1:end-1,:),alpha(1:end-1,:));
    % Multiply by of C wrt x
    C = PtEvalMat*XFromTau*GradRotateAndIntegrate;
end

function C = GradMatNormals(alpha,x,NormalMat,XFromTau,XFromTauInv,PtEvalMat)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    Nx = length(x)/3;
    tauMP = reshape(XFromTauInv*x,3,[])';
    alpha = NormalMat*alpha;
    alpha = reshape(alpha,3,[])';
    % Gradient of rotate wrt tau
    GradRotateAndIntegrate = eye(3*Nx);
    GradRotateAndIntegrate(1:3*Nx-3,1:3*Nx-3) = DrotateTau(tauMP(1:end-1,:),alpha(1:end-1,:));
    % Multiply by of C wrt x
    C = PtEvalMat*XFromTau*GradRotateAndIntegrate*NormalMat;
end

function H = HessMat(alpha,x,XFromTau,XFromTauInv,PtEvalMat)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    if (size(alpha,2)==1)
        alpha=reshape(alpha,3,[])';
    end
    Nx = length(x)/3;
    nC = size(PtEvalMat,1);
    H = zeros(3*Nx,3*Nx,nC);
    tauMP = reshape(XFromTauInv*x,3,[])';
    % Gradient of rotate wrt tau
    HRotTau=zeros(3*Nx,3*Nx,3*Nx);
    HRotTau(1:3*Nx-3,1:3*Nx-3,1:3*Nx-3) = RotateHessian(tauMP(1:end-1,:),alpha(1:end-1,:)); %[(i,j,k)=d^2 taubar_k / dalpha_i dalpha_j for sc = 1..3 the third dim
    HRotTau(isnan(HRotTau))=0;
    for iP=1:3*Nx
        for jP = 1:3*Nx
            H(iP,jP,:) = PtEvalMat*XFromTau*reshape(HRotTau(iP,jP,:),[],1); %[d^2 c / dalpha_i dalpha_j] for sc = 1..3 the third dim
        end
    end
end

function H = HessMatNormals(alpha,x,NormalMat,XFromTau,XFromTauInv,PtEvalMat)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    Nx = length(x)/3;
    nC = size(PtEvalMat,1);
    tauMP = reshape(XFromTauInv*x,3,[])';
    alpha = NormalMat*alpha;
    alpha = reshape(alpha,3,[])';
    % Gradient of rotate wrt alphas
    HRotTau=zeros(3*Nx,3*Nx,3*Nx);
    HonAlphas=zeros(2*Nx+1,2*Nx+1,3*Nx);
    HRotTau(1:3*Nx-3,1:3*Nx-3,1:3*Nx-3) = RotateHessian(tauMP(1:end-1,:),alpha(1:end-1,:)); %[(i,j,k)=d^2 taubar_k / dalpha_i dalpha_j for sc = 1..3 the third dim
    HRotTau(isnan(HRotTau))=0; % to fix later
    for iP=1:3*Nx
        HonAlphas(:,:,iP)=NormalMat'*HRotTau(:,:,iP)*NormalMat;
    end
    H = zeros(2*Nx+1,2*Nx+1,nC);
    for iP=1:2*Nx+1
        for jP = 1:2*Nx+1
            H(iP,jP,:) = PtEvalMat*XFromTau*reshape(HonAlphas(iP,jP,:),[],1); %[d^2 c / dalpha_i dalpha_j] for sc = 1..3 the third dim
        end
    end
end