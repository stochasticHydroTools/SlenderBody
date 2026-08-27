% Projection method for spectral chain
function SpectralChain(seed,Nx,dt)
%if (0)
addpath(genpath('../../'))
nRuns = 1;
%seed=1;
wrongdrift=0;
clamp0=1;

L = 1;
kbT = 4.1e-3; % pN * um
lp = L;
K_b = lp*kbT;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
mu = 0.6;
delta = 1e-5;
%dt=2.5e-4;
implicit=1;
tf = 100;
nSt = (tf/dt);
saveEvery=max(1,floor(1e-2/dt+1e-10));
nSave = nSt/saveEvery;
rng(seed);
MaxIts = 10;
x0=[0;0;0];
tau0=[1;0;0];

[sX,wX,bX]=chebpts(Nx,[0 L],2);
s = L*(0.5:Nx-1)'/(Nx-1);%chebpts(Nx-1,[0 L],1);
DX = diffmat(Nx,[0 L],'chebkind2');
D = barymat(s,sX,bX)*DX;
D = kron(D,eye(3));

% Energy matrix
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sX,bX);
WTilde_1D = R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx;
WTilde_Inv = kron(WTilde_1D^(-1),eye(3));
WTilde_Nx = stackMatrix(WTilde_1D);
EMat = K_b*stackMatrix(DX^2)'*WTilde_Nx*...
    stackMatrix(DX^2);

nW = 1;
MobConst = -log(eps^2)/(8*pi*mu);
Mobility = @(x) LocalDragMob(x,DX,MobConst,WTilde_Inv); %MobRPY(x,DX,eps,mu);% Hessians are constant
H = HessMat(Nx,D,clamp0);
AllTanVecDots = zeros(nRuns,Nx-1);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,nSave);
Xpts=[];

% Gradient check
% dx = rand(3*Nx,1);
% gc = GradMat(x,D,clamp0)*dx;
% for iEps=1:10
%     [~,trudiff] = c(x+10^(-iEps)*dx,D,clamp0,x0,tau0);
%     ers(iEps) = norm(trudiff/10^(-iEps)-gc);
% end

for iRun=1:nRuns
% Initial state
x = x0'+sX.*tau0';
x = reshape(x',[],1);
nC = Nx-1;
if (clamp0)
    nC = Nx+4;
end
nX = length(x);

% Statistics
NumIts = zeros(nSave,1);
eedists = zeros(nSave,1);
nFail = 0;

% Unconstrained step
for iT=1:nSt
    M = Mobility(x);
    Mhalf = chol(M)';
    C = GradMat(x,D,clamp0);
    prefac = M*C'*(C*M*C')^(-1);
    GradU = EMat*x;
    
    divM = zeros(nX,1);
    divMC = zeros(nX,1);
    for iP=1:nW
        w1 = randn(nX,1);
        xr = x + delta*w1;
        Mu = Mobility(xr);
        divM = divM + 1/(nW*delta)*(Mu-M)*w1;
    
        w2 = randn(nC,1);
        xc = x + delta*prefac*w2;
        Mc = Mobility(xc);
        Cc = GradMat(xc,D,clamp0);
        divMC = divMC + 1/(nW*delta)*(Mc*Cc'-M*C')*w2;
    end
    divMtru = divM;
    divMctru = divMC;
    if (wrongdrift==1)
        divMtru=0*divMtru;
        divMctru=0*divMctru;
    elseif (wrongdrift==2)
        divMctru=0*divMctru;
    end

    % Take unconstrained step 
    W = randn(nX,1);
    ExPart = dt*kbT*(divMtru-divMctru)+sqrt(2*dt*kbT)*Mhalf*W;
    if (implicit)
        xtilde = (eye(3*Nx)+dt*M*EMat) \ (x+ExPart);
    else
        xtildeEx = x - dt*M*GradU + ExPart;
    end
    
    % Nonlinear system for the projection
    % x - xtilde + M*C(x)'*lambda = 0 
    % c(x) = 0
    % Newton solve
    xg = x;
    lam = zeros(nC,1);
    er=1;
    tol = 1e-10;
    Allresids = zeros(MaxIts,1);
    for it=1:MaxIts
        % Compute the gradient and Hessian at x
        C = GradMat(xg,D,clamp0);
        Htot = sum(H.*reshape(lam,1,1,nC),3);
        J = [eye(nX)+M*Htot M*C'; C zeros(nC)];
        [~,ceqc]=c(xg,D,clamp0,x0,tau0);
        resid = [(xg-xtilde) + M *C'*lam;ceqc ];
        er=norm(resid);
        Allresids(it)=er;
        if (er > tol)
            newsol = [xg;lam] - J \ resid;
            xg = newsol(1:nX);
            lam = newsol(nX+1:end);
        else
            break
        end
    end
    if (it>=MaxIts)
        nFail=nFail+1;
        % Matlab default
        Minv = M^(-1);
        fun = @(xvar) ProjectionObjective(xvar,xtilde,Minv);
        eqconstr = @(xvar) c(xvar,D,clamp0,x0,tau0);
        opts=optimoptions(@lsqnonlin,'OptimalityTolerance',1e-10,...
            'SpecifyObjectiveGradient',true,'Display','off');
        [xg,~,~,exitflag,~,~,~] = ...
            lsqnonlin(fun,x,[],[],[],[],[],[],eqconstr,opts);
    end
    x = xg;
    if (mod(iT,saveEvery)==0)
        index = floor(1e-10+iT/saveEvery)+1;
        NumIts(index)=it;
        eedists(index)=norm(x(1:3)-x(end-2:end));
        Xpts=[Xpts;reshape(x,3,[])'];
    end
end
AllEE(iRun,:)=eedists;
FailureRates(iRun) = nFail/nSt;
AllItCounts(iRun,:)=NumIts;
end
if (wrongdrift==0)
    save(strcat('ClmpProj_Lp',num2str(lp),...
    '_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
elseif (wrongdrift==1)
    save(strcat('NoDrSpectral_dt',num2str(dt),'_',num2str(seed),'.mat'))
elseif (wrongdrift==2)
    save(strcat('WrongDrSpectral_dt',num2str(dt),'_',num2str(seed),'.mat'))
end
end


function [val,J] = ProjectionObjective(x,xtilde,Minv)
    val = 1/2*(x-xtilde)'*Minv*(x-xtilde);
    if nargout > 1  
        J = Minv*(x-xtilde);
        J = J';
    end
end

function [cleq,cd] = c(x,D,clamp0,x0,tau0)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    tau = reshape(D*x,3,[])';
    cd = sum(tau.*tau,2)-1;
    if (clamp0)
        cd = [cd(2:end); x(1:3)-x0; tau(1,:)'-tau0];
    end
    cleq=[];
end

function C = GradMat(x,D,clamp0)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    Nx = length(x)/3;
    C = zeros(Nx-1,3*Nx);
    for j=1:Nx-1
        DDt = D(3*j-2:3*j,:)'*D(3*j-2:3*j,:);
        C(j,:)=(2*DDt*x)';
    end
    if (clamp0)
        Ct = zeros(Nx+4,3*Nx);
        Ct(1:Nx-2,:)=C(2:end,:);
        Ct(Nx-1:Nx+1,1:3)=eye(3);
        Ct(Nx+2:Nx+4,:)=D(1:3,:);
        C=Ct;
    end
end

function H = HessMat(Nx,D,clamp0)
    H = zeros(3*Nx,3*Nx,Nx-1);
    for j=1:Nx-1
        H(:,:,j)=2*D(3*j-2:3*j,:)'*D(3*j-2:3*j,:);
    end
    if (clamp0)
        Ht = zeros(3*Nx,3*Nx,Nx+4);
        Ht(:,:,1:Nx-2)=H(:,:,2:Nx-1);
        H = Ht;
    end
end

function M = MobRPY(x,DX,a,mu)
    % Starting with free space RPY kernel
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    Xs = DX*x;
    Xs = Xs./sqrt(sum(Xs.*Xs,2));
    M = zeros(3*Nx);
    for j=1:Nx
        M(3*j-2:3*j,3*j-2:3*j)=...
            log(a^(-2))/(8*pi*mu)*(eye(3)+Xs(j,:)'*Xs(j,:));
    end
end
