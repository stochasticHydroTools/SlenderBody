% Parameters for chain and mobility
function WormLikeChain(seed,Nlinks,dt)
addpath(genpath('../../'))
%Nlinks=10;
%seed=1;
%dt=1e-2;
nRuns=1;
ds = 1/Nlinks;
L = ds*Nlinks;
kbT = 4.1e-3; % pN * um
lp = 1*L;
K_b = lp*kbT;
a = 1e-2;
mu = 1;
delta = 1e-5;
implicit=1;
tf = 25;
nSt = (tf/dt);
saveEvery = max(1e-2/dt,1);
nSave = floor(1e-10+nSt/saveEvery);
rng(seed);
MaxIts = 10;

nW = 1;
Mobility = @(x) MobRPY(x,a,mu,ds);
% Hessians are constant
H = HessMat(Nlinks);
AllTanVecDots = zeros(nRuns,Nlinks);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,nSave);
Xpts=[];

for iRun=1:nRuns
% Initial state
tau = [ones(Nlinks,1) zeros(Nlinks,2)];
x = reshape(([0 0 0; cumsum(ds*tau)])',[],1);
nC = Nlinks;
nX = length(x);

% The energy matrix
EMat = WLCEnergyMatrix(K_b,Nlinks,ds);

% Statistics
TanVecDots = zeros(Nlinks,1);
nSamplesDs = zeros(Nlinks,1);
NumIts = zeros(nSave,1);
eedists = zeros(nSave,1);
nFail = 0;
% For when Newton fails
opts=optimoptions(@fsolve,'OptimalityTolerance',1e-10,...
    'SpecifyObjectiveGradient',true,'Display','off');

% Unconstrained step
for iT=1:nSt
    M = Mobility(x);
    Mhalf = chol(M)';
    
    divM = zeros(nX,1);
    for iP=1:nW
        w1 = randn(nX,1);
        xr = x + delta*w1;
        Mu = Mobility(xr);
        divM = divM + 1/(nW*delta)*(Mu-M)*w1;
    end

    % Take unconstrained step 
    W = randn(nX,1);
    ExPart = dt*kbT*divM+sqrt(2*dt*kbT)*Mhalf*W;
    if (implicit)
        xtilde = (eye(3*(Nlinks+1))+dt*M*EMat) \ (x+ExPart);
    else
        xtildeEx = x - dt*M*GradU + ExPart;
    end

    % Half step
    xHalf = x + sqrt(kbT*dt/2)*Mhalf*W;
    Mhalf = Mobility(xHalf);
    Chalf = GradMat(xHalf);
    
    % Nonlinear system for the projection
    % x - xtilde + M*C(x)'*lambda = 0 
    % c(x) = 0
    % Newton solve
    xg = x;
    lam = zeros(nC,1);
    tol = 1e-10;
    Allresids = zeros(MaxIts,1);
    for it=1:MaxIts
        % Compute the gradient and Hessian at x
        C = GradMat(xg);
        J = [eye(nX) -Mhalf*Chalf'; C zeros(nC)];
        ceqc=c(xg,ds);
        resid = [(xg-xtilde) - Mhalf *Chalf'*lam;ceqc ];
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
        NLFcn = @(x) NonLinSys(x,xtilde,Mhalf,Chalf,ds);
        [xnlsolve,fval,exitflag] = fsolve(NLFcn,[x;zeros(nC,1)],opts);
        xg = xnlsolve(1:nX);
    end
    x = xg;
    if (mod(iT,saveEvery)==0)
        index = floor(1e-6+iT/saveEvery);
        NumIts(index)=it;
        eedists(index)=norm(x(1:3)-x(end-2:end));
        Xpts=[Xpts;reshape(x,3,[])'];
        if (iT/nSt>1/2)
            % Tangent vector dot products
            x3 = reshape(x,3,[])';
            tau = (x3(2:end,:)-x3(1:end-1,:))/ds;
            for iLink=1:Nlinks
                for jLink=iLink:Nlinks
                    index = jLink-iLink+1;
                    nSamplesDs(index)=nSamplesDs(index)+1;
                    TanVecDots(index)=TanVecDots(index)+dot(tau(iLink,:),tau(jLink,:));
                end
            end
        end
    end
end
AllEE(iRun,:)=eedists;
FailureRates(iRun) = nFail/nSt;
AllTanVecDots(iRun,:) = TanVecDots./nSamplesDs;
AllItCounts(iRun,:)=NumIts;
end
save(strcat('WLC',num2str(Nlinks),'_dt',num2str(dt),'_',num2str(seed),'.mat'))
end

function EnergyMat = WLCEnergyMatrix(K_b,Nlinks,ds)
    N = Nlinks+1;
    EnergyMat = diff(eye(N+4),4);
    EnergyMat(:,[1 2 N+3 N+4]) = [];
    EnergyMat(1,1:3) = -1*[-1 2 -1];
    EnergyMat(2,1:4) = -1*[2 -5 4 -1];
    EnergyMat(end,end:-1:end-2) = -1*[-1 2 -1];
    EnergyMat(end-1,end:-1:end-3) = -1*[2 -5 4 -1];
    EnergyMat = K_b*kron(sparse(EnergyMat),eye(3))/ds^3;
end

function [val,J] = NonLinSys(xin,xtilde,Mhalf,Chalf,ds)
    nX = length(xtilde);
    x = xin(1:nX);
    lam = xin(nX+1:end);
    val = [(x-xtilde) - Mhalf *Chalf'*lam; c(x,ds)];
    C = GradMat(x);
    J = [eye(length(x)) -Mhalf*Chalf'; C zeros(length(lam))];
end

function cd = c(x,ds)
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    cd = zeros(Nx-1,1);
    for j=1:Nx-1
        cd(j) = norm(x(j+1,:)-x(j,:)).^2-ds^2;
    end
end

function C = GradMat(x)
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    C = zeros(Nx-1,3*Nx);
    for j=1:Nx-1
        C(j,3*j-2:3*j)=2*(x(j,:)-x(j+1,:));
        C(j,3*j+1:3*j+3)=2*(x(j+1,:)-x(j,:));
    end
end

function H = HessMat(Nlinks)
    H = zeros(3*Nlinks+3,3*Nlinks+3,Nlinks);
    for j=1:Nlinks
        H(3*j-2:3*j,3*j-2:3*j,j)=2*eye(3);
        H(3*j-2:3*j,3*j+1:3*j+3,j)=-2*eye(3);
        H(3*j+1:3*j+3,3*j+1:3*j+3,j)=2*eye(3);
        H(3*j+1:3*j+3,3*j-2:3*j,j)=-2*eye(3);
    end
end

function M = MobRPY(x,a,mu,ds)
    % Starting with free space RPY kernel
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    tauavg = zeros(Nx,3);
    for j=2:Nx-1
        tauavg(j,:)=(x(j+1,:)-x(j-1,:));
    end
    tauavg(1,:)=x(2,:)-x(1,:);
    tauavg(Nx,:)=x(Nx,:)-x(end-1,:);
    tauavg = tauavg./sqrt(sum(tauavg.*tauavg,2));
    M = zeros(3*Nx);
    for j=1:Nx
        M(3*j-2:3*j,3*j-2:3*j)=...
            log(a^(-2))/(8*pi*mu*ds)*(eye(3)+tauavg(j,:)'*tauavg(j,:));
    end
    % M = 1/(6*pi*mu*a)*eye(3*Nx);
    % for i=1:Nx
    %     for j=i:Nx
    %         rvec = x(i,:)-x(j,:);
    %         mij =  RPYTot(rvec,a,mu);
    %         M(3*i-2:3*i,3*j-2:3*j) =mij;
    %         M(3*j-2:3*j,3*i-2:3*i) =mij;
    %     end
    % end
   % M = RPYMatrixWithWall(x,mu, a);
end
