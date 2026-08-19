% Parameters for chain and mobility
function WormLikeChain(seed,dt,wrongdrift)
addpath(genpath('../../'))
nRuns = 1;
%seed=1;
%wrongdrift=1;

ds = 0.1;
Nlinks = 10;
L = ds*Nlinks;
kbT = 4.1e-3; % pN * um
lp = 1*L;
K_b = lp*kbT;
a = 1e-2;
mu = 1;
delta = 1e-5;
%dt=2.5e-4;
implicit=1;
tf = 200;
nSt = (tf/dt);
saveEvery = max(1e-2/dt,1);
nSave = nSt/saveEvery;
rng(seed);
MaxIts = 10;
Confine = 0;

nW = 1;
Mobility = @(x) MobRPY(x,a,mu,ds);
% Hessians are constant
H = HessMat(Nlinks);
AllTanVecDots = zeros(nRuns,Nlinks);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,nSave);

for iRun=1:nRuns
% Initial state
tau = [ones(Nlinks,1) zeros(Nlinks,2)];
x = reshape(([0 0 8*a]+[0 0 0; cumsum(ds*tau)])',[],1);
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

% Unconstrained step
for iT=1:nSt
    M = Mobility(x);
    Mhalf = chol(M)';
    C = GradMat(x);
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
        Cc = GradMat(xc);
        divMC = divMC + 1/(nW*delta)*(Mc*Cc'-M*C')*w2;
    end
    divMtru = divM;
    divMctru = divMC;
    if (wrongdrift)
        divMtru=0*divMtru;
        divMctru=0*divMctru;
    end

    % Take unconstrained step 
    W = randn(nX,1);
    ExPart = dt*kbT*(divMtru-divMctru)+sqrt(2*dt*kbT)*Mhalf*W;
    if (Confine)
        FConf = zeros(nX,1);
        FConf(3:3:end)=-20*(x(3:3:end)-8*a);
        ExPart = ExPart + dt*M*FConf;
    end
    if (implicit)
        xtilde = (eye(3*(Nlinks+1))+dt*M*EMat) \ (x+ExPart);
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
        C = GradMat(xg);
        Htot = sum(H.*reshape(lam,1,1,nC),3);
        J = [eye(nX)+M*Htot M*C'; C zeros(nC)];
        [~,ceqc]=c(xg,ds);
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
        eqconstr = @(xvar) c(xvar,ds);
        opts=optimoptions(@lsqnonlin,'OptimalityTolerance',1e-10,...
            'SpecifyObjectiveGradient',true,'Display','off');
        [xg,~,~,exitflag,~,~,~] = ...
            lsqnonlin(fun,x,[],[],[],[],[],[],eqconstr,opts);
    end
    x = xg;
    if (mod(iT,saveEvery)==0)
        index = iT/saveEvery;
        NumIts(index)=it;
        eedists(index)=norm(x(1:3)-x(end-2:end));
        if (iT/nSt>0)
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
if (~wrongdrift)
    save(strcat('WLC_dt',num2str(dt),'_',num2str(seed),'.mat'))
else
    save(strcat('WrongDrWLC_dt',num2str(dt),'_',num2str(seed),'.mat'))
end
end
%end
% dts = [2.5e-3 1e-3 2.5e-4 1e-4 1e-4];
% for cIndex=4:5
% nRuns=50;
% nSave=2000;
% Nlinks=10;
% dt = dts(cIndex);
% TotalTanVecDots = zeros(nRuns,Nlinks);
% TotalFailureRates = zeros(nRuns,1);
% TotalItCounts = zeros(nRuns,nSave);
% TotalEE  = zeros(nRuns,nSave);
% for kRun=1:nRuns
%     if (cIndex==5)
%         load(strcat('WrongDrConfinedWLC_dt',num2str(dt),'_',num2str(kRun),'.mat'))
%     else
%         load(strcat('ConfinedWLC_dt',num2str(dt),'_',num2str(kRun),'.mat'))
%     end
%     TotalTanVecDots(kRun,:)=AllTanVecDots;
%     TotalFailureRates(kRun)=FailureRates;
%     TotalItCounts(kRun,:)=AllItCounts;
%     TotalEE(kRun,:)=AllEE;
% end
% AllTanVecDots=TotalTanVecDots;
% cIndex=1;
% diffc = (0:Nlinks-1)*ds;
% MC = mean(AllTanVecDots);
% SC = 2*std(AllTanVecDots)/sqrt(nRuns);
% Colors=get(gca,'ColorOrder');
% fill([diffc, fliplr(diffc)], [MC-SC, fliplr(MC+SC)],...
%     Colors(cIndex,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
% hold on
% plot(diffc,MC,'-o','Color',Colors(cIndex,:),'LineWidth',2)
% hold on
% plot(diffc,exp(-diffc/lp))


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

function [val,J] = ProjectionObjective(x,xtilde,Minv)
    val = 1/2*(x-xtilde)'*Minv*(x-xtilde);
    if nargout > 1  
        J = Minv*(x-xtilde);
        J = J';
    end
end

function [cleq,cd] = c(x,ds)
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    cd = zeros(Nx-1,1);
    for j=1:Nx-1
        cd(j) = norm(x(j+1,:)-x(j,:)).^2-ds^2;
    end
    cleq=[];
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
    tau = (x(2:end,:)-x(1:end-1,:))/ds;
    tauavg = zeros(Nx,3);
    for j=2:Nx-1
        tauavg(j,:)=1/2*(tau(j-1,:)+tau(j,:));
    end
    tauavg(1,:)=tau(1,:);
    tauavg(end,:)=tauavg(end,:);
    M = zeros(3*Nx);
    for j=1:Nx
        M(3*j-2:3*j,3*j-2:3*j)=...
            log(a^(-2))/(8*pi*mu)*(eye(3)+tauavg(j,:)'*tauavg(j,:));
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
