% Parameters for chain and mobility
function SpectralChain(seed,dt,wrongdrift)
%if (0)
addpath(genpath('../../'))
nRuns = 1;
%seed=1;
%wrongdrift=0;

ds = 0.1;
Nx = 11;
L = 1;
kbT = 4.1e-3; % pN * um
lp = L;
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
WTilde_Nx = stackMatrix(WTilde_1D);
EMat = K_b*stackMatrix(DX^2)'*WTilde_Nx*...
    stackMatrix(DX^2);

nW = 1;
Mobility = @(x) MobRPY(x,DX,a,mu);
% Hessians are constant
H = HessMat(Nx,D);
AllTanVecDots = zeros(nRuns,Nx-1);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,nSave);

for iRun=1:nRuns
% Initial state
x = [sX zeros(Nx,2)];
x = reshape(x',[],1);
nC = Nx-1;
nX = length(x);

% Statistics
TanVecDots = zeros(Nx-1,1);
nSamplesDs = zeros(Nx-1,1);
NumIts = zeros(nSave,1);
eedists = zeros(nSave,1);
nFail = 0;

% Unconstrained step
for iT=1:nSt
    M = Mobility(x);
    Mhalf = chol(M)';
    C = GradMat(x,D);
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
        Cc = GradMat(xc,D);
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
        C = GradMat(xg,D);
        Htot = sum(H.*reshape(lam,1,1,nC),3);
        J = [eye(nX)+M*Htot M*C'; C zeros(nC)];
        [~,ceqc]=c(xg,D);
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
        if (iT/nSt>1/2)
            % Tangent vector dot products
            tau = reshape(D*x,3,[])';
            for iLink=1:Nx-1
                for jLink=iLink:Nx-1
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
if (wrongdrift==0)
    save(strcat('Spectral_dt',num2str(dt),'_',num2str(seed),'.mat'))
elseif (wrongdrift==1)
    save(strcat('NoDrSpectral_dt',num2str(dt),'_',num2str(seed),'.mat'))
elseif (wrongdrift==2)
    save(strcat('WrongDrSpectral_dt',num2str(dt),'_',num2str(seed),'.mat'))
end
end
% dts = [1e-4 1e-4];
% for cIndex=1:2
% nRuns=50;
% nSave=4000;
% Nx=11;
% dt = dts(cIndex);
% TotalTanVecDots = zeros(nRuns,Nx-1);
% TotalFailureRates = zeros(nRuns,1);
% TotalItCounts = zeros(nRuns,nSave);
% TotalEE  = zeros(nRuns,nSave);
% for kRun=1:nRuns
%     if (cIndex==2)
%         load(strcat('aWrongDrConfinedWLC_dt',num2str(dt),'_',num2str(kRun),'.mat'))
%     else
%         load(strcat('aConfinedWLC_dt',num2str(dt),'_',num2str(kRun),'.mat'))
%     end
%     TotalTanVecDots(kRun,:)=AllTanVecDots;
%     TotalFailureRates(kRun)=FailureRates;
%     TotalItCounts(kRun,:)=AllItCounts;
%     TotalEE(kRun,:)=AllEE;
% end
% AllTanVecDots=TotalTanVecDots;
% ds = L/(Nx-1);
% diffc = (0:Nx-2)*ds;
% MC = mean(AllTanVecDots);
% SC = 2*std(AllTanVecDots)/sqrt(nRuns);
% Colors=get(gca,'ColorOrder');
% fill([diffc, fliplr(diffc)], [MC-SC, fliplr(MC+SC)],...
%     Colors(cIndex,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
% hold on
% plot(diffc,MC,'-o','Color',Colors(cIndex,:),'LineWidth',2)
% hold on
% plot(diffc,exp(-diffc/lp))
% end


function [val,J] = ProjectionObjective(x,xtilde,Minv)
    val = 1/2*(x-xtilde)'*Minv*(x-xtilde);
    if nargout > 1  
        J = Minv*(x-xtilde);
        J = J';
    end
end

function [cleq,cd] = c(x,D)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    tau = reshape(D*x,3,[])';
    cd = sum(tau.*tau,2)-1;
    cleq=[];
end

function C = GradMat(x,D)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    Nx = length(x)/3;
    C = zeros(Nx-1,3*Nx);
    for j=1:Nx-1
        DDt = D(3*j-2:3*j,:)'*D(3*j-2:3*j,:);
        C(j,:)=(2*DDt*x)';
    end
end

function H = HessMat(Nx,D)
    H = zeros(3*Nx,3*Nx,Nx-1);
    for j=1:Nx-1
        H(:,:,j)=2*D(3*j-2:3*j,:)'*D(3*j-2:3*j,:);
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
