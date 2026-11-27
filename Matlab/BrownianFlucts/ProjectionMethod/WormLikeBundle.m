% Parameters for chain and mobility
function WormLikeBundle(seed,Nlinks,dt,Fext0)
addpath(genpath('../../'))
%Nlinks=16;
nFib=1;
x0s=[0 0 0; 0.1 0 0];
clamp0=1;
%seed=1;
%dt=1e-4;
nRuns=1;
ds = 1/Nlinks;
L = ds*Nlinks;
kbT = 4.1e-3; % pN * um
lp = 10*L;
K_b = lp*kbT;
a = 1e-2;
mu = 1;
implicit=1;
tf = 10;
nSt = (tf/dt);
saveEvery = max(1e-2/dt,1);
nSave = floor(1e-10+nSt/saveEvery);
rng(seed);
MaxIts = 10;
tol = 1e-10;

Mobility = @(x) MobRPY(x,a,mu,ds);
GC = @(x) GradMat(x,clamp0,nFib);
cfcn = @(x) c(x,ds,clamp0,nFib,x0s);
AllTanVecDots = zeros(nRuns,Nlinks);
FailureRates = zeros(nRuns,1);
AllItCounts = zeros(nRuns,nSave);
AllEE  = zeros(nRuns,nSave);
Xpts=[];
zcoords=[];

for iRun=1:nRuns
% Initial state
tau = [ zeros(Nlinks,2) ones(Nlinks,1)];
x = [];
for iFib=1:nFib
    x=[x; reshape(([0 0 0; cumsum(ds*tau)])',[],1)+repmat(x0s(iFib,:)',Nlinks+1,1)];
end
nX = length(x);


% The energy matrix
EMat1 = WLCEnergyMatrix(K_b,Nlinks,ds);
EMat=[];
for iFib=1:nFib
    EMat = blkdiag(EMat,EMat1);
end

% Statistics
TanVecDots = zeros(Nlinks,1);
nSamplesDs = zeros(Nlinks,1);
NumIts = zeros(nSave,1);
eedists = zeros(nSave,1);
nFail = 0;
nReallyFail = 0;
% For when Newton fails
opts=optimoptions(@fsolve,'OptimalityTolerance',1e-10,...
    'SpecifyObjectiveGradient',true,'Display','off');

% Gradient check
dx = rand(nX,1);
gc = GC(x)*dx;
for iEps=1:10
    trudiff = cfcn(x+10^(-iEps)*dx)/10^(-iEps);
    ers(iEps) = norm(trudiff-gc);
end
nC = length(cfcn(x));


% Unconstrained step
for iT=1:nSt
    Fex = zeros(nX,1);
    Fex(3*(Nlinks+1):(3*Nlinks+3):end)=Fext0;
    [xg,newtits,newtfail,allfail] = ...
        AdvanceProjection(x,dt,kbT,EMat,Mobility,GC,cfcn,...
         MaxIts,tol,nC,implicit,opts,Fex);
    if (newtfail)
        nFail=nFail+1;
        nReallyFail=nReallyFail+allfail;
    end
    x = xg;
    if (mod(iT,saveEvery)==0)
        index = floor(1e-6+iT/saveEvery);
        NumIts(index)=newtits;
        eedists(index)=norm(x(1:3)-x(end-2:end));
        xnow = reshape(x,3,[])';
        Xpts=[Xpts;xnow];
        zcoords = [zcoords; mean(xnow(Nlinks+1:Nlinks+1:end,3))];
        if (0)
        for iFib=1:nFib
            xthis = xnow((iFib-1)*(1+Nlinks)+(1:Nlinks+1),:);
            plot3(xthis(:,1),xthis(:,2),xthis(:,3),'-o')
            hold on
        end
        xlim([-0.25 0.25])
        ylim([-0.25 0.25])
        zlim([0 1])
        PlotAspect
        drawnow
        hold off
        end
    end
end
AllEE(iRun,:)=eedists;
FailureRates(iRun) = nFail/nSt;
AllTanVecDots(iRun,:) = TanVecDots./nSamplesDs;
AllItCounts(iRun,:)=NumIts;
end
save(strcat('BundleNFib',num2str(nFib),'Fext',num2str(Fext0),'RPY',num2str(Nlinks),'_dt',...
    num2str(dt),'_',num2str(seed),'.mat'))
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

function cdAll = c(x,ds,clamp0,nFib,x0s)
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1)/nFib;
    cdAll=[];
    for iFib=1:nFib
        xThis = x((iFib-1)*Nx+(1:Nx),:);
        cd = zeros(Nx-1,1);
        for j=1:Nx-1
            cd(j) = norm(xThis(j+1,:)-xThis(j,:)).^2-ds^2;
        end
        % Clamping conditions (freeze first two pts)
        if (clamp0)
            cd(1)=[];
            cd=[cd;xThis(1,:)'-x0s(iFib,:)';xThis(2,:)'-x0s(iFib,:)'-[0;0;ds]];
        end
        cdAll=[cdAll;cd];
    end
    % Cross linking constraint
    if (nFib>1)
    diff = x0s(2,1)-x0s(1,1);
    cdAll = [cdAll; norm(x(2*Nx,:)-x(Nx,:))^2-diff^2];
    end
end

function CAll = GradMat(x,clamp0,nFib)
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1)/nFib;
    CAll=[];
    for iFib=1:nFib
        xThis = x((iFib-1)*Nx+(1:Nx),:);
        C = zeros(Nx-1,3*Nx);
        for j=1:Nx-1
            C(j,3*j-2:3*j)=2*(xThis(j,:)-xThis(j+1,:));
            C(j,3*j+1:3*j+3)=2*(xThis(j+1,:)-xThis(j,:));
        end
        % Clamping conditions
        if (clamp0)
            C(1,:)=[];
            C=[C;zeros(6,3*Nx)];
            for p=1:6
                C(Nx-2+p,p)=1;
            end
        end
        CAll=blkdiag(CAll,C);
    end
    % Cross linking constraint
    if (nFib>1)
    CAll=[CAll;zeros(1,3*Nx*nFib)];
    CAll(end,3*Nx-2:3*Nx)=2*(x(Nx,:)-x(2*Nx,:));
    CAll(end,6*Nx-2:6*Nx)=2*(x(2*Nx,:)-x(Nx,:));
    end
end

function M = MobRPY(x,a,mu,ds)
    % Starting with free space RPY kernel
    if (size(x,2)==1)
        x=reshape(x,3,[])';
    end
    Nx = size(x,1);
    % tauavg = zeros(Nx,3);
    % for j=2:Nx-1
    %     tauavg(j,:)=(x(j+1,:)-x(j-1,:));
    % end
    % tauavg(1,:)=x(2,:)-x(1,:);
    % tauavg(Nx,:)=x(Nx,:)-x(end-1,:);
    % tauavg = tauavg./sqrt(sum(tauavg.*tauavg,2));
    % M = zeros(3*Nx);
    % for j=1:Nx
    %     M(3*j-2:3*j,3*j-2:3*j)=...
    %         log(a^(-2))/(8*pi*mu*ds)*(eye(3)+tauavg(j,:)'*tauavg(j,:));
    % end
    M = 1/(6*pi*mu*a)*eye(3*Nx);
    for i=1:Nx
        for j=i+1:Nx
            rvec = x(i,:)-x(j,:);
            mij =  RPYTot(rvec,a,mu);
            M(3*i-2:3*i,3*j-2:3*j) =mij;
            M(3*j-2:3*j,3*i-2:3*i) =mij;
        end
    end
   % M = RPYMatrixWithWall(x,mu, a);
end
