% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
%% Define constants 
addpath(genpath('../'))
seed=30;
N = 15;
Nx = N+1;
L = 1;
ell = 0.1;
% List of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
% Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
%     4 0.5 5 0 0;  1 0.9 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0.1 1; ...
%     8 0.5 9 0 0; 9 0.5 10 0.1 1];
%nFib=10;
nFib=2;
Connections = [(1:nFib-1)' 0.8*ones(nFib-1,1) (2:nFib)' zeros(nFib-1,2)];
Connections(2:3:end,5)=1;
NLinks = sum(Connections(:,5));
NBranch = length(Connections(:,5))-NLinks;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;

%% Initialization
rng(seed);
impcoeff = 1;
makeMovie = 1;
dt = 1e-4;
tf = 1e-3;

[paths,DOFs,TangentVectorNodes,IntegrationMatrix,DiffMatrix,...
    NodesByBranch,PairwiseXMats] = ...
    InitializeConnectedNetwork(Connections,nFib,N,L,ell);

% Initialize X and X inverse functions
X=XConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,DOFs,IntegrationMatrix,1);
XFcn = @(dof3d) XConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,dof3d,IntegrationMatrix,0);
XInvFcn = @(x3d) XInvConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,x3d,DiffMatrix);
XTrFcn = @(lams) XTrConnectedNetwork(Connections,nFib,N,L,ell,...
        paths,lams,IntegrationMatrix);
DOFs = XInvFcn(X);
Xt = reshape(X',[],1);
Nxx = 3*Nx*nFib;
Ndd = 3*(nFib*N-NBranch+NLinks+1);

% Bending energy matrix (2Nx grid)
[sX,wX,bX] = chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sX,bX);
WTilde_1D = R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx;
WTilde_Nx = stackMatrix(WTilde_1D);
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix(DX^2)'*WTilde_Nx*stackMatrix(DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));

% Pre-computations for mobility
MobConst = -log(eps^2)/(8*pi*mu);
MobFcn = @(x1d) LocalDragMob(x1d,DX,MobConst,WTilde_Nx_Inverse);
ApplyBigC = @(x,Xt) ApplyBigCMatrix(x,Xt,XFcn,XInvFcn,XTrFcn,MobFcn,NodesByBranch,...
        BendForceMat,impcoeff*dt,nFib);
PrecompPCMat = @(Xt) PrecomputePairwisePC(Xt,XInvFcn,MobFcn,...
    PairwiseXMats,paths,Connections,BendForceMat,impcoeff*dt,nFib);
PairPC = @(b,PrecompMats) PairwisePCConnNet(b,PrecompMats,...
    paths,Connections,NodesByBranch,nFib,Nx);

%% Initialize arrays to save 
stopcount=floor(tf/dt+1e-5);
saveEvery=1;%max(1,floor(1e-2/dt+1e-10));
Xpts=[];
ee=[];
MDDist=[];
Xbars=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sX,bX);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;

%% Computations
for count=0:stopcount
    t=count*dt;
    Xt = reshape(XFcn(DOFs)',[],1);

    if (mod(count,saveEvery)==0)
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        if (makeMovie)
            clf;
            frameNum=frameNum+1;
            % Fibers
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            %ylim([-1 1])
            %xlim([-1 1])
            PlotAspect
            % Connections
            for iConn=1:size(Connections,1)
                iFib = Connections(iConn,1);
                iS = Connections(iConn,2);
                jFib = Connections(iConn,3);
                jS = Connections(iConn,4);
                pts = [barymat(iS,sX,bX)*PtsThisT((iFib-1)*Nx+(1:Nx),:); ...
                    barymat(jS,sX,bX)*PtsThisT((jFib-1)*Nx+(1:Nx),:)];
                plot3(pts(:,1),pts(:,2),pts(:,3),':ko')
            end
            movieframes(frameNum)=getframe(f);
        end
        MotherEnd = barymat(L,sX,bX)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sX,bX)*PtsThisT(Nx+1:2*Nx,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        MDDist=[MDDist; dispMDT];
        Xpts=[Xpts;PtsThisT];
        ee =[ee; norm(PtsThisT(1,:)-PtsThisT(Nx,:)); ...
            norm(PtsThisT(Nx+1,:)-PtsThisT(2*Nx,:))];
    end  
    
    % Matrices at time step n 
    RandomVelBM = zeros(nFib*3*Nx,1);
    BendForce = zeros(nFib*3*Nx,1);
    g = randn(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymOne = MobFcn(Xt(finds));
        MWsymHalfOne = chol(MWsymOne)';
        % Obtain Brownian velocity
        RandomVelBM(finds) = sqrt(2*kbT/dt)*MWsymHalfOne*g(finds);
        BendForce(finds)=BendForceMat*Xt(finds);
    end
    
    % Advance to midpoint
    OmegaTilde = XInvFcn(reshape(RandomVelBM,3,[])');
    for iR =1:size(OmegaTilde,1)-1
        OmegaTilde(iR,:)=cross(DOFs(iR,:),OmegaTilde(iR,:));
    end
    if (~isempty(NodesByBranch))
        OmegaTilde(NodesByBranch(:,2),:)=OmegaTilde(NodesByBranch(:,1),:);
    end
    TauBarTilde = updateByRotate(DOFs,reshape(OmegaTilde',[],1),dt/2);
    Xtilde = reshape(XFcn(TauBarTilde)',[],1);

    Fext = zeros(3*Nx*nFib,1);
    MTildeF = zeros(nFib*3*Nx,1);
    M_RFD = zeros(nFib*3*Nx,1);
    RandomVelBE = zeros(nFib*3*Nx,1);
    g2 = randn(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = MobFcn(Xtilde(finds));
        MWsymOne = MobFcn(Xt(finds));
        M_RFD(finds) = (MWsymTildeOne-MWsymOne)*(MWsymOne \ RandomVelBM(finds));
        RandomVelBE(finds) = sqrt(kbT)*MWsymTildeOne*BendMatHalf*g2(finds);
        MTildeF(finds)=MWsymTildeOne*(BendForceMat*Xt(finds)+Fext(finds));
    end
    U0 = zeros(3*Nx*nFib,1);
    %U0(1:3:end)=Xt(1:3:end);
    %U0(2:3:end)=-Xt(2:3:end);
    RHSVel = RandomVelBM + M_RFD + RandomVelBE + MTildeF + U0;
    RHSAll = [RHSVel; zeros(Ndd,1)];

    % Iterative (not exact - errors could accumulate(?))
    %tic
    PrecompMats = PrecompPCMat(Xtilde);
    [AllxG,flag,~,~,rv] = gmres(@(y)ApplyBigC(y,Xtilde),RHSAll,...
       [],1e-10,100,@(b) PairPC(b,PrecompMats));
    norm(ApplyBigC(AllxG,Xtilde)-RHSAll)/norm(RHSAll)
    length(rv)
    Lambda = AllxG(1:Nxx);
    alphaU = reshape(AllxG(Nxx+1:end),3,[])';
    %toc
    % Assign branch nodes
    for iBr=1:NBranch
        masternode = NodesByBranch(iBr,1);
        slavenode = NodesByBranch(iBr,2);
        alphaU = [alphaU(1:slavenode-1,:); alphaU(masternode,:); ...
            alphaU(slavenode:end,:)];
    end
    % Evolve constants by rotating and translating the link
    DOFs_next = updateByRotate(DOFs,reshape(alphaU',[],1),dt);
    DOFs=DOFs_next;
end
%Totaltime=toc(tStart);
%save(strcat('Branched_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts','MDDist')

function DOFsNew = updateByRotate(DOFs,alphaU,dt)
    DOFsNew = 0*DOFs;
    % Update the tangent vectors
    Omega = reshape(alphaU(1:end-3),3,[])';
    DOFsNew(1:end-1,:) = rotateTau(DOFs(1:end-1,:),Omega,dt);
    % Add the constant
    DOFsNew(end,:)=DOFs(end,:)+dt*alphaU(end-2:end)';
end



