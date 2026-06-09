% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
function ConnectedNetworkBD(seed,CL,nLayers)
%% Define constants 
addpath(genpath('../'))
%seed=1;
Nx = 8;
L = 0.5; 
ell = 0.25;
%CL=1;
clamp0=1;
ConfineZ=1;
%nLayers=8;
impcoeff = 1; 
anglebr=70;

% List of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
% Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
%     4 0.5 5 0 0;  1 0.1 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0 0; ...
%     8 0.5 9 0 0; 2 1 3 1 1; 1 0.3 2 0.2 1; 2 0.3 3 0.2 1; 1 1 7 0.7 1; ...
%     8 1 9 1 1];
% nFib=9;
%Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 1 0.2 2 1 1; 1 1 3 1 1];
% Two PCs should be equivalent for the first 3 connections below. Then
% almost the same for the last one. 
%Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 1 0.3 2 0.2 1; 1 1 3 1 1; 2 1 1 0 1]; 
%nFib=3;

% Networks for testing the preconditioner
% nFib=40;
% Connections = [(1:nFib/2-1)' [0.4;0.8*ones(nFib/2-2,1)] (2:nFib/2)' zeros(nFib/2-1,2); ...
%   [1;(nFib/2+1:nFib-1)'] 0.8*ones(nFib/2,1) (nFib/2+1:nFib)' zeros(nFib/2,2)];
%  NewConn = [(2:2:nFib/2-1)'; (nFib/2+1:2:nFib-1)'];
%  Connections=[Connections;  NewConn 0.5*ones(length(NewConn),1) ...
%    NewConn+1 0.6*ones(length(NewConn),1) ones(length(NewConn),1)];
% %NewConn = [(2:5)' ones(4,1) (nFib/2+1:nFib/2+4)' 0.1*ones(4,1) ones(4,1)]
% %Connections=[Connections; NewConn];
% specfib=[];

if (1)
x1 = cos(anglebr*pi/180);
Backbone = [(1:2*nLayers-1)' L*(ones(2*nLayers-1,1)-x1) (2:2*nLayers)' zeros(2*nLayers-1,2)];
Backbone(end,2)=0.5-x1;
Connections = Backbone;
LayerFibs=cell(nLayers,1);
for k=1:nLayers-1
    LastFib  = max(Connections(:,3));
    nThisLayer = 2*nLayers+1-2*k;
    Layer = [[(2*k-1) LastFib+1:LastFib+nThisLayer-1]' L*[0.5-x1;ones(nThisLayer-1,1)-x1] ...
        (LastFib+1:LastFib+nThisLayer)' zeros(nThisLayer,2)];
    Connections = [Connections;Layer];
    LayerFibs{k} = Layer(:,3);
end
LayerFibs{nLayers}=Backbone(:,3);

nFib = max(max(Connections(:,1)),max(Connections(:,3)));
specfib=2*nLayers;
if (CL)
    for iL=1:size(LayerFibs,1)-1
        for pL=1:nLayers
            try
            Connections=[Connections;LayerFibs{iL}(end-(2*pL-1)) L LayerFibs{iL+1}(end-(2*pL-2)) L 1];
            %Connections=[Connections;LayerFibs{iL}(end-(2*pL-1)) L LayerFibs{iL}(end-(2*pL-2)) L 1];
            catch
            end
        end
    end
end
end

rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 10;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;

%% Initialization
rng(seed);
makeMovie = 0;
dt=1e-4;
tf = 100;
gmrestol = 1e-3;

[DOFs,MasterConnections,SlaveConnections, ConstrainedPosNodes,...
  TangentVectorNodes,BranchIndices,IntegrationMatrix,...
  DiffMatrix,RegGridMatrix,LeadIndicesByFib,clampedTau] = ...
    InitializeConnectedNetwork(Connections,nFib,Nx,L,ell,anglebr,clamp0,specfib);
nAlphas = 3*(size(DOFs,1)-sum(MasterConnections(:,5)==0)-clamp0);

RegGridMatrixInv = cell(size(RegGridMatrix));
for iFib=1:nFib
    RegGridMatrixInv{iFib}=RegGridMatrix{iFib}^(-1);
end
% Initialize X and X inverse functions
XFcn = @(dof3d) XConnectedNetwork(dof3d,MasterConnections,...
    SlaveConnections,LeadIndicesByFib, Nx,nFib,L,RegGridMatrix,IntegrationMatrix,clamp0,0);
XInvFcn = @(x3d) XInvConnectedNetwork(x3d,MasterConnections,...
    SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrixInv,DiffMatrix,clamp0);
XTrFcn = @(lam3d) XTrConnectedNetwork(lam3d,MasterConnections,...
    SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrix,IntegrationMatrix,clamp0);
XInvTrFcn = @(Om3d) XInvTrConnectedNetwork(Om3d,MasterConnections,...
     SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrix,DiffMatrix,clamp0);
KFcn = @(alpha,x) KApplyConnNet(alpha,x,XFcn,XInvFcn,BranchIndices,clampedTau);
KTFcn =  @(Lam,x) KTApplyConnNet(Lam,x,XTrFcn,XInvFcn,BranchIndices,clampedTau);
KInvFcn = @(U,x) KInvApplyConnNet(U,x,XInvFcn,BranchIndices,clampedTau);
KTInvFcn = @(alpha,x)  KInvTApplyConnNet(alpha,x,XInvTrFcn,XInvFcn,BranchIndices,clampedTau);

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
ApplyAMat = @(x,Xt,AllMs) ApplyBigMatrix(x,Xt,AllMs,KFcn,KTFcn, ...
    BendForceMat,impcoeff*dt,nFib);
PrecompPCMat = @(Xtilde,AllMs) PrecomputePairwisePC(Xtilde,...
        XInvFcn,AllMs, BranchIndices,MasterConnections,SlaveConnections,...
        IntegrationMatrix,RegGridMatrix,RegGridMatrixInv,BendForceMat,...
        impcoeff*dt,L,nFib,clampedTau);
PairPC = @(RHS,PCMats,AllRHSInds,NodeOrderByPair,TauStart) ...
    PairwisePCConnNet(RHS,PCMats,AllRHSInds,NodeOrderByPair,...
    MasterConnections,SlaveConnections,RegGridMatrix,RegGridMatrixInv,...
    Nx,L,nFib,TauStart,clampedTau);

%% Initialize arrays to save 
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-2/dt+1e-10));
Xpts=[];
ee=[];
MDDist=[];
NumGIts=[];
rv=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sX,bX);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end

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
            %ylim([0 2])
            %xlim([-1.75 1.5])
            PlotAspect
            % Connections
            for iConn=1:size(Connections,1)
                iFib = Connections(iConn,1);
                iS = Connections(iConn,2);
                jFib = Connections(iConn,3);
                jS = Connections(iConn,4);
                pts = [barymat(iS,sX,bX)*PtsThisT((iFib-1)*Nx+(1:Nx),:); ...
                    barymat(jS,sX,bX)*PtsThisT((jFib-1)*Nx+(1:Nx),:)];
                if (Connections(iConn,5)==1)
                plot3(pts(:,1),pts(:,2),pts(:,3),':ko','MarkerSize',2,'MarkerFaceColor','k')
                else
                plot3(pts(:,1),pts(:,2),pts(:,3),'bo','MarkerSize',2,'MarkerFaceColor','k')
                end
            end
            movieframes(frameNum)=getframe(f);
        end
        t
        MotherEnd = barymat(L,sX,bX)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sX,bX)*PtsThisT(Nx+1:2*Nx,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        MDDist=[MDDist; dispMDT];
        Xpts=[Xpts;PtsThisT];
        ee =[ee; norm(PtsThisT(1,:)-PtsThisT(Nx,:)); ...
            norm(PtsThisT(Nx+1,:)-PtsThisT(2*Nx,:))];
        NumGIts=[NumGIts;length(rv)];
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
    OmegaTilde = KInvFcn(RandomVelBM, Xt);
    OmegaTilde = AssignBranchNodes(OmegaTilde,BranchIndices,clampedTau);
    TauBarTilde = updateByRotate(DOFs,OmegaTilde,dt/2,clamp0);
    Xtilde = reshape(XFcn(TauBarTilde)',[],1);

    % RFD and other RHS velocities
    g3 = randn(nAlphas,1);
    OmRFD=AssignBranchNodes(reshape(g3,3,[])',BranchIndices,clampedTau);
    delta = 1e-5;
    TauBarPlus = updateByRotate(DOFs,OmRFD,delta,clamp0);
    XPlus = reshape(XFcn(TauBarPlus)',[],1);
    KInvTOm = reshape(KTInvFcn(g3,Xt)',[],1);
    KInvTOmPlus = reshape(KTInvFcn(g3,XPlus)',[],1);
    M_RFD = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    if (ConfineZ)
        Fext(3:3:end)=-Xt(3:3:end);
    end
    U0 = zeros(3*Nx*nFib,1);
    MTildeF = zeros(nFib*3*Nx,1);
    AllMs = zeros(3*Nx,3*Nx,nFib);
    RandomVelBE = zeros(nFib*3*Nx,1);
    g2 = randn(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWPlusOne = MobFcn(XPlus(finds));
        MW = MobFcn(Xt(finds));
        MWTilde = MobFcn(Xtilde(finds));
        AllMs(:,:,iFib) = MWTilde;
        M_RFD(finds) = kbT/delta*(MWPlusOne*KInvTOmPlus(finds)...
            -MW*KInvTOm(finds));
        RandomVelBE(finds) = sqrt(kbT)*MWTilde*BendMatHalf*g2(finds);
        MTildeF(finds)=MWTilde*(BendForceMat*Xt(finds)+Fext(finds));
    end
    RHSVel = RandomVelBM + M_RFD + RandomVelBE + MTildeF + U0;
    RHSAll = [RHSVel; zeros(nAlphas,1)];

    % Iterative solver with preconditioner
    [PCMats,AllRHSInds,TauStart,NodeOrderByPair] = PrecompPCMat(Xtilde,AllMs);
    PreconPW = @(RHS) PairPC(RHS,PCMats,AllRHSInds,NodeOrderByPair,TauStart);
    [AllxG,~,~,~,rv] = gmres(@(y)ApplyAMat(y,Xtilde,AllMs),RHSAll,...
       [],gmrestol,100,PreconPW);
    Lambda = AllxG(1:3*Nx*nFib);
    alphaU = AllxG(3*Nx*nFib+1:end);

    % Assign branch nodes
    alphaU=AssignBranchNodes(reshape(alphaU,3,[])',BranchIndices,clampedTau);
    % Evolve constants by rotating and translating the link
    DOFs_next = updateByRotate(DOFs,reshape(alphaU',[],1),dt,clamp0);
    DOFs=DOFs_next;
end
if (CL==1)
save(strcat('BranchedCLNtwkL',num2str(nLayers),'_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
else
save(strcat('BranchedNtwkL',num2str(nLayers),'_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
end
end

function DOFsNew = updateByRotate(DOFs,alphaU,dt,clamp0)
    DOFsNew = 0*DOFs;
    % Update the tangent vectors
    if (size(alphaU,2)==1)
        alphaU = reshape(alphaU,3,[])';
    end
    DOFsNew(1:end-1,:) = rotateTau(DOFs(1:end-1,:),alphaU(1:end-1,:),dt);
    % Add the constant
    if (clamp0)
        DOFsNew(end,:) = rotateTau(DOFs(end,:),alphaU(end,:),dt);
    else
        DOFsNew(end,:)=DOFs(end,:)+dt*alphaU(end,:);
    end
end

% Apply the big matrix
% Input: vector of (Lambda,alphaU)
% Output: [-M*Lambda+K*alphaU; K'*Lambda 0]
% (without explicitly forming matrices)
function Ax = ApplyBigMatrix(x,X1D,AllMs,KFcn,KTFcn, ...
    BendForceMat,impcodt,nFib)

    Nxx = length(X1D);
    Nx = Nxx/(3*nFib);
    Lam1D=x(1:Nxx);
    Lam = reshape(Lam1D,3,[])';
    alphaU = reshape(x(Nxx+1:end),3,[])';
    KAlpha1D=reshape(KFcn(alphaU,X1D)',[],1);

    MLam = zeros(Nxx,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = AllMs(:,:,iFib);
        MLam(finds)=MWsymTildeOne*(Lam1D(finds)+impcodt*BendForceMat*KAlpha1D(finds));
    end
    Eq1 = -MLam+KAlpha1D;

    % Second eqn K^T Lam = 0 
    % K^T Lam = C^T X^T Lam = -C X^T Lam = cross(DOFs(iTau,:),XTLam(iTau,:)
    KTLam=reshape(KTFcn(Lam,X1D)',[],1);
    Eq2 = reshape(KTLam',[],1);
    
    Ax=[Eq1;Eq2];
end



