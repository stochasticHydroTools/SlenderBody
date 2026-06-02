% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
%function ConnectedNetworkBDDirect(seed,Nx,dt)
for seed=1:5
for CL=[0 1]
%% Define constants 
addpath(genpath('../'))
%seed=1;
Nx = 8;
L = 0.5;
ell = 0.25;
clamp0 = 1;
% List of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
% Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
%     4 0.5 5 0 0;  1 0.1 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0 0; ...
%     8 0.5 9 0 0; 2 1 3 1 1; 1 0.3 2 0.2 1; 2 0.3 3 0.2 1; 1 1 7 0.7 1; ...
%     8 1 9 1 1];
% nFib=9;
brang=70;
Connections = [1 L*(1-cos(brang/180*pi)) 2 0 0];% 2 0.5 3 0 0];% 1 0.1 2 0.9 1; 2 0.6 1 0.05 1];
nFib=2;
if (CL)
    Connections =[Connections; 1 L 2 L 1];
end
%nFib=40;
%Connections = [(1:nFib/2-1)' [0.4;0.8*ones(nFib/2-2,1)] (2:nFib/2)' zeros(nFib/2-1,2); ...
%    [1;(nFib/2+1:nFib-1)'] 0.8*ones(nFib/2,1) (nFib/2+1:nFib)' zeros(nFib/2,2)];
% Connections(2:2:end,5)=1;
% Connections=[Connections; 3 0.7 4 0.1 1; 4 1 5 0.7 1];
specfib=[];
%BranchedNetForFluct;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 10;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;

%% Initialization
rng(seed);
impcoeff = 1;
makeMovie = 0;
dt=1e-4;
tf=100;

[DOFs,MasterConnections,SlaveConnections, ConstrainedPosNodes,...
  TangentVectorNodes,BranchIndices,IntegrationMatrix,...
  DiffMatrix,RegGridMatrix,LeadIndicesByFib,clampedTau] = ...
    InitializeConnectedNetwork(Connections,nFib,Nx,L,ell,brang,clamp0,specfib);

% Initialize X and X inverse functions
[X,XMat]=XConnectedNetwork(DOFs,MasterConnections,SlaveConnections,...
    LeadIndicesByFib,Nx,nFib,L,RegGridMatrix,IntegrationMatrix,clamp0,1);
RegGridMatrixInv = cell(size(RegGridMatrix));
for iFib=1:nFib
    RegGridMatrixInv{iFib}=RegGridMatrix{iFib}^(-1);
end
[DOFs2,InvXMat] = XInvConnectedNetwork(X,MasterConnections,...
    SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrixInv,DiffMatrix,clamp0);

% Lam=randn(size(X));
% XTLam1=XMat'*Lam;
% XTLam2=XTrConnectedNetwork(Lam,MasterConnections,SlaveConnections,...
%     Nx,nFib,L,RegGridMatrix,IntegrationMatrix,clamp0);
% max(abs(XTLam1-XTLam2))
% 
% max(abs(DOFs2-DOFs))
% max(max(abs(InvXMat*XMat-eye(size(XMat,2)))))
% Om = randn(size(DOFs));
% XInvTD = XInvTrConnectedNetwork(Om,MasterConnections,...
%     SlaveConnections,Nx,nFib,L,RegGridMatrix,DiffMatrix,clamp0);
% max(abs(XInvTD-InvXMat'*Om))

XMat = stackMatrix(XMat);
InvXMat = stackMatrix(InvXMat);

% Matrices to assign taus for the branch node
AssignMat = eye(size(DOFs,1));
for iBr=1:size(BranchIndices,1)
    AssignMat(BranchIndices(iBr,2),:)=0;
    AssignMat(BranchIndices(iBr,2),BranchIndices(iBr,1))=1;
end
IndsToDel=[];
if (clamp0)
    AssignMat(clampedTau,:)=0;
    IndsToDel=clampedTau;
end
if ~isempty(BranchIndices)
    IndsToDel=[IndsToDel;BranchIndices(:,2)];
end
AssignMat(:,IndsToDel)=[];
AssignMat=stackMatrix(AssignMat);

% Initialize X and X inverse functions
Xt = reshape(X',[],1);

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
BendMatAll = [];
BendMatHalfAll = [];
for iFib=1:nFib
    BendMatAll = blkdiag(BendMatAll, BendForceMat);
    BendMatHalfAll = blkdiag(BendMatHalfAll,BendMatHalf);
end
% Pre-computations for mobility
MobConst = -log(eps^2)/(8*pi*mu);
MobFcn = @(x1d) LocalDragMob(x1d,DX,MobConst,WTilde_Nx_Inverse);

%% Initialize arrays to save 
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-2/dt+1e-10));
Xpts=[];
ee=[];
MDDist=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sX,bX);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;

%FDAll=zeros(6*Nx-6,1);
%SDAll=zeros(6*Nx-6,1);

%% Computations
for count=0:stopcount
    t=count*dt;
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
        ee=[ee;norm(PtsThisT(1,:)-PtsThisT(Nx,:));norm(PtsThisT(Nx+1,:)-PtsThisT(2*Nx,:))];
    end  

    % Matrices at time step n 
    [K,KInv] = KWithLink(Xt,XMat,InvXMat,...
        AssignMat,BranchIndices,clampedTau);
    MWsym = zeros(nFib*3*Nx);
    RandomVelBM = zeros(nFib*3*Nx,1);
    g = randn(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymOne = MobFcn(Xt(finds));
        MWsymHalfOne = chol(MWsymOne)';
        MWsym(finds,finds)=MWsymOne;
        % Obtain Brownian velocity
        RandomVelBM(finds) = sqrt(2*kbT/dt)*MWsymHalfOne*g(finds);
    end
    
    % Advance to midpoint
    OmegaTilde = KInv*RandomVelBM;
    OmAll = AssignMat*OmegaTilde;
    Xtilde = updateByRotate(Xt,OmAll,XMat,InvXMat,dt/2,clamp0);
    [Ktilde,KInvTilde] = KWithLink(Xtilde,XMat,InvXMat,...
        AssignMat,BranchIndices,clampedTau);
    MWsymTilde = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = MobFcn(Xtilde(finds));
        MWsymTilde(finds,finds)=MWsymTildeOne;
    end
    
    %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    g3 = randn(size(AssignMat,2),1);
    OmM = AssignMat*g3;
    delta = 1e-5;
    XPlus = updateByRotate(Xt,OmM,XMat,InvXMat,delta,clamp0);
    [KPlus,KInvPlus] = KWithLink(XPlus,XMat,InvXMat,...
        AssignMat,BranchIndices,clampedTau);
    MWSymPlus = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWPlusOne = MobFcn(XPlus(finds));
        MWSymPlus(finds,finds)=MWPlusOne;
    end
    M_RFD = kbT/delta*(MWSymPlus*KInvPlus'-MWsym*KInv')*g3;

    % Solve at midpoint
    g2 = randn(3*Nx*nFib,1);
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalfAll*g2;

    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    KWithImp=Ktilde-impcoeff*dt*MWsymTilde*BendMatAll*Ktilde;
    U0 = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    %[Nxx,Ndd] = size(Ktilde);
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    %Mob_og = pinv(K'*(MWsym \ K));
    %FirstDrift = (MobK*Ktilde'*MWsymTilde^(-1)-Mob_og*K'*MWsym^(-1))*RandomVelBM;
    %FDAll=FDAll+FirstDrift;
    %SecondDrift = Mob_og*K'*MWsym^(-1)*M_RFD;
    %SDAll=SDAll+SecondDrift;
    RHSU = MWsymTilde*(BendMatAll*Xt+ Fext)+RandomVel+U0;
    RHSV = zeros(size(K,2),1);
    alphaUProj = MobK*(Ktilde'*(MWsymTilde \ RHSU)+RHSV);
    LambdaUProj = MWsymTilde \ (KWithImp*alphaUProj-RHSU);
    RHS=[RHSU;RHSV];
    alphaU = AssignMat*alphaUProj;
    %return

    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt,clamp0);
    Xt=Xp1;
end
%FDAll=FDAll/(count+1);
%SDAll=SDAll/(count+1);
Totaltime=toc(tStart);
%save(strcat('BranchedP_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
save(strcat('BranchedCL',num2str(CL),'_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end
end
end

function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat,...
    AssignMat,BranchIndices,clampedTau)
    TausAndXBar = InvXMat*Xt;
    if (clampedTau <=0)
        Tau3 = reshape(TausAndXBar(1:end-3),3,[])';
        TauVelocity = zeros(3*size(Tau3,1)+3);
        InvTauVelocity = zeros(3*size(Tau3,1)+3);
    else
        Tau3 = reshape(TausAndXBar,3,[])';
        TauVelocity = zeros(3*size(Tau3,1));
        InvTauVelocity = zeros(3*size(Tau3,1));
    end
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
        InvTauVelocity(inds,inds) = CMat;
    end
    % The COM
    if (clampedTau<=0)
        TauVelocity(end-2:end,end-2:end)=eye(3);
        InvTauVelocity(end-2:end,end-2:end)=eye(3);
    end
    KTogether = XMat*TauVelocity*AssignMat;
    KTogetherInv = AssignMat'*InvTauVelocity*InvXMat;
    OmegaFromProjections = eye(length(TausAndXBar));
    for iBr=1:size(BranchIndices,1)
        % Construct local ONB
        tauM = Tau3(BranchIndices(iBr,1),:);
        tauD = Tau3(BranchIndices(iBr,2),:);
        crossMD = cross(tauM,tauD);
        % 3 x 3 matrix going from P1Omega+P2Omega -> (Omega1,Omega2,Omega3)
        InvertMe = [tauM'-tauD'*dot(tauM,tauD) tauD'-tauM'*dot(tauD,tauM) ...
            2*crossMD'];
        ActOmega = [tauM' tauD' crossMD'];
        brInds = 3*BranchIndices(iBr,1)+(-2:0);
        OmegaFromProjections(brInds,brInds)=ActOmega*InvertMe^(-1);
    end
    brInds=[];
    for iBr=1:size(BranchIndices,1)
        brInds = [brInds;3*BranchIndices(iBr,2)+(-2:0)'];
    end
    if (clampedTau>0)
        brInds = [brInds;(3*clampedTau-2:3*clampedTau)'];
    end
    OmegaFromProjections(brInds,:)=[];
    OmegaFromProjections(:,brInds)=[];
    KTogetherInv=OmegaFromProjections*KTogetherInv;
end

function XNew = updateByRotate(Xt,alphaU,XMat,InvXMat,dt,clamp0)
    TausXBar = InvXMat*Xt;
    TausXBarNew = 0*TausXBar;
    % Update the tangent vectors
    if (~clamp0)
        Taus = reshape(TausXBar(1:end-3),3,[])';
        Omega = reshape(alphaU(1:end-3),3,[])';
        newTaus = rotateTau(Taus,Omega,dt);
        TausXBarNew(1:end-3)=reshape(newTaus',[],1);
        % Add the constant
        TausXBarNew(end-2:end)=TausXBar(end-2:end)+dt*alphaU(end-2:end);
    else
        Taus = reshape(TausXBar,3,[])';
        Omega = reshape(alphaU,3,[])';
        newTaus = rotateTau(Taus,Omega,dt);
        TausXBarNew=reshape(newTaus',[],1);
    end    
    XNew = XMat*TausXBarNew;
end


