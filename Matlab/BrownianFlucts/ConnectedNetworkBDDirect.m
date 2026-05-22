% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
%function ConnectedNetworkBDDirect(seed,Nx,dt)
%% Define constants 
addpath(genpath('../'))
seed=1;
Nx = 8;
L = 1;
ell = 0.1;
% List of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
    4 0.5 5 0 0;  1 0.9 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0.1 1; ...
    8 0.5 9 0 0; 9 0.5 10 0.1 1; 9 0.7 10 0.3 1; 2 0.1 1 0.6 1; 10 1 1 0 1; ...
    9 1 1 0.25 1];
nFib=10;
%Connections = [(1:nFib-1)' 0.8*ones(nFib-1,1) (2:nFib)' zeros(nFib-1,2)];
%Connections(2:3:end,5)=1;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;

%% Initialization
rng(seed);
impcoeff = 1;
makeMovie = 1;
dt=1e-4;
tf =25;

[DOFs,MasterConnections,SlaveConnections, ConstrainedPosNodes,...
  TangentVectorNodes,BranchIndices,IntegrationMatrix,DiffMatrix,RegGridMatrix] = ...
    InitializeConnectedNetwork(Connections,nFib,Nx,L,ell);

% Initialize X and X inverse functions
[X,XMat]=XConnectedNetwork(DOFs,MasterConnections,SlaveConnections,...
    Nx,nFib,L,RegGridMatrix,IntegrationMatrix,1);
[DOFs2,XInvMat] = XInvConnectedNetwork(X,MasterConnections,...
    SlaveConnections,Nx,nFib,L,RegGridMatrix,DiffMatrix);
XMat = stackMatrix(XMat);
InvXMat = stackMatrix(XInvMat);
% Lam=randn(size(X));
% XTLam1=XMat'*Lam;
% XTLam2=XTrConnectedNetwork(Lam,MasterConnections,SlaveConnections,...
%     Nx,nFib,L,RegGridMatrix,IntegrationMatrix);
% max(abs(XTLam1-XTLam2))
% 
% max(abs(DOFs2-DOFs))
% max(max(abs(XInvMat*XMat-eye(size(XMat,2)))))
% Om = randn(size(DOFs));
% XInvTD = XInvTrConnectedNetwork(Om,MasterConnections,...
%     SlaveConnections,Nx,nFib,L,RegGridMatrix,DiffMatrix);
% max(abs(XInvTD-XInvMat'*Om))

% Matrices to assign taus for the branch node
AssignMat = eye(size(DOFs,1));
for iBr=1:size(BranchIndices,1)
    AssignMat(BranchIndices(iBr,2),:)=0;
    AssignMat(BranchIndices(iBr,2),BranchIndices(iBr,1))=1;
end
if ~isempty(BranchIndices)
    AssignMat(:,BranchIndices(:,2))=[];
end
AssignMat=stackMatrix(AssignMat);

% Initialize X and X inverse functions
XFcn = @(dof3d) XConnectedNetwork(dof3d,MasterConnections,...
    SlaveConnections, Nx,nFib,L,RegGridMatrix,IntegrationMatrix,0);
XInvFcn = @(x3d) XInvConnectedNetwork(x3d,MasterConnections,...
    SlaveConnections,Nx,nFib,L,RegGridMatrix,DiffMatrix);
XTrFcn = @(lam3d) XTrConnectedNetwork(lam3d,MasterConnections,...
    SlaveConnections,Nx,nFib,L,RegGridMatrix,IntegrationMatrix);
XInvTrFcn = @(Om3d) XInvTrConnectedNetwork(Om3d,MasterConnections,...
     SlaveConnections,Nx,nFib,L,RegGridMatrix,DiffMatrix);
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
% ApplyBigC = @(x,Xt) ApplyBigCMatrix(x,Xt,XFcn,XInvFcn,XTrFcn,MobFcn,NodesByBranch,...
%         BendForceMat,impcoeff*dt,nFib);

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
        AssignMat,BranchIndices);
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
    % Iterative version
    OmegaTilde2 = KInvApplyConnNet(RandomVelBM,Xt,XInvFcn,BranchIndices);
    OmAll = AssignMat*OmegaTilde;
    Xtilde = updateByRotate(Xt,OmAll,XMat,InvXMat,dt/2);
    [Ktilde,KInvTilde] = KWithLink(Xtilde,XMat,InvXMat,...
        AssignMat,BranchIndices);
    MWsymTilde = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = MobFcn(Xtilde(finds));
        MWsymTilde(finds,finds)=MWsymTildeOne;
    end
    
    % Solve at midpoint
    g2 = randn(3*Nx*nFib,1);
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalfAll*g2;

    %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    g3 = randn(size(AssignMat,2),1);
    OmM = AssignMat*g3;
    delta = 1e-5;
    XPlus = updateByRotate(Xt,OmM,XMat,InvXMat,delta);
    [KPlus,KInvPlus] = KWithLink(XPlus,XMat,InvXMat,...
        AssignMat,BranchIndices);
    MWSymPlus = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWPlusOne = MobFcn(XPlus(finds));
        MWSymPlus(finds,finds)=MWPlusOne;
    end
    M_RFD = kbT/delta*(MWSymPlus*KInvPlus'-MWsym*KInv')*g3;

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
    alphaUProj = MobK*Ktilde'*(MWsymTilde \ RHSU);
    %LambdaUProj = MWsymTilde \ (KWithImp*alphaUProj-RHSU);
    U1 = Ktilde*alphaUProj;
    U2 = KApplyConnNet(alphaUProj,Xtilde,XFcn,XInvFcn,BranchIndices);
    r=randn(3*Nx*nFib,1);
    KTLam = KTApplyConnNet(r,Xtilde,XTrFcn,XInvFcn,BranchIndices);
    KTLam2 = reshape(Ktilde'*r,3,[])';
    max(abs(KTLam-KTLam2))
    KInvLam = KInvApplyConnNet(r,Xtilde,XInvFcn,BranchIndices);
    KInvLam2 = reshape(KInvTilde*r,3,[])';
    max(abs(KInvLam-KInvLam2))
    r=randn(size(K,2),1);
    KmTOm = KInvTApplyConnNet(r,Xtilde,XInvTrFcn,XInvFcn,BranchIndices);
    KmTOm2 = reshape(KInvTilde'*r,3,[])';
    max(abs(KmTOm-KmTOm2))
    alphaU = AssignMat*alphaUProj;

    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt);
    Xt=Xp1;
end
%FDAll=FDAll/(count+1);
%SDAll=SDAll/(count+1);
Totaltime=toc(tStart);
%save(strcat('BranchedP_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end

function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat,...
    AssignMat,BranchIndices)
    TausAndXBar = InvXMat*Xt;
    Tau3 = reshape(TausAndXBar(1:end-3),3,[])';
    TauVelocity = zeros(3*size(Tau3,1)+3);
    InvTauVelocity = zeros(3*size(Tau3,1)+3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
        InvTauVelocity(inds,inds) = CMat;
    end
    % The COM
    TauVelocity(end-2:end,end-2:end)=eye(3);
    InvTauVelocity(end-2:end,end-2:end)=eye(3);
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
    OmegaFromProjections(brInds,:)=[];
    OmegaFromProjections(:,brInds)=[];
    KTogetherInv=OmegaFromProjections*KTogetherInv;
end

function XNew = updateByRotate(Xt,alphaU,XMat,InvXMat,dt)
    TausXBar = InvXMat*Xt;
    TausXBarNew = 0*TausXBar;
    % Update the tangent vectors
    Taus = reshape(TausXBar(1:end-3),3,[])';
    Omega = reshape(alphaU(1:end-3),3,[])';
    newTaus = rotateTau(Taus,Omega,dt);
    TausXBarNew(1:end-3)=reshape(newTaus',[],1);
    % Add the constant
    TausXBarNew(end-2:end)=TausXBar(end-2:end)+dt*alphaU(end-2:end);
    XNew = XMat*TausXBarNew;
end


