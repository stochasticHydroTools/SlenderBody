% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
function ConnectedNetworkBDDirect(seed,Nx,dt)
%% Define constants 
addpath(genpath('../'))
%seed=30;
N = Nx-1;
%Nx = N+1;
L = 1;
ell = 0.1;
% List of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
% Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
%     4 0.5 5 0 0;  1 0.9 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0.1 1; ...
%     8 0.5 9 0 0; 9 0.5 10 0.1 1];
% Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; ...
%     4 0.5 5 0 0;  1 0.9 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0 0; ...
%     8 0.5 9 0 0; 9 0.5 10 0 0];
nFib=2;
Connections = [(1:nFib-1)' 0.8*ones(nFib-1,1) (2:nFib)' zeros(nFib-1,2)];
Connections(2:3:end,5)=1;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;

%% Initialization
rng(seed);
impcoeff = 1;
makeMovie = 0;
%dt = 1e-3;
tf = 25;

[paths,DOFs,TangentVectorNodes,IntegrationMatrix,DiffMatrix,...
    NodesByBranch] = InitializeConnectedNetwork(Connections,nFib,N,L,ell);
[sX,wX,bX]=chebpts(Nx,[0 L]);

% Initialize X and X inverse functions
[X,XMat]=XConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,DOFs,IntegrationMatrix,1);
XMat = stackMatrix(XMat);
% Alternative def of X^-1
LinkInds = find(Connections(:,5)==1);
NLinks=length(LinkInds);
InvXMat = zeros(nFib*N+NLinks+1,Nx*nFib);
for iFib=1:nFib
    InvXMat((iFib-1)*N+1:iFib*N,(iFib-1)*Nx+1:iFib*Nx)=DiffMatrix{iFib};
end
for iLink=1:NLinks
    iFib = Connections(LinkInds(iLink),1);
    iS = Connections(LinkInds(iLink),2);
    jFib = Connections(LinkInds(iLink),3);
    jS = Connections(LinkInds(iLink),4);
    InvXMat(N*nFib+iLink,(iFib-1)*Nx+1:iFib*Nx)=-barymat(iS,sX,bX)/ell;
    InvXMat(N*nFib+iLink,(jFib-1)*Nx+1:jFib*Nx)=barymat(jS,sX,bX)/ell;
end
InvXMat(end,:)=1/(nFib*L)*repmat(wX,1,nFib);
InvXMat = stackMatrix(InvXMat);
AssignMat = eye(N*nFib+NLinks+1);
for iBr=1:size(NodesByBranch,1)
    AssignMat(NodesByBranch(iBr,2),:)=0;
    AssignMat(NodesByBranch(iBr,2),NodesByBranch(iBr,1))=1;
end
if ~isempty(NodesByBranch)
AssignMat(:,NodesByBranch(:,2))=[];
end
AssignMat=stackMatrix(AssignMat);
InvAssignMat = pinv(AssignMat);
% XFcn = @(dof3d) XConnectedNetwork(Connections,nFib,N,L,ell,...
%     paths,dof3d,IntegrationMatrix,0);
% XInvFcn = @(x3d) XInvConnectedNetwork(Connections,nFib,N,L,ell,...
%     paths,x3d,DiffMatrix);
% XTrFcn = @(lams) XTrConnectedNetwork(Connections,nFib,N,L,ell,...
%         paths,lams,IntegrationMatrix);
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
    if (mod(count,saveEvery)==0)
        DOFs = InvXMat*Xt;
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
    end  

    % Matrices at time step n 
    [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat,...
        AssignMat,InvAssignMat);
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
    OmegaTilde = KTogetherInv*RandomVelBM;
    OmAll = AssignMat*OmegaTilde;
    Xtilde = updateByRotate(Xt,OmAll,XMat,InvXMat,dt/2);
    Ktilde = KWithLink(Xtilde,XMat,InvXMat,AssignMat,InvAssignMat);
    MWsymTilde = zeros(nFib*3*Nx);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = LocalDragMob(Xtilde(finds),DX,MobConst,WTilde_Nx_Inverse);
        MWsymTilde(finds,finds)=MWsymTildeOne;
    end
    
    % Solve at midpoint
    M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
    g2 = randn(3*Nx*nFib,1);
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalfAll*g2;
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    KWithImp=Ktilde-impcoeff*dt*MWsymTilde*BendMatAll*Ktilde;
    U0 = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    %[Nxx,Ndd] = size(Ktilde);
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    %RHSAll = [MWsymTilde*(BendMatAll*Xt+ Fext)+RandomVel+U0; zeros(Ndd,1)];
    alphaUProj = MobK*Ktilde'*(BendMatAll*Xt+ Fext + ...
        MWsymTilde \ (RandomVel + U0));
    %LambdaUProj = MWsymTilde \ (KWithImp*alphaUProj-RHSAll(1:Nxx));
    alphaU = AssignMat*alphaUProj;

    Xp1 = updateByRotate(Xt,alphaU,XMat,InvXMat,dt);
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('BranchedN_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts','MDDist')
end

function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat,...
    AssignMat,InvAssignMat)
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
    KTogetherInv = InvAssignMat*InvTauVelocity*InvXMat;
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


