function BranchedNetworkPenalty(seed,Nx,dt,Kstiff,Kang)
% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
%seed=1;
%Nx=8;
%dt=1e-3;
%Kstiff=10;
%Kang=0.002;
gtype=1;
addpath(genpath('../'))
BranchLoc = 0.8;
%close all;
rng(seed);
nFib=2;
Connections = [(1:nFib-1)' BranchLoc*ones(nFib-1,1) (2:nFib)' zeros(nFib-1,2)];
%Connections(3:3:end,5)=Connections(3:3:end,5)+1;
NLinks = sum(Connections(:,5));
NBranch = length(Connections(:,5))-NLinks;
N = Nx - 1; % Number of off-grid tangent vector constraints 
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 0.6;
impcoeff = 1;
makeMovie =0;
tf = 25;
Kcl=Kstiff;
ell = 0;
RotAng = 70/180*pi;

[paths,DOFs,TangentVectorNodes,IntegrationMatrix,DiffMatrix,...
    NodesByBranch,PairwiseXMats] = ...
    InitializeConnectedNetwork(Connections,nFib,N,L,ell);
X=XConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,DOFs,IntegrationMatrix,1);
Xt = reshape(X',[],1);

% Chebyshev grids
[s,~,b] = chebpts(N,[0 L], gtype);
[sNx,wNx,bNx]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNx,s,b);
RNp1ToN = barymat(s,sNx,bNx);
IntDX = pinv(DX);
AvgMat = 1/L*stackMatrix(wNx); % Average matrix
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=repmat(eye(3),Nx,1);
XonNp1Mat = [(eye(3*Nx)-I*AvgMat)*...
    stackMatrix(IntDX*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DX); AvgMat];

% Bending energy matrix (2Nx grid)
[s2Nx, w2x, ~] = chebpts(2*Nx, [0 L], 2);
W2Nx = diag(w2x);
R_Nx_To_2Nx = barymat(s2Nx,sNx,bNx);
WTilde_Nx = stackMatrix((R_Nx_To_2Nx'*W2Nx*R_Nx_To_2Nx));
WTilde_Nx_Inverse = WTilde_Nx^(-1);
BendingEnergyMatrix_Nx = Eb*stackMatrix(DX^2)'*WTilde_Nx*stackMatrix(DX^2);
BendForceMat = -BendingEnergyMatrix_Nx;
BendMatHalf = real(BendingEnergyMatrix_Nx^(1/2));

Nuni=11;
su=(0:0.1:1)'*L;
Runi = barymat(su,sNx,bNx);
linkPt = find(su==BranchLoc);
links = [Nuni*(0:nFib-2)'+linkPt Nuni*(1:nFib-1)'+1 zeros(nFib-1,3)];

saveEvery=max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
Xbars=[];
MDDist=[];
LinkErs=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNx,bNx);
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
        %t
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        [~,X1stars,X2stars] = getCLforceEn(links,reshape(Xt,3,Nx*nFib)',Runi, Kcl, ell*ones(NBranch,1),0,0);
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            for pL = 1:NBranch
               linkPts=[X1stars(pL,:); X2stars(pL,:)];
               plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko')
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            % ylim([-1 1])
            % xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        % Find the closest pt on the branch to the end of the mother
        MotherEnd = barymat(L,sNx,bNx)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sNx,bNx)*PtsThisT(Nx+1:2*Nx,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        LinkNorms = sqrt(sum((X1stars-X2stars).*(X1stars-X2stars),2))-ell;
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
        MDDist=[MDDist; dispMDT];
    end  
    
    % Cross linking force
    X3 = reshape(Xt,3,Nx*nFib)';
    [CLForce,~,~] = getCLforceEn(links,X3,Runi, Kcl, ell*ones(NBranch,1),0,0);
    AngCLForce=0*X3;
    for iBr=1:size(Connections,1)
        iFib = Connections(iBr,1);
        iInds=(iFib-1)*Nx+1:iFib*Nx;
        iS = Connections(iBr,2);
        jFib = Connections(iBr,3);
        jInds = (jFib-1)*Nx+1:jFib*Nx;
        Xpair = [X3(iInds,:); X3(jInds,:)];
        [ThisAngCLForce,~] = AngularSpringForce(Xpair,Kang,RotAng,...
            barymat(iS,sNx,bNx),barymat(0,sNx,bNx),DX);
        AngCLForce([iInds';jInds'],:)=AngCLForce([iInds';jInds'],:)+ThisAngCLForce;
    end

    Xp1=Xt;
    U0 = zeros(3*Nx*nFib,1);
    Fext = reshape(CLForce'+AngCLForce',[],1);
    % Matrices at time step n 
    gAll = randn(3*Nx*nFib,1);
    BEAll = randn(3*Nx*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXbar = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        Xs3 = XsXbar(1:Nx-1,:);
        MWsym = LocalDragMob(Xt(finds),DX,MobConst,WTilde_Nx_Inverse);
        MWsymHalf = chol(MWsym)';
        % Obtain Brownian velocity
        g = gAll(finds);
        RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
        TauVelocity = zeros(3*N+3);
        % The matrix for all the taus (incl links) to evolve
        for iR =1:size(Xs3,1)
            inds = (iR-1)*3+1:iR*3;
            CMat = CPMatrix(Xs3(iR,:));
            TauVelocity(inds,inds) =  -CMat;
        end
        TauVelocity(end-2:end,end-2:end)=eye(3);
        % The COM
        KInv = -TauVelocity*InvXonNp1Mat;

        OmegaTilde = cross(Xs3,RNp1ToN*DX*reshape(RandomVelBM,3,[])');
        Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
        Xdr = XsXbar(end,:)'+dt/2*AvgMat*RandomVelBM;
        Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
        Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);Xdr];
        MWsymTilde = LocalDragMob(Xtilde,DX,MobConst,WTilde_Nx_Inverse);
 
        % Solve at midpoint
        %M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
        g3 = randn(3*N+3,1);
        OmRFD = g3;
        delta = 1e-5;
        XsPlus = rotateTau(Xs3,reshape(OmRFD(1:3*N),3,[])',delta);
        XPlus = XonNp1Mat*[reshape(XsPlus',[],1); zeros(3,1)];
        MWSymPlus = LocalDragMob(XPlus,DX,MobConst,WTilde_Nx_Inverse);
        M_RFD = kbT/delta*(MWSymPlus-MWsym)*KInv'*g3;
        if (impcoeff==1)
            RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf*BEAll(finds);
        else
            RandomVelBE = 0;
        end
        RandomVel = RandomVelBM + M_RFD + RandomVelBE;
        KWithImp=Ktilde-impcoeff*dt*MWsymTilde*BendForceMat*Ktilde;
        MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
        alphaU = MobK*Ktilde'*(BendForceMat*Xt(finds)+ Fext(finds) + MWsymTilde \ (RandomVel + U0(finds)));
        Omega = reshape(alphaU(1:3*(Nx-1)),3,Nx-1)';
        newXs = rotateTau(Xs3,Omega,dt);
        Xsp1 = reshape(newXs',[],1);
        XBR_p1 = XsXbar(end,:)'+dt*alphaU(end-2:end);
        Xp1(finds) = XonNp1Mat*[Xsp1;XBR_p1];
    end
    Xt=Xp1;
end
Totaltime=toc(tStart);
save(strcat('BranchK',num2str(Kcl),'Kang',num2str(Kang),'_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'),'Xpts','MDDist','LinkErs')
end