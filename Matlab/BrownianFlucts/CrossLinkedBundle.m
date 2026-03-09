%function FluctClamped_AddDOFs(seed,ForceRt,N,dt,gtype)
% Single fluctuating clamped filament
seed=1;
N=16;
dt=1e-3;
gtype=1;
addpath(genpath('../'))
%close all;
rng(seed);
nFib = 2;
Nc_offgrid = 0; % Number of off-grid tangent vector constraints 
Nog = N - Nc_offgrid;
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie = 1;
tf = 5;
Tau0BC = [0;1;0];
TrkLoc = L/2;
XTrk=[0 0.1;TrkLoc TrkLoc;0 0];
[sog,~,~] = chebpts(Nog, [0 L], gtype);
Xs3=repmat(Tau0BC',nFib*N,1);
% Add rows for the constraints 
[s,~,b] = chebpts(N,[0 L], gtype);
sC=s;
ChebToConstr = eye(N);
ConstrToCheb = eye(N);
Nx = N + 1;
[sNp1,~,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*Nx,3);
for iR=1:Nx
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
XonNp1Mat = [(eye(3*Nx)-repmat(BMNp1,Nx,1))*...
    stackMatrix(IntDNp1*RToNp1*ConstrToCheb) I];
InvXonNp1Mat = [stackMatrix(ConstrToCheb \ RNp1ToN*DNp1); BMNp1];

% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, ~] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
BendMatHalf_Np1 = real(BendingEnergyMatrix_Np1^(1/2));

Xt = zeros(3*nFib*Nx,1);
for iFib=1:nFib
    Xt(3*Nx*(iFib-1)+1:3*Nx*iFib) = XonNp1Mat* [reshape(Xs3(N*(iFib-1)+1:N*iFib,:)',[],1);XTrk(:,iFib)];
end
saveEvery=max(1,floor(1e-2/dt+1e-10));
MobConst = -log(eps^2)/(8*pi*mu);
ConnectedMat = [stackMatrix([barymat(TrkLoc,sNp1,bNp1)]) -stackMatrix([barymat(TrkLoc,sNp1,bNp1)])];
Constr=zeros(1,1);

%% Initialization 
stopcount=floor(tf/dt+1e-5);
ConstrErs = zeros(stopcount,1);
Xpts=[];
ee=[];
Npl=100;
[spl,~,~]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
if (makeMovie)
    close all;
    f=figure;
    frameNum=0;
end
tStart=tic;
nConstr=1;

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        %t
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            linkPts=[PtsThisT(9,:); PtsThisT(26,:)];
            plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko')
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 2])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(end,:))];
    end
    ConsMat = Xt'*ConnectedMat'*ConnectedMat;
    % Evolve system
    MWsymTilde = zeros(nFib*3*Nx);
    Ktilde = zeros(nFib*3*Nx);
    KWithImp = zeros(nFib*3*Nx);
    RandomVel = zeros(nFib*3*Nx,1);
    BendForce = zeros(nFib*3*Nx,1);
    TrkPts = zeros(3*nFib,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        XTrk = XsXTrk(end,:)';
        TrkPts(3*iFib-2:3*iFib)=XTrk;
        Xs3 = XsXTrk(1:Nx-1,:);
        MWsym = LocalDragMob(Xt(finds),DNp1,MobConst,WTilde_Np1_Inverse);
        MWsymHalf = real(MWsym^(1/2));
        % Obtain Brownian velocity
        g = randn(3*Nx,1);
        RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
        % Advance to midpoint
        OmegaTilde = cross(Xs3,ChebToConstr*RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
        Xstilde = rotateTau(Xs3,OmegaTilde(1:Nx-1,:),dt/2);
        KtildeOne = KonNp1(Xstilde,XonNp1Mat,I);
        Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);XTrk];
        MWsymTildeOne = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
        M_RFD = (MWsymTildeOne-MWsym)*(MWsym \ RandomVelBM);
        RandomVelBE = sqrt(kbT)*...
             MWsymTildeOne*BendMatHalf_Np1*randn(3*Nx,1);
        RandomVel(finds) = RandomVelBM + M_RFD + RandomVelBE;
        Ktilde(finds,finds)=KtildeOne;
        KWithImp(finds,finds)=KtildeOne-impcoeff*dt*MWsymTildeOne*BendForceMat*KtildeOne;
        MWsymTilde(finds,finds)=MWsymTildeOne;
        BendForce(finds)=BendForceMat*Xt(finds);
    end
    U0 = zeros(3*Nx*nFib,1);
    Fext = zeros(3*Nx*nFib,1);
    MobK = pinv(Ktilde'*(MWsymTilde \ KWithImp));
    MobC = ConsMat'*pinv(ConsMat*Ktilde*MobK*Ktilde'*ConsMat')*ConsMat;
    alphaU = (MobK*Ktilde' - ...
        MobK*Ktilde'*MobC*Ktilde*MobK*Ktilde')*...
        (BendForce+ Fext + MWsymTilde \ (RandomVel + U0));
    % Mat2 = [-MWsymTilde KWithImp zeros(3*Nx,nConstr); ...
    %     Ktilde' zeros(3*Nx) Ktilde'*ConsMat'; ...
    %     zeros(nConstr,3*Nx) ConsMat*Ktilde zeros(nConstr)];
    % RHS2 = [MWsymTilde*(BendForceMat*Xt + Fext) + RandomVel+U0; ...
    %     zeros(3*Nx+nConstr,1)];
    % Sol2 = pinv(Mat2)*RHS2;
    % Lambda = Sol2(1:3*Nx);
    % alphaU1 = Sol2(3*Nx+1:6*Nx);
    % Gamma = Sol2(6*Nx+1:end);
    % if (max(abs(alphaU1-alphaU))>0.5)
    %     keyboard
    % end
    % Evolve constants by rotating and translating the link
    linkpts = [TrkPts(1:3)'; TrkPts(4:6)'];
    r = linkpts(2,:)-linkpts(1,:);
    u1u2 = [alphaU(3*Nx-2:3*Nx)'; alphaU(end-2:end)'];
    xc = mean(linkpts);
    uc = mean(u1u2);
    xc=xc+dt*uc;
    ud = u1u2(2,:)-u1u2(1,:);
    OmLink=cross(r,ud)/norm(r)^2;
    % Compute matrix representation you are using to get U and Omega
    P = zeros(6,nFib*3*Nx);
    for d=1:3
        for iFib=1:nFib
            P((iFib-1)*3+d,3*Nx*iFib-3+d)=1;
        end
    end
    AvgMat = zeros(3,6);
    for d=1:3
        AvgMat(d,d:3:end)=1/2;
    end
    DiffMat = zeros(3,6);
    for d=1:3
        DiffMat(d,d:3:end)=1;
    end
    DiffMat(:,4:6)=-1*DiffMat(:,4:6);
    OmLinkMat = eye(6);
    OmLinkMat(4:6,4:6)=-CPMatrix(r)/norm(r)^2;
    UcUDiff = [AvgMat;DiffMat]*P*alphaU;
    TotalMatGetOmUcMat = OmLinkMat*[AvgMat;DiffMat]*P;
    OmUc = TotalMatGetOmUcMat*alphaU;
    linkHat = rotateTau(r,OmLink,dt);
    linkpts_og=linkpts;
    linkpts(1,:)=xc-linkHat/2;
    linkpts(2,:)=xc+linkHat/2;
    Xp1=0*Xt;
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        XsXTrk = reshape(InvXonNp1Mat*Xt(finds),3,Nx)';
        Xs3 = XsXTrk(1:Nx-1,:);
        Omega = reshape(alphaU((iFib-1)*3*Nx+1:iFib*3*Nx-3),3,N)';
        newXs = rotateTau(Xs3,Omega,dt);
        Xsp1 = reshape(newXs',[],1);
        XTrk_p1 = linkpts(iFib,:)';
        Xp1(finds) = XonNp1Mat*[Xsp1;XTrk_p1];
    end
    % Correct the constant
    % Solve for minimum Omega s.t. you get back on constraint
    ConstrErs(count+1) = abs(norm(ConnectedMat*Xp1)-0.1); 
    Xt=Xp1;
end
ConstrErs=ConstrErs(1:saveEvery:end);
Totaltime=toc(tStart);
%save(strcat('CDOFType',num2str(gtype),'_N',num2str(N),'_Dt',num2str(dt),'_Seed',num2str(seed),'.mat'))
%end