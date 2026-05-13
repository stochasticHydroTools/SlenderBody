function BranchedFibers_NotConstr(seed,Nx,dt,Kstiff,Kang)
% Fluctuating bundle of cross-linked filaments with Nlinks at arbitrary
% locations
%seed=1;
%Nx=16;
%dt=1e-5;
%Kstiff=1000;
%Kang=0.2;
gtype=1;
addpath(genpath('../../'))
BranchLoc = 0.8;
nBr=length(BranchLoc);
%close all;
rng(seed);
nFib = 2;
N = Nx - 1; % Number of off-grid tangent vector constraints 
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
lp = 2*L;
Eb = lp*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
impcoeff = 1;
makeMovie = 1;
tf = 25;
Kcl=Kstiff;
Tau0 = [0 1 0];
RotAng = 70/180*pi;
TauBr = Tau0*[cos(RotAng) -sin(RotAng) 0; sin(RotAng) cos(RotAng) 0; 0 0 1]';
Xbar = [0.234923155196478; -0.235505035831418; 0];
Xbar(:,2)=-Xbar(:,1);
Xs3=[repmat(Tau0,N,1);repmat(TauBr,N,1)];
ell = 0;

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
links = [find(su==BranchLoc) Nuni+1 zeros(nBr,3)];

Xt = zeros(3*nFib*Nx,1);
for iFib=1:nFib
    Xt(3*Nx*(iFib-1)+1:3*Nx*iFib) = XonNp1Mat* [reshape(Xs3(N*(iFib-1)+1:N*iFib,:)',[],1);Xbar(:,iFib)];
end
saveEvery=max(1,floor(1e-4/dt+1e-10));
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
        DOFs = blkdiag(InvXonNp1Mat(1:3:end,1:3:end),InvXonNp1Mat(1:3:end,1:3:end))*PtsThisT;
        BrTaus = [barymat(BranchLoc,s,b)*DOFs(1:N,:); barymat(0,s,b)*DOFs(Nx+1:Nx+N,:)];
        [F,theta] = AngularSpringForce(PtsThisT,Kang,RotAng,barymat(BranchLoc,sNx,bNx),barymat(0,sNx,bNx),DX);
        Xbar = 1/2*(DOFs(Nx,:)+DOFs(2*Nx,:));
        MPs = blkdiag(barymat(L/2,sNx,bNx),barymat(L/2,sNx,bNx))*PtsThisT;
        [~,X1stars,X2stars] = getCLforceEn(links,reshape(Xt,3,Nx*nFib)',Runi, Kcl, ell*ones(nBr,1),0,0);
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                plot3(RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,1),RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,2),...
                    RplNp1*PtsThisT((iFib-1)*Nx+1:iFib*Nx,3));
                hold on
            end
            for pL = 1:nBr
               linkPts=[X1stars(pL,:); X2stars(pL,:)];
               plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko')
            end
            title(sprintf('$t=$ %2.1f',(frameNum-1)*saveEvery*dt),'Interpreter','latex')
            view(2)
            ylim([-1 1])
            xlim([-1 1])
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        % Find the closest pt on the branch to the end of the mother
        MotherEnd = barymat(L,sNx,bNx)*PtsThisT(1:Nx,:);
        DaughterPts = barymat((0:0.001:1)',sNx,bNx)*PtsThisT(Nx+1:end,:);
        dispMD = DaughterPts - MotherEnd;
        [dispMDT,cpt] = min(sqrt(sum(dispMD.*dispMD,2)));
        LinkNorms = sqrt(sum((X1stars-X2stars).*(X1stars-X2stars),2))-ell;
        Xpts=[Xpts;PtsThisT];
        ee=[ee;norm(PtsThisT(end-Nx,:)-PtsThisT(1,:)); norm(PtsThisT(end,:)-PtsThisT(end-Nx+1,:))];
        MDDist=[MDDist; dispMDT];
        LinkErs = [LinkErs;LinkNorms theta*180/pi];
        Xbars = [Xbars; Xbar];
    end  
    
    % Cross linking force
    [CLForce,~,~] = getCLforceEn(links,reshape(Xt,3,Nx*nFib)',Runi, Kcl, ell*ones(nBr,1),0,0);
    [AngCLForce,~] = AngularSpringForce(reshape(Xt,3,Nx*nFib)',Kang,RotAng,...
        barymat(BranchLoc,sNx,bNx),barymat(0,sNx,bNx),DX);
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
        OmegaTilde = cross(Xs3,RNp1ToN*DX*reshape(RandomVelBM,3,[])');
        Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
        Xdr = XsXbar(end,:)'+dt/2*AvgMat*RandomVelBM;
        Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
        Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);Xdr];
        MWsymTilde = LocalDragMob(Xtilde,DX,MobConst,WTilde_Nx_Inverse);
 
        % Solve at midpoint
        M_RFD = (MWsymTilde-MWsym)*(MWsym \ RandomVelBM);
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