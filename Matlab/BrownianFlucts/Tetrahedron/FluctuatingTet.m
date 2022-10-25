addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
% Brownian dynamics simulation of tetrahedron. We are simulating the
% equation dX = (M*F + (div M)kBT) dt + sqrt(2 kBT)*M^(1/2) dW
% Generate tetrahedron
rng(0);
AboveWall = 1;
Kone = 4000;
alpha = 1; % 0 = forward Euler, 1 = backward Euler, 1/2 = CN
nSeeds = 300;
nTrials = 300;
dtfactor = 0.25; % dt = tau_K *dtfactor
EqEvery = 1/dtfactor;
mu = 1;
MeanLength = 0.1; % of a tetrahedron side
a = MeanLength/4;
WalltoBottom = 1.5*a;
MaxMove = 0.05*a;
nP = 4;
kbT = 1e-4;
delta = 1e-4; %RFD
% Initially a parallelogram lattice with positions (-h/2,-h/2, h), (-h/2,-h/2,h),
% (h/2,h/2,h), (h/2,h/2,h)
SquareLat = [-MeanLength/2 -MeanLength/2 WalltoBottom; ...
    MeanLength/2 MeanLength/2 WalltoBottom;...
   -MeanLength/2 MeanLength/2 WalltoBottom; ...
    MeanLength/2 -MeanLength/2 WalltoBottom];
% Add some noise to get the resting positions
X = SquareLat+a*(rand(nP,3)-1/2);
Xeq = X;

% If it were rigid, what is the diffusion matrix and diffusion coeffient
% (this sets the final time)
if (AboveWall)
    Mbase = WallTransTransMatrix(X, mu, a);
else
    [Mbase, ~, ~] = getGrandMBlobs(nP,X,a,mu);
end
Kin = computeK(nP,X);
N = (Kin'*Mbase^(-1)*Kin)^(-1);
rsqCoeff = 2*kbT*trace(N(1:3,1:3));
tf = MaxMove^2/rsqCoeff;
K = Kone*ones(nP*(nP-1)/2,1);
KstrT = computeKstrT(nP,X);

% Set time step from expected timescale
tscale = (6*pi*mu*a)/max(K); % units s
dt = tscale*dtfactor;
if (kbT==0)
    tf=10*tscale;
end
tf/dt

% Calculate initial rest lengths
Connections = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
dX = X(Connections(:,1),:)-X(Connections(:,2),:);
RestLen = sqrt(sum(dX.*dX,2));
stopcount = ceil(tf/dt);
nEqs = ceil(stopcount/EqEvery);

AllStrMeans = zeros(nTrials,9);
AllStrCovar = zeros(nTrials,81);
AllMeanPositions = zeros(nTrials,3*nP*stopcount);
AllMeanEqConfs = zeros(nTrials,3*nP*nEqs);

closeToWall = zeros(nTrials,nSeeds);

for iTrial=1:nTrials
disp(strcat('New trial = ',num2str(iTrial)))
StrMean = zeros(1,9);
StrCovar = zeros(1,81);
TrialMeanPositions = zeros(1,stopcount*3*nP);
TrialMeanEqConfs = zeros(1,nEqs*3*nP);
tic

for seed=1:nSeeds
rng(seed+nSeeds*(iTrial-1));
% Sample initial configuration from GB distribution
if (kbT > 0)
    dXi = sampleInitial(nP,Connections,K,X,dX,RestLen,kbT);
else
    dXi = sampleInitial(nP,Connections,K,X,dX,RestLen,1e-4);
end
X=Xeq+dXi;

% Run dynamics and compute first and second moments of stress
MeanStress = zeros(3);
for iT=1:stopcount
    if (AboveWall && min(X(:,3) < a))
        % Too close to wall 
        warning('Beads are too close to the wall - simulation is junk!')
        closeToWall(iTrial,seed)=1;
        break;
    end
    inds = 3*nP*(iT-1)+1:iT*3*nP;
    TrialMeanPositions(:,inds)=TrialMeanPositions(:,inds)+reshape(X',1,[])/nSeeds;
    if (mod(iT-1,EqEvery)==0)
        EqNum = (iT-1)/EqEvery+1;
        XeqNow=findEqConfig(nP,X,K,RestLen,Connections,tscale);
        inds = 3*nP*(EqNum-1)+1:EqNum*3*nP;
        TrialMeanEqConfs(:,inds)=TrialMeanEqConfs(:,inds)+reshape(XeqNow',1,[])/nSeeds;
    end

    F = calculateForce(X,K,RestLen,Connections);
    str = zeros(3);
    for iP=1:nP
        str=str+X(iP,:)'*F(iP,:);
    end
    MeanStress=MeanStress+str/stopcount;
    F = reshape(F',[],1);
    if (AboveWall)
        M = WallTransTransMatrix(X, mu, a);
        WRFD = randn(nP,3);
        MPlus = WallTransTransMatrix(X+delta*a/2*WRFD, mu, a);
        MMinus = WallTransTransMatrix(X-delta*a/2*WRFD, mu, a);
        U_RFD = 1/(delta*a)*(MPlus-MMinus)*reshape(WRFD',[],1);
    else
        [M, ~, ~] = getGrandMBlobs(nP,X,a,mu);
        U_RFD = zeros(3*nP,1);
    end
    UBrown = sqrt(2*kbT/dt)*M^(1/2)*randn(3*nP,1);
%     % Add Brownian force to stress
%     FBrown = reshape(M \ UBrown,3,nP)';
%     for iP=1:nP
%         str=str+X(iP,:)'*FBrown(iP,:);
%     end
%     Stress(iT,:)=reshape(str,1,9);
    Xold=reshape(X',[],1);
    if (alpha > 0)
        dFMat = linearizedForceMatrix(X,K,RestLen,Connections);
    else
        dFMat = eye(3*nP);
    end
    X = (eye(3*nP)/dt-alpha*M*dFMat) \ ...
        (Xold/dt+M*(F-alpha*dFMat*Xold)+kbT * U_RFD + UBrown);
    % Check
%     LHS=(X-Xold)/dt;
%     RHS = M*((1-alpha)*F+alpha*(F+dFMat*(X-Xold)))+kbT * U_RFD + UBrown;
%     max(abs(LHS-RHS))
    X = reshape(X,3,nP)';
end
% Compute mean and covariance of stress
StrMean(seed,:) = reshape(MeanStress,1,9);
StrCovar(seed,:)=reshape(StrMean(seed,:)'*StrMean(seed,:),1,81);
end
toc
AllStrMeans(iTrial,:)=mean(StrMean);
AllStrCovar(iTrial,:)=mean(StrCovar);
AllMeanPositions(iTrial,:)=TrialMeanPositions;
AllMeanEqConfs(iTrial,:)=TrialMeanEqConfs;
end
MeanPos = reshape(mean(AllMeanPositions),3*nP,[])';
StdPos = reshape(std(AllMeanPositions),3*nP,[])';
DeltaMeanPos = AllMeanPositions - repmat(AllMeanPositions(:,1:3*nP),1,stopcount);
MeanDeltPos = reshape(mean(DeltaMeanPos),3*nP,[])';
StdDeltPos = reshape(std(DeltaMeanPos),3*nP,[])';
MeanEq = reshape(mean(AllMeanEqConfs),3*nP,[])';
StdEq = reshape(std(AllMeanEqConfs),3*nP,[])';
DeltaEqPos = AllMeanEqConfs - repmat(AllMeanEqConfs(:,1:3*nP),1,nEqs);
MeanDeltEqPos = reshape(mean(DeltaEqPos),3*nP,[])';
StdDeltEqPos = reshape(std(DeltaEqPos),3*nP,[])';


% tiledlayout(4,3, 'Padding', 'none', 'TileSpacing', 'compact');
% for iP=1:4
% for iD=1:3
% %subplot(4,3,3*(iP-1)+iD,'Padding', 'none');% 'TileSpacing', 'compact');
% nexttile
% errorbar((0:stopcount-1)*dt/tscale,MeanDeltPos(:,3*(iP-1)+iD),...
%    2*StdDeltPos(:,3*(iP-1)+iD)/sqrt(nTrials),'s','LineWidth',1.0)
% hold on
% errorbar((0:nEqs-1)*(dt*EqEvery)/tscale,MeanDeltEqPos(:,3*(iP-1)+iD),...
%     2*StdDeltEqPos(:,3*(iP-1)+iD)/sqrt(nTrials),'o','LineWidth',2.0)
% plot((0:stopcount-1)*dt/tscale,(0:stopcount-1)*dt*DivKNUF(iP,iD)*kbT)
% xlim([0 nEqs-1])
% if (iP==4)
%     xlabel('$t/\tau_K$')
% end
% if (iD==1)
%     ylabel('$X(t)-X(0)$')
% end
% end
% end
% legend(strcat('Position, $K=$',num2str(max(K))),strcat('Equilibrium, $K=$',num2str(max(K))),...
%     '$k_B T\nabla \cdot (KN)$')
% return;

AllStrCovar = AllStrCovar*tf/kbT;
MeanCov = triu(reshape(mean(AllStrCovar),9,9));
MeanCov = MeanCov(:); MeanCov=MeanCov(MeanCov~=0);
ErCov = triu(reshape(std(AllStrCovar),9,9));
ErCov = ErCov(:); ErCov = ErCov(ErCov~=0);
%errorbar(MeanCov,2*ErCov/sqrt(nTrials),'o','LineWidth',2.0)

MeanMean = triu(reshape(mean(AllStrMeans/kbT),3,3));
MeanMean = MeanMean(:); MeanMean=MeanMean(MeanMean~=0);
ErMean = triu(reshape(std(AllStrMeans/kbT),3,3));
ErMean = ErMean(:); ErMean = ErMean(ErMean~=0);
% errorbar(MeanMean,2*ErMean/sqrt(nTrials),'o','LineWidth',2.0)
% hold on
% xlabel('Index (1,3,6 diagonal)')
% ylabel('$\langle \sigma \rangle/k_B T$')
% return
% 
% Theoretical values (for covariance)
N_UF = (Kin'*Mbase^(-1)*Kin)^(-1);
ProjR = Mbase^(-1)*Kin*N_UF*Kin'*Mbase^(-1)-Mbase^(-1);
Cov1 = -KstrT*ProjR*KstrT'*2;
Cov1 = triu(Cov1); Cov1=Cov1(Cov1~=0);
Cov2 = KstrT*Mbase^(-1)*Kin*N_UF*Kin'*Mbase^(-1)*KstrT'*2;
Cov2 = triu(Cov2); Cov2=Cov2(Cov2~=0);

% Compute divergene of Nsf = KstrT*M^(-1)*K*N_UF
for nSamps=[5e4]
DivNSF = zeros(9,1);
DivNUF = zeros(6,1);
DivFroNSF = zeros(9,1);
DivMPos = zeros(3*nP,1);
DivKNUF = zeros(3*nP,1);
for iR=1:nSamps
WRotTransRFD = randn(6,1);
WPosRFD = randn(nP,3);
Xplus = RigidUpdate(nP,Xeq,WRotTransRFD,delta/2,MeanLength);
Xminus = RigidUpdate(nP,Xeq,WRotTransRFD,-delta/2,MeanLength);
if (AboveWall)
    MPlus = WallTransTransMatrix(Xplus, mu, a);
    MMinus = WallTransTransMatrix(Xminus, mu, a);
    MPosPlus = WallTransTransMatrix(Xeq+delta*a/2*WPosRFD, mu, a);
    MPosMinus = WallTransTransMatrix(Xeq-delta*a/2*WPosRFD, mu, a);
else
    [MPlus, ~, ~] = getGrandMBlobs(nP,Xplus, mu, a);
    [MMinus, ~, ~] = getGrandMBlobs(nP,Xminus, mu, a);
    [MPosPlus, ~, ~] = getGrandMBlobs(nP,Xeq+delta*a/2*WPosRFD, mu, a);
    [MPosMinus, ~, ~] = getGrandMBlobs(nP,Xeq-delta*a/2*WPosRFD, mu, a);
end
KPlus = computeK(nP,Xplus);
KMinus = computeK(nP,Xminus);
KstrTPlus = computeKstrT(nP,Xplus);
KstrTMinus = computeKstrT(nP,Xminus);
NUF_plus = (KPlus'*MPlus^(-1)*KPlus)^(-1);
NUF_minus = (KMinus'*MMinus^(-1)*KMinus)^(-1);
NSF_plus = KstrTPlus*MPlus^(-1)*KPlus*NUF_plus;
NSF_minus = KstrTMinus*MMinus^(-1)*KMinus*NUF_minus;
NSF_Frozen_plus = KstrTPlus*Mbase^(-1)*Kin*N_UF;
NSF_Frozen_minus = KstrTMinus*Mbase^(-1)*Kin*N_UF;
NSF_RFD = 1/delta*(NSF_plus-NSF_minus)*[WRotTransRFD(1:3)/MeanLength; WRotTransRFD(4:6)];
FroNSF_RFD = 1/delta*(NSF_Frozen_plus-NSF_Frozen_minus)...
    *[WRotTransRFD(1:3)/MeanLength; WRotTransRFD(4:6)];
DivNSF = DivNSF+NSF_RFD/nSamps;
DivFroNSF = DivFroNSF+FroNSF_RFD/nSamps;
KNUF_plus = KPlus*NUF_plus;
KNUF_minus = KMinus*NUF_minus;
KNUF_RFD = 1/delta*(KNUF_plus-KNUF_minus)*[WRotTransRFD(1:3)/MeanLength; WRotTransRFD(4:6)];
DivKNUF = DivKNUF+KNUF_RFD/nSamps;
NUF_RFD = 1/delta*(NUF_plus-NUF_minus)*[WRotTransRFD(1:3)/MeanLength; WRotTransRFD(4:6)];
DivNUF = DivNUF+NUF_RFD/nSamps;
MPos_RFD = 1/(delta*a)*(MPosPlus-MPosMinus)*reshape(WPosRFD',[],1);
DivMPos = DivMPos+MPos_RFD/nSamps;
end
DivNSF = triu(reshape(DivNSF,3,3)); DivNSF=DivNSF(DivNSF~=0);
SecondResult = KstrT*Mbase^(-1)*(DivKNUF-DivMPos);
%plot(1:9,SecondResult,'o')
%plot(1:6,DivNSF,'s')
%hold on
end
DivKNUF = reshape(DivKNUF,3,[])';
save('TetwEqK4000_Dt025_kbT4.mat')
exit;


function Kin = computeK(nP,X)
    Xc = mean(X);
    Kin = zeros(3*nP,6);
    for iP=1:nP
        Kin(3*(iP-1)+1:3*iP,1:3)=eye(3);
        Kin(3*(iP-1)+1:3*iP,4:6)=-CPMatrix(X(iP,:)-Xc);
    end
end

function KstrT = computeKstrT(nP,X)
    % Define kinematic matrix (9 x 3 Np) that acts on forces to give stress,
    % evaluted at the eq config
    KstrT = zeros(9,3*nP);
    for iP=1:nP
        KstrT(1,3*iP-2)=X(iP,1); KstrT(2,3*iP-1)=X(iP,1); KstrT(3,3*iP)=X(iP,1);
        KstrT(4,3*iP-2)=X(iP,2); KstrT(5,3*iP-1)=X(iP,2); KstrT(6,3*iP)=X(iP,2);
        KstrT(7,3*iP-2)=X(iP,3); KstrT(8,3*iP-1)=X(iP,3); KstrT(9,3*iP)=X(iP,3);
    end
end

function Xnew = RigidUpdate(nP,X,UOmega,delta,L)
    % Performs update
    % X -> Omega x X + U
    % Rotate around midpoint and translate
    XMP = mean(X);
    RotatedX=0*X;
    for iP=1:nP
        RotatedX(iP,:)=XMP+rotate(X(iP,:)-XMP,delta*UOmega(4:6)');
    end
    Xnew = RotatedX+delta*L*UOmega(1:3)';
end
