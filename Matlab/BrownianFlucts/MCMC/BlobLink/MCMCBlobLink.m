addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
% Generate initial chain
L = 2;
kbT = 4.1e-3; % pN * um
lp = 1*L;
K_b = lp*kbT;
Nlinks = 200;
a = L/Nlinks;
ds= a;
nSamp = 1e6;
nSaveSamples = 0.8*nSamp;
nTrial = 10;
lpstar = (K_b)/kbT*1/L;
penaltyCoeff = kbT*16000/L^3;
ProposeFromC = 0;
if (penaltyCoeff>0)
    CurvedX0 = 1;
else
    CurvedX0=0;
end
xUni = (0:ds:L)';

%%% Base state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_grid = 0.5*(xUni(2:end) + xUni(1:end-1)); 
s = s_grid;
if (CurvedX0)
    q = 1; %1,3,5
    t1 = (1.0/sqrt(2.0)) * cos(q .* s.^3 .* (s - L).^3 );
    t2 = (1.0/sqrt(2.0)) * sin(q .* s.^3 .* (s - L).^3 );
    t3 = (1.0/sqrt(2.0)) + 0*s;
else
    t2 = ones(Nlinks,1);
    t1 = zeros(Nlinks,1); t3 = zeros(Nlinks,1);
end
taus = [t1 t2 t3]';
clear s;
Xs = [[0;0;0] cumsum(ds*taus,2)];
x_grid = xUni;
middleIndex = find(x_grid==L/2);
Xs = Xs-Xs(:,middleIndex);
X_Mid = [0;0;0];
AllXs = Xs';
Xp = Xs(:,1:end-1);
if (CurvedX0)
    X0 = Xs;
else
    X0 = 0*Xs;
end

u1 = [1; 0; 0];
t1 = taus(:,1);
u1 = u1 - dot(u1,t1)*t1;
u1 = u1./norm(u1);
v1 = cross(t1,u1);
[U,V] = Bishop(taus,u1);
SampleInds = [0 1/4 1/2 3/4 1]*Nlinks+1;
nBins = 1000;

% The energy matrix
N = length(taus)+1;
D4 = diff(eye(N+4),4);
D4(:,[1 2 N+3 N+4]) = [];
D4(1,1:3) = -1*[-1 2 -1];
D4(2,1:4) = -1*[2 -5 4 -1];
D4(end,end:-1:end-2) = -1*[-1 2 -1];
D4(end-1,end:-1:end-3) = -1*[2 -5 4 -1];
Wts = eye(N)*ds; Wts(1,1)=Wts(1,1)/2; Wts(end,end)=Wts(end,end)/2;
D4 = -(K_b/ds^4)*D4 - penaltyCoeff*Wts/ds; % Add penalty forcing
EMat = -ds*kron(sparse(D4),eye(3)); % FORCE not density
KT0 = KT_mat(ds,U,V,0);
K0 = full(KT0)';
if (penaltyCoeff > 0)
    C_omega = kbT*(K0'*EMat*K0)^(-1);
    C_full = kbT*EMat^(-1);
else
    C_omega = kbT*pinv(K0'*EMat*K0);
    C_full = kbT*pinv(full(EMat));
end
ComHalf = real(C_omega^(1/2));
CfullHalf = full(real(C_full^(1/2)));
C = K0*C_omega*K0';

% Compute eigenvalues to preserve L^2 norms
Wts = stackMatrix(Wts);
Ctilde = Wts^(1/2)*C*Wts^(1/2);
[DiscV,DiscL]=eig(1/2*(Ctilde'+Ctilde));
% Sort columns of V based on eigenvalues
[~,inds]=sort(diag(DiscL),'descend');
eigVL = diag(DiscL);
eigVL = eigVL(inds);
%ds = xUni(2)-xUni(1);
DiscV = DiscV(:,inds);
Vtrue = Wts^(-1/2)*DiscV;
%max(max(abs(Vtrue'*Wts*Vtrue-eye(3*N))))
%max(max(abs(Vtrue*diag(eigVL)*Vtrue'-C)))

% Propose a move around the state and evaluate its energy
EPrev = 0;
nAcc=0;
AllMeanCoeffs = zeros(nTrial,3*N);
AllMeanDevs = zeros(nTrial,1);
AllEndToEndDists = zeros(nTrial,nBins);
AllMiddleHalfDists = zeros(nTrial,nBins);
AllEndToMiddleDists = zeros(nTrial,nBins);
AllEndToQuarterDists = zeros(nTrial,nBins);
Deltas = (0:Nlinks-1)*ds;
AllTanVecDots = zeros(nTrial,length(Deltas));
AllCovMats = zeros(3*N,3*N,nTrial);


for iTrial=1:nTrial
disp(strcat('New trial = ',num2str(iTrial)))
tic
MeanSqCoeffs = zeros(3*N,1);
TanVecDots = zeros(Nlinks,1);
nSamplesDs = zeros(Nlinks,1);
MeanDev = 0;
CovMat = zeros(3*N);
for iSamp=1:nSamp
    if (ProposeFromC)
        Om = zeros(3*Nlinks,1);
        indUV = 1:3*Nlinks+3;
        indUV(4:3:end) = [];
        Om(indUV) = 0.1*ComHalf*randn(2*Nlinks+3,1);
        DZero = Om(1:3);
        Om = Om(4:end);
    else
        dXAll = reshape(0.1*CfullHalf*randn(3*N,1),3,[])';
        DZero = dXAll(1,:)';
        DTau = (dXAll(2:end,:)-dXAll(1:end-1,:))/ds;
        Om = zeros(3*Nlinks,1);
        Om(2:3:end) = sum(DTau.*U',2);
        Om(3:3:end) = sum(DTau.*V',2);       
    end
    [tauProp,UProp,VProp] = Tau_Rot_Full(1,Om,U,V,taus);
    XProp = [[0;0;0] cumsum(ds*tauProp,2)];
    XProp = XProp+Xs(:,1)+DZero;
    % Eval energy
    dX = XProp-X0;
    Energy = 1/2*reshape(dX,[],1)'*EMat*reshape(dX,[],1);
    p_acc = exp(-Energy/kbT)/exp(-EPrev/kbT);
    r=rand;
    if (r < p_acc)
        Xs=XProp;
        EPrev=Energy;
        taus = tauProp;
        U = UProp;
        V = VProp;
        nAcc = nAcc+1;
    end
    % Compute coefficients and add to array
    if (iSamp > nSamp-nSaveSamples)
        if (penaltyCoeff > 0)
            ChainCoeffs = Vtrue'*Wts*reshape(Xs-X0,[],1);
            MeanSqCoeffs = MeanSqCoeffs+ChainCoeffs.*ChainCoeffs;
            CovMat = CovMat + ChainCoeffs*ChainCoeffs';
            % Compute L^2 norm of Xs-X0
            ChainDev = reshape(Xs-X0,[],1)'*Wts*reshape(Xs-X0,[],1);
            MeanDev = MeanDev+ChainDev;
        else
            % Sample the fiber at 5 points
            KeyPoints = Xs(:,SampleInds);
            EndBinNum = min(ceil(norm(KeyPoints(:,1)-KeyPoints(:,end))/L*nBins),nBins); % [0,1000]
            AllEndToEndDists(iTrial,EndBinNum)=AllEndToEndDists(iTrial,EndBinNum)+1;
            MidBinNum = min(ceil(norm(KeyPoints(:,2)-KeyPoints(:,4))/(0.5*L)*nBins),nBins);
            AllMiddleHalfDists(iTrial,MidBinNum) = AllMiddleHalfDists(iTrial,MidBinNum)+1;
            EMidBinNum1 = min(ceil(norm(KeyPoints(:,1)-KeyPoints(:,3))/(0.5*L)*nBins),nBins); 
            EMidBinNum2 = min(ceil(norm(KeyPoints(:,5)-KeyPoints(:,3))/(0.5*L)*nBins),nBins); 
            AllEndToMiddleDists(iTrial,EMidBinNum1) = ...
                AllEndToMiddleDists(iTrial,EMidBinNum1)+1; 
            AllEndToMiddleDists(iTrial,EMidBinNum2) = ...
                AllEndToMiddleDists(iTrial,EMidBinNum2)+1;
            EQtrBinNum1 = min(ceil(norm(KeyPoints(:,1)-KeyPoints(:,2))/(0.25*L)*nBins),nBins); 
            EQtrBinNum2 = min(ceil(norm(KeyPoints(:,5)-KeyPoints(:,4))/(0.25*L)*nBins),nBins);
            AllEndToQuarterDists(iTrial,EQtrBinNum1) = ...
                AllEndToQuarterDists(iTrial,EQtrBinNum1)+1; 
            AllEndToQuarterDists(iTrial,EQtrBinNum2) = ...
                AllEndToQuarterDists(iTrial,EQtrBinNum2)+1;
            % Tangent vector dot products
            for iLink=1:Nlinks
                for jLink=iLink:Nlinks
                    index = jLink-iLink+1;
                    nSamplesDs(index)=nSamplesDs(index)+1;
                    TanVecDots(index)=TanVecDots(index)+dot(taus(:,iLink),taus(:,jLink));
                end
            end
        end
    end
end
TanVecDots = TanVecDots./nSamplesDs;
AllTanVecDots(iTrial,:) = TanVecDots;
AllMeanCoeffs(iTrial,:) = MeanSqCoeffs/nSaveSamples;
AllMeanDevs(iTrial) = MeanDev/nSaveSamples;
AllCovMats(:,:,iTrial)=CovMat/nSaveSamples;
%save(strcat('FreeUnifMCMCkbT_Lp',num2str(lpstar),'.mat'))
toc
end
%exit;




