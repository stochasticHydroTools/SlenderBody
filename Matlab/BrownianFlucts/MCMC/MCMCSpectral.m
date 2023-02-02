addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
% Generate initial chain
L = 2;
kbT = 4.1e-3; % pN * um
lp = 1*L;
K_b = lp*kbT;
a = 1e-2;
nSamp = 1e6;
nSaveSamples = 0.8*nSamp;
nTrial = 10;
OversampCheb = 1;
N = 12;
UConst = 7.5e-3*L;
TauConst=0.04*sqrt(L/lp)*12/N;
lpstar = (K_b)/kbT*1/L;
PenaltyForceInsteadOfFlow = 0;
penaltyCoeff = 1.6e4/L^3*kbT*PenaltyForceInsteadOfFlow;

%%% Base state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
q=1; 
gam0 = penaltyCoeff;
if (penaltyCoeff > 0)
    X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
else
    X_s = [ones(N,1) zeros(N,2)];
end
Lmat = cos(acos(2*s/L-1).*(0:N-1));
D = diffmat(N, 1, [0 L], 'chebkind1');
Eb=K_b;           % Bending modulus
mu=1; deltaLocal=0; nFib=1; clamp0=0; clampL=0; dt=1; tf=1; ...
    twmod=0;strongthetaBC=0; makeMovie=0; theta0=zeros(N,1);
fibpts = pinv(D)*X_s;
fibpts = fibpts-barymat(L/2,s,b)*fibpts;
InitFiberVars;

load(strcat('CovN100KbTConst_Lp',num2str(lpstar),'.mat'));
ResampForC = stackMatrix(barymat(xUni,s,b));
ResampFromDouble = stackMatrix(barymat(xUni,sOversamp,bOversamp));
ResampFromNp1 = stackMatrix(barymat(xUni,sNp1,bNp1));
eigs2nd = eigVL; Vtrue2nd=Vtrue;
EMat2nd = EMat;
SampleInds = [0 1/4 1/2 3/4 1]*(length(xUni)-1)+1;
nBins = 1000;


% The energy matrix and expected covariance
EMatParams = XonNp1Mat'*EMat_Np1*XonNp1Mat;

% Propose a move around the state and evaluate its energy
EPrev = 0;
nAcc=0;
X = fibpts;
X0=fibpts;
Xs0=X_s;
X0Double = reshape((RNToOversamp*X0)',[],1);
X0_Np1 = reshape((RToNp1*X0)',[],1);
X0=reshape(X0',[],1);
Dinv = pinv(D);
BM1 = BM(1:3:end,1:3:end);
XMP = (BM1*X)';
XMP0=XMP;
if (penaltyCoeff==0)
    X0=0*X0; Xs0=0*Xs0; X0Double=0*X0Double; X0_Np1 = 0*X0_Np1; XMP0=0*XMP0;
end
BMoversamp = stackMatrix(barymat(L/2,sOversamp,bOversamp));
AllMeanCoeffs = zeros(nTrial,3*N);
AllMeanSecCoeffs = zeros(nTrial,3*length(xUni));
AllMeanDevs = zeros(nTrial,1);
AllEndToEndDists = zeros(nTrial,nBins);
AllMiddleHalfDists = zeros(nTrial,nBins);
AllEndToMiddleDists = zeros(nTrial,nBins);
AllEndToQuarterDists = zeros(nTrial,nBins);
Deltas = zeros(N);
for iPt=1:N
    for jPt=iPt:N
        Deltas(iPt,jPt)=abs(s(iPt)-s(jPt));
    end
end
Deltas = unique(Deltas(:));
AllTanVecDots = zeros(nTrial,length(Deltas));
AllCovMats = zeros(3*length(xUni),3*length(xUni),nTrial);

for iTrial=1:nTrial
disp(strcat('New trial = ',num2str(iTrial)))
tic
MeanSqCoeffs = zeros(3*N,1);
TanVecDots = zeros(length(Deltas),1);
nSamplesDs = zeros(length(Deltas),1);
MeanSqSecCoeffs = zeros(3*length(xUni),1);
MeanDev = 0;
CovMat = zeros(3*length(xUni));
for iSamp=1:nSamp
    DTau = TauConst*randn(N,3);
    DeltaMP = UConst*randn(3,1);
    XsProp = rotateTau(X_s,DTau,Dinv,BM1);
    % Eval energy
    %dX = XProp-X0;
    % Upsample and integrate, then upsample again
    if (OversampCheb)
        dParams = [reshape((XsProp-Xs0)',[],1); XMP+DeltaMP-XMP0];
        Energy = 1/2*dParams'*EMatParams*dParams;
    else
        XProp = reshape((IntMat*XsProp)',[],1);
        XProp = XProp+repmat(XMP+DeltaMP-BM*XProp,N,1);
        dX = XProp-X0;
        Energy=1/2*dX'*EMat*dX;
    end
    p_acc = exp(-Energy/kbT)/exp(-EPrev/kbT);
    r=rand;
    if (r < p_acc)
        EPrev=Energy;
        X_s = XsProp;
        XMP = XMP+DeltaMP;
        nAcc = nAcc+1;
        if (OversampCheb)
            dX = XonNp1Mat*dParams;
            deltaX=ResampFromNp1*dX;
        else 
            deltaX = dX;
            if (~SecondOrderEnergy)
                deltaX = ResampForC*dX;
            end
        end
    end
    if (iSamp > nSamp-nSaveSamples)
        if (penaltyCoeff > 0)
            SecOrderCoeffs = Vtrue2nd'*Wts*deltaX;
            MeanSqSecCoeffs = MeanSqSecCoeffs+SecOrderCoeffs.*SecOrderCoeffs;
            CovMat = CovMat + SecOrderCoeffs*SecOrderCoeffs';
            % Compute L^2 norm of Xs-X0
            ChainDev = deltaX'*Wts*deltaX;
            MeanDev = MeanDev+ChainDev;
        else
            X3 = reshape(deltaX,3,[]);
            KeyPoints = X3(:,SampleInds);
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
            for iPt=1:N
                for jPt=iPt:N
                    ds = abs(s(iPt)-s(jPt));
                    index = find(Deltas==ds);
                    nSamplesDs(index)=nSamplesDs(index)+1;
                    TanVecDots(index)=TanVecDots(index)+dot(X_s(iPt,:),X_s(jPt,:));
                end
            end
        end
    end
end
TanVecDots = TanVecDots./nSamplesDs;
AllTanVecDots(iTrial,:) = TanVecDots;
AllMeanCoeffs(iTrial,:) = MeanSqCoeffs/nSaveSamples;
AllMeanSecCoeffs(iTrial,:)=MeanSqSecCoeffs/nSaveSamples;
AllMeanDevs(iTrial) = MeanDev/nSaveSamples;
AllCovMats(:,:,iTrial)=CovMat/nSaveSamples;
AllPositions(:,iTrial)=dX;
toc
%save(strcat('SpecMCMCFreeConstKbTFull_N',num2str(N),'_Lp',num2str(lpstar),'.mat'))
end
%exit;

function newXs = rotateTau(Xsin,Omega,Dinv,BM)
    nOm = sqrt(sum(Omega.*Omega,2));
    % Have to truncate somewhere to avoid instabilities
    k = Omega./nOm;
    k(nOm < 1e-12,:) = 0;
    % Rodriguez formula on the N grid. 
    newXs = Xsin.*cos(nOm)+cross(k,Xsin).*sin(nOm)...
        +k.*sum(k.*Xsin,2).*(1-cos(nOm));
    %newX = Dinv*newXs;
    %newX = newX-BM*newX+NewMP;
end