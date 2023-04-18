addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
% Generate initial chain
L = 2;
kbT = 4.1e-3; % pN * um
lp = 1*L;
K_b = lp*kbT;
a = 1e-2;
eps=a/L;
nSamp = 1e5;
nSaveSamples = 0.8*nSamp;
nTrial = 10;
OversampCheb = 1;
N = 16;
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
nFib=1; RectangularCollocation=0; upsamp=0; mu=1;
XMP=[0;0;0];
InitFiberVars;
load(strcat('CovN100KbTConst_Lp',num2str(1),'.mat'));
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
X = reshape(Xt,3,Nx)';
X0=X;
Xs0=X_s;
Dinv = pinv(D);
XMP0=XMP;
if (penaltyCoeff==0)
    X0=0*X0; Xs0=0*Xs0; XMP0=0*XMP0;
end
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
    XsProp = rotateTau(X_s,DTau);
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
%AllPositions(:,iTrial)=dX;
toc
save(strcat('SpecMCMCFreeConstKbT_N',num2str(N),'_Lp',num2str(lpstar),'.mat'))
end
%exit;

function newXs = rotateTau(Xsin,Omega)
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