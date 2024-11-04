addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
% Generate initial chain
L = 1;
kbT = 4.1e-3; % pN * um
lp = 2*L;
K_b = lp*kbT;
nSamp = 1e7;
nSaveSamples = 0.8*nSamp;
nTrial = 10;
N = 12; % number tangent vectors
UConst = 7.5e-3*L;
TauConst=0.04*sqrt(L/lp)*12/N;
lpstar = (K_b)/kbT*1/L;
PenaltyForceInsteadOfFlow = 0;
penaltyCoeff = 1.6e4/L^3*kbT*PenaltyForceInsteadOfFlow;
a = 1e-2;
eps=a/L;

%%% Base state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
gam0 = penaltyCoeff;
if (penaltyCoeff > 0)
    q=1; 
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
if (PenaltyForceInsteadOfFlow)
    load(strcat('CovN100KbTConst_Lp',num2str(1),'.mat'));
    eigs2nd = eigVL; Vtrue2nd=Vtrue;
    EMat2nd = EMat;
else
    Nu = 1/eps+1;
    xUni = (0:Nu-1)'*a;
end
SampleInds = [0 1/4 1/2 3/4 1]*(length(xUni)-1)+1;
ResampFromNp1 = stackMatrix(barymat(xUni,sNp1,bNp1));

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
Deltas = zeros(N);
for iPt=1:N
    for jPt=iPt:N
        Deltas(iPt,jPt)=abs(s(iPt)-s(jPt));
    end
end
Deltas = unique(Deltas(:));
AllTanVecDots = zeros(nTrial,length(Deltas));
AllCovMats = zeros(3*length(xUni),3*length(xUni),nTrial);
nBins=1000;
AllEndToEndDists = zeros(nTrial,nBins);

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
    dParams = [reshape((XsProp-Xs0)',[],1); XMP+DeltaMP-XMP0];
    Energy = 1/2*dParams'*EMatParams*dParams;
    p_acc = exp(-Energy/kbT)/exp(-EPrev/kbT);
    r=rand;
    if (r < p_acc)
        EPrev=Energy;
        X_s = XsProp;
        XMP = XMP+DeltaMP;
        nAcc = nAcc+1;
        dX = XonNp1Mat*dParams;
    end
    if (iSamp > (nSamp-nSaveSamples))
        deltaX=ResampFromNp1*dX;
        if (penaltyCoeff > 0)
            SecOrderCoeffs = Vtrue2nd'*Wts*deltaX;
            MeanSqSecCoeffs = MeanSqSecCoeffs+SecOrderCoeffs.*SecOrderCoeffs;
            CovMat = CovMat + SecOrderCoeffs*SecOrderCoeffs';
            % Compute L^2 norm of Xs-X0
            ChainDev = deltaX'*Wts*deltaX;
            MeanDev = MeanDev+ChainDev;
        else
            X3 = reshape(deltaX,3,[])';
            EEDist=norm(X3(1,:)-X3(end,:));
            EndBinNum = min(ceil(EEDist/L*nBins),nBins); % [0,1000]
            AllEndToEndDists(iTrial,EndBinNum)=AllEndToEndDists(iTrial,EndBinNum)+1;
%             % Tangent vector dot products
%             for iPt=1:N
%                 for jPt=iPt:N
%                     ds = abs(s(iPt)-s(jPt));
%                     index = find(Deltas==ds);
%                     nSamplesDs(index)=nSamplesDs(index)+1;
%                     TanVecDots(index)=TanVecDots(index)+dot(X_s(iPt,:),X_s(jPt,:));
%                 end
%             end
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
save(strcat('SpecMCMCFreeL',num2str(L),'_ConstKbT_N',num2str(N),'_Lp',num2str(lpstar),'.mat'))
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
