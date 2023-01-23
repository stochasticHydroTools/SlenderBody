% Spectral covariance calculation around a particular state
addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
kbT = 4.1e-3; % pN * um
lpstar = 1;
L = 2;
Eb = lpstar*L*kbT;
eps = 1e-3;
gam0 = kbT*16000/L^3;
SplitScheme = 1;
IdForM = 0;
impcoeff=1;
ModifyBE=1;
N = 12;
upsamp = 0;
eigThres = MEigThreshold(N,eps,L);
%load(strcat('LinRPY_OSMBE_Lp',num2str(lpstar),'_N',num2str(N),'.mat'))
loadPython=0;

CurvedX0=1;
PenaltyForceInsteadOfFlow = 1;
RectangularCollocation =0;
TimescalesLinearized;
TimescaleInds = [1 10];% 13 16 19 22 25];
dts = Timescales(TimescaleInds);
tff = 10*Timescales(1);
rng(0);
for iDt=1:length(dts)
dt = dts(iDt);
saveEvery = 1;%max(1,floor(1e-3/dt+1e-6));
% Compute modes of trajectory
nErrorBars = 5;
nTrials = 20;
% Load in the covariance from the second-order method
lpstar = Eb/kbT*1/L;
load(strcat('CovN100KbTConst_Lp',num2str(lpstar),'.mat'));
eigs2nd = eigVL; Vtrue2nd=Vtrue;
tf=tff;
for iError = 1:nErrorBars
MeanCoeffsAll = zeros(nTrials,length(eigs2nd));
for iTrial=1:nTrials
if (~loadPython)    
    ExtensionalFlow;
    nSaves = length(Xspts)/N;
end
if (loadPython)
    seedIndex = nTrials*(iError-1)+iTrial;
    FileString = strcat('Penalty_N',num2str(N),'_Lp',num2str(lpstar),...
        '_dt',num2str(log10(1/dt)),'_',num2str(seedIndex),'.txt');
    Locations = load(FileString);
    if (sum(isnan(Locations)) > 0)
        error('NaN location!')
    end
    load(strcat('X0_N',num2str(N),'.mat'));
    nSaves = length(Locations)/(N+1);
    [sNp1,~,bNp1]=chebpts(N+1,[0 L],2);
end
CoeffsAll = zeros(nSaves-1,length(eigs2nd));
ResampFromNp1 = stackMatrix(barymat(xUni,sNp1,bNp1));
for iT=2:nSaves
    if (loadPython)
        dX = reshape(Locations((iT-1)*(N+1)+1:iT*(N+1),:)',[],1)-X0;
    else
        inds = (iT-1)*N+1:iT*N;
        dParams = [reshape(Xspts(inds,:)',3*N,1)-Xs0; XMPs(iT,:)'-XMP0];
        dX = XonNp1Mat*dParams;
    end
    deltaX=ResampFromNp1*dX;
    CoeffsAll(iT-1,:) = Vtrue2nd'*Wts*deltaX;
end
StartIndex = floor(Timescales(1)/dt)+1;%/(dt*saveEvery);
MeanCoeffsAll(iTrial,:)=mean(CoeffsAll(StartIndex:end,:).*CoeffsAll(StartIndex:end,:));
end
MeanCoeffsErAll(iError,:)=mean(MeanCoeffsAll);
end
errorbar(0:length(eigs2nd)-1,mean(MeanCoeffsErAll)./eigs2nd',...
  2*(std(MeanCoeffsErAll)./eigs2nd')/sqrt(nErrorBars),'o','LineWidth',2.0)
hold on
AllCovToPlot{iDt}=MeanCoeffsErAll;
%save(strcat('LinRPYKbt_Eps3_Lp',num2str(lpstar),'_N',num2str(N),'.mat'))
end
%exit;
