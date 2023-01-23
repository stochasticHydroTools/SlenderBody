addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
kbT = 4.1e-3;
lpstar = 1;
L = 2;
Eb = lpstar*L*kbT;
SplitScheme = 1;
impcoeff=1;
ModifyBE=1;
IdForM = 0;
eps = 1e-2;
N = 12;
upsamp = 0 ;
if (eps == 1e-3)
if (N==12)
    eigThres = 3.2/L;
    dts = [0.0301; 0.0163; 3.65e-3];
elseif (N==24)
    eigThres = 5.0/L;
    dts = [1.99e-3; 1.08e-3; 0.554e-3];
elseif (N==36)
    eigThres = 6.7/L;
    dts = [0.2245e-3; 0.1449e-3; 0.0387e-3];
end
elseif (eps==1e-2)
if (N==12)
    eigThres = 1.6/L;
    dts = [0.0665; 0.0319; 0.0100];
elseif (N==24)
    eigThres = 1.0/L;
    dts = [0.00554; 0.00314; 0.00191];
elseif (N==36)
    eigThres = 0.34/L;
    dts = [1.08e-3; 0.742e-3; 0.436e-3];
end
end
%load(strcat('EndEndkbT_Lp',num2str(lpstar),'_N_',num2str(N),'.mat'))
loadPython=0;

nTrial= 10;
nSample = 100;
CurvedX0=0;
PenaltyForceInsteadOfFlow=0;
gam0=0;
RectangularCollocation=0;
%tf = 10*Timescales(7);
mu = 1;
tf = 0.01*mu*L^4/(log(eps^(-1))*Eb);
for iDT=2
dt = dts(iDT);
saveEvery = 1;
nBins = 1000;

AllEndToEndDists = zeros(nTrial,nBins);
AllMiddleHalfDists = zeros(nTrial,nBins);
AllEndToMiddleDists = zeros(nTrial,nBins);
AllEndToQuarterDists = zeros(nTrial,nBins);

for iTrial=1:nTrial
for iSample=1:nSample
rng((iTrial-1)*nSample+iSample);
if (~loadPython)
    ExtensionalFlow;
else
    seedIndex = nSample*(iTrial-1)+iSample;
    logeps = -log10(eps);
    FileString = strcat('Eps',num2str(logeps),'.0EndToEnd_N',num2str(N),'_Lp',num2str(lpstar),...
        '.0_dt',num2str(iDT),'_',num2str(seedIndex),'.txt');
    Xpts = load(FileString);
    if (sum(isnan(Xpts)) > 0)
        error('NaN location!')
    end
    Nx = N+1;
    [sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
    nSaves = length(Xpts)/(N+1);
end
nSaves = length(Xpts)/(N+1);
StartIndex = 1;%floor(Timescales(7)/dt)+1;
saveXPerpInds = ceil([0.05*nSaves 0.1*nSaves 0.2*nSaves 0.5*nSaves nSaves]);
EndEndDists=zeros(nSaves,3);
COMvec = zeros(nSaves,3);
RotateDot = zeros(nSaves,1);
XPerpSq = zeros(Nx,length(saveXPerpInds));
for iT=StartIndex:nSaves
    inds = (iT-1)*(N+1)+1:iT*(N+1);
    KeyPoints = [barymat(0,sNp1,bNp1)*Xpts(inds,:); ...
        barymat(L/4,sNp1,bNp1)*Xpts(inds,:);
        barymat(L/2,sNp1,bNp1)*Xpts(inds,:);
        barymat(3*L/4,sNp1,bNp1)*Xpts(inds,:);
        barymat(L,sNp1,bNp1)*Xpts(inds,:)]'; 
    EndEndDists(iT,:)=KeyPoints(:,5)-KeyPoints(:,1);
    if (sum(saveXPerpInds==iT)>0)
        indT = find(saveXPerpInds==iT);
        % Save XPerp
        XPerp = Xpts(inds,:);
        XPerp(:,1)=0;
        XPerpSq(:,indT)= sum(XPerp.*XPerp,2);
    end
    COMvec(iT,:)=KeyPoints(:,3);
    RotateDot(iT)=dot(1/L*(KeyPoints(:,5)-KeyPoints(:,1)),[1;0;0]);
    EndBinNum = min(ceil(norm(KeyPoints(:,1)-KeyPoints(:,end))/L*nBins),nBins); % [0,1000]
%     if (norm(KeyPoints(:,1)-KeyPoints(:,end))/L < 0.9)
%         keyboard
%     end
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
end
%AllTrajectories{iTrial,iSample}=Xpts;
AllEndToEnds{iTrial,iSample}=EndEndDists;
AllRotateDots{iTrial,iSample}=RotateDot;
AllCOMDisps{iTrial,iSample}=COMvec;
AllXPerpSq{iTrial,iSample}=XPerpSq;
end
end
AllEndToEndsToAnalyze{iDT}=AllEndToEnds;
PlotEndToEndDists{iDT} = AllEndToEndDists;
PlotMiddleHalfDists{iDT}  = AllMiddleHalfDists;
PlotEndToMiddleDists{iDT}  = AllEndToMiddleDists;
PlotEndToQuarterDists{iDT}  = AllEndToQuarterDists;
PlotRotateDots{iDT} = AllRotateDots;
PlotCOMDisps{iDT} = AllCOMDisps;
PlotXPerpSq{iDT} = AllXPerpSq;
%save(strcat('EndEndkbT_Lp',num2str(lpstar),'_N_',num2str(N),'.mat'))
%exit;
if (upsamp==1)
    MobStr='Ref';
elseif (upsamp==-1)
    MobStr='Direct';
elseif (upsamp==-2)
    MobStr='LocDrag';
else
    MobStr='Quad';
end
logeps=-log10(eps);
save(strcat('nnRelaxing',MobStr,'N',num2str(N),'Eps',num2str(logeps),'Lp1.mat'))
end
%exit;