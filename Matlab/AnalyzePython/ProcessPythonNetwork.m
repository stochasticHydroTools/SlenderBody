addpath(genpath('../../Python'))
names = ...%["UnifStericsBundlingLp2.0Dt_5e-05_" "SegStericsBundlingLp2.0Dt_5e-05_" ...
    ["UnifStericsBundlingLp2.0Dt_2e-05_" "SegStericsBundlingLp2.0Dt_2e-05_" ...
    "UnifStericsBundlingLp2.0Dt_1e-05_" "SegStericsBundlingLp2.0Dt_1e-05_" ...
    "NoStericsBundlingLp2.0Dt_0.0001_"];
%"UnifStericsBundlingLp2.0Dt_1e-05_" "SegStericsBundlingLp2.0Dt_1e-05_" ...  
%names = ["DetN12IFQuadBundLp17.0Dt_0.0001_" "DetN12IFOSBundLp17.0Dt_0.0001_"];
tmaxes = 8*ones(1,length(names));
dtsaves = 2e-2*ones(1,length(names));
clear nLinks nBund nInBund MBAlign MaxBundSize meanDispP meanCurvatures meanBund
if (exist('parSet','var'))
else
    parSet=1;
end
F = 200;
L = 1;
Ld = 2;
plots = 1;
eebinwidths = [0.02 0.001 0.001 0.001 0.001 0.001];
AllLinksPerFib = [];
for iName=1:length(names)
seedmax=5;
if (iName>0)
N=13;
[s,w,b]=chebpts(N,[0 L],2);
else
N=16;
[s,w,b]=chebpts(N,[0 L],1);  
end
[sD,wD,bD]=chebpts(2*N,[0 L],2);
Ext = barymat(sD,s,b);
COMsample = barymat(L/2,s,b);
for seed=1:seedmax
FileName = strcat(names(iName),num2str(seed),'.txt'); % ALWAYS LOOK @ MOVIE!
Locs=load(strcat('Locs',FileName));
%writematrix(Locs(end-N*F+1:end,:),strcat('FinalLocs',FileName),'Delimiter','tab');
% Separate bundle
BundleOrders_Sep = load(strcat('BundleOrderParams_Sep',FileName));
NPerBundle_Sep = load(strcat('NFibsPerBundle_Sep',FileName));
NBundlesPerstep_Sep = load(strcat('NumberOfBundles_Sep',FileName));
BundDensity=NBundlesPerstep_Sep/Ld^3;
BundStart = [0; cumsum(NBundlesPerstep_Sep)];
% Fiber info
try
FiberCurves = load(strcat('FibCurvesF',FileName));
catch
FiberCurves = load(strcat('FibCurves',FileName));
end
try 
nContacts = load(strcat('nContacts',FileName));
catch
nContacs = 0;
end
LocalFibAlignment = load(strcat('LocalAlignmentPerFib',FileName));
nFibsConnected = load(strcat('NumFibsConnectedPerFib',FileName));
% Link info
LinkStrains = load(strcat('LinkStrains',FileName));
max(LinkStrains(2:end))
try
NLinksPerFib = load(strcat('nLinksPerFib',FileName));
catch
end
nLinksPerT = sum(NLinksPerFib')/2;
[~,numTs]=size(nLinksPerT);
startind = [0 cumsum(nLinksPerT)];
% Labels for visualization
labels = load(strcat('FinalLabels_Sep',FileName));
[nts,~] = size(labels);
% InBundle = zeros(F,nts);
% for iT=1:nts
%     lab=1;
%     newlabels = zeros(F,1);
%     for iFib=1:F
%         if (newlabels(iFib)==0)
%             iFiblab = labels(iT,iFib);
%             if (sum(labels(iT,:)==iFiblab)==1)
%                 newlabels(iFib)=-1;
%             else
%                 newlabels(labels(iT,:)==iFiblab,iT)=lab;
%                 lab=lab+1;
%                 InBundle(iFib,iT)=1;
%             end
%         else
%             InBundle(iFib,iT)=1;
%         end
%     end
% end
% 
freefibval = 1;
lag=1;
StepDisplacements = zeros(F,numTs-lag);
EndEndDistances = zeros(F,numTs);
MeanInBundleDisp = zeros(numTs-lag,1);
MeanOutofBundleDisp = zeros(numTs-lag,1);
step=N*F;
for iT=1:numTs-1
    XThisT = Locs((iT-1)*step+1:iT*step,:);
    if (numTs-iT >= lag)
    XNextT = Locs((iT+lag-1)*step+1:(iT+lag)*step,:);
    end
    for iF=1:F
        ThisX=XThisT((iF-1)*N+1:iF*N,:);
        if (numTs-iT >= lag)
            D = ThisX-XNextT((iF-1)*N+1:iF*N,:);
            NormD = wD*sum((Ext*D).*(Ext*D),2);
            StepDisplacements(iF,iT) = sqrt(NormD);
        end
        EndEndDistances(iF,iT)=norm(barymat(0,s,b)*ThisX-barymat(L,s,b)*ThisX);
    end
    stepdisps = StepDisplacements(:,iT);
    %MeanInBundleDisp(iT) = mean(stepdisps(InBundle(:,end)==1));
    %MeanOutofBundleDisp(iT) = mean(stepdisps(InBundle(:,end)==0));
%     unfilteredDisps = StepDisplacements(:,iT);
%     filtered = StepDisplacements(:,iT);%unfilteredDisps(unfilteredDisps < 0.1);
%     AllMeanVels(seed,iT) = mean(filtered)/dtsave;
%     AllMaxVels(seed,iT) = max(filtered)/dtsave;
end
max(StepDisplacements(:))
% We want a distribution over times when the bundle density is between 2
% and 1 on the way down
% [~,TMaxBund]=max(BundDensity);
% TsWanted = find(BundDensity > 1 & BundDensity < 3.5);
% TsWanted = TsWanted(TsWanted > TMaxBund);
% EndEndDistsToSort = EndEndDistances(:,TsWanted);
% TsWanted = TsWanted(TsWanted <= numTs-lag);
% nEndObs = length(EndEndDistsToSort(:));
% MeanDispsToSort = StepDisplacements(:,TsWanted);
% EndEndBins = histcounts(EndEndDistsToSort(:),0:eebinwidths(iName):L)...
%     /(eebinwidths(iName)*nEndObs);
% dispwidth = freefibval/50;
% nDispObs = length(MeanDispsToSort(:));
% MeanDispBins = histcounts(MeanDispsToSort(:),0:dispwidth:2*freefibval)...
%     /(dispwidth*nDispObs);
MeanDispBins=0;

meanLinkStr = zeros(numTs,1);
meanFibCurv = zeros(numTs,1);
meanBundleOrder = zeros(numTs,1);
meanPerBundle= zeros(numTs,1);
elasticFibEnergy = zeros(numTs,1);
elasticLinkEnergy = zeros(numTs,1);
maxPerBundle = zeros(numTs,1);
AtLeast5InBundle = zeros(numTs,1);
EndEndDists = zeros(numTs,F);
for iT=1:numTs
%     tLinkStr = LinkStrains(startind(iT)+1:startind(iT+1));
%     meanLinkStr(iT)=mean(tLinkStr);
% %     elasticLinkEnergy(iT) = sum((rl*tLinkStr).^2)*Ksp;
    tFibCurves = FiberCurves((iT-1)*F+1:iT*F);
    meanFibCurv(iT)=mean(tFibCurves)/(2*pi/L);
%     elasticFibEnergy(iT) = kappa*L*sum(tFibCurves.^2);
    if (NBundlesPerstep_Sep(iT) > 0)
        start = BundStart(iT)+1;
        meanPerBundle(iT) = sum(NPerBundle_Sep(start:BundStart(iT+1)))/NBundlesPerstep_Sep(iT);
        maxPerBundle(iT) = max(NPerBundle_Sep(start:BundStart(iT+1)));
        AtLeast5InBundle(iT) = sum(NPerBundle_Sep(start:BundStart(iT+1)) > 4.5);
        meanBundleOrder(iT)=sum(BundleOrders_Sep(start:BundStart(iT+1)).*NPerBundle_Sep(start:BundStart(iT+1)))/...
            sum(NPerBundle_Sep(start:BundStart(iT+1)));
        FibersInBundles{iT} = NPerBundle_Sep(start:BundStart(iT+1));
    end
end
% End-end distances
if (seed==0)
    maxstr=0;
    OmHz=1;
    makePythonMovie;
end
nLinks(seed,:)=nLinksPerT*2/(F*L);
nBund(seed,:)=BundDensity;
nContactsTrials(seed,:)=nContacts;
%nBund(seed,:)=NBundlesPerstep_Sep+(F-meanPerBundle.*NBundlesPerstep_Sep);
%nBund(seed,:)=AtLeast5InBundle;
meanBund(seed,:)=meanPerBundle;
nInBund(seed,:)=meanPerBundle.*NBundlesPerstep_Sep/F*100;
MBAlign(seed,:)=meanBundleOrder;
MaxBundSize(seed,:)=maxPerBundle;
meanDispP(seed,:)=MeanDispBins;
meanEndEnd(seed,:)=mean(EndEndDistances);
meanStepDisplacementsP(seed,:)=mean(StepDisplacements);
maxStepDisplacementsP(seed,:)=max(StepDisplacements);
MeanInBundleDispP(seed,:)=MeanInBundleDisp;
MeanOutBundleDispP(seed,:)=MeanOutofBundleDisp;
meanCurvatures(seed,:)=meanFibCurv;
AllFibsInBundlesS{seed}=FibersInBundles;
AllEndEndDists{parSet,seed}=EndEndDists;
clear FibersInBundles EndEndDists
end
nLinksAll{parSet}=nLinks;
nBundAll{parSet}=nBund;
nInBundAll{parSet}=nInBund;
nContactsAll{parSet}=nContactsTrials;
MBAlignAll{parSet}=MBAlign;
MaxBundAll{parSet}=MaxBundSize;
meanBundAll{parSet}=meanBund;
MeanDispAll{parSet}=meanStepDisplacementsP;
MaxDispAll{parSet}=maxStepDisplacementsP;
meanDispB{parSet}=MeanInBundleDispP;
meanDispOB{parSet}=MeanOutBundleDispP;
meanCurvesAll{parSet}=meanCurvatures;
meanEndEndAll{parSet}=meanEndEnd;
AllFibsInBundles{parSet}=AllFibsInBundlesS;
parSet=parSet+1;
clear nLinks nBund nInBund MBAlign MaxBundSize meanDispP meanCurvatures meanBund meanEndEnd
clear AllFibsInBundlesS MeanInBundleDispP MeanOutBundleDispP meanStepDisplacementsP
clear maxStepDisplacementsP nContactsTrials
end
if (plots)
    subplot(3,2,1)
    box on
    ErrorEvery = 20*ones(1,length(names));
    ErrStart=5;
    for iP=1:parSet-1
        hold on
        nLinks = nLinksAll{iP};
        [nTri,nSteps] = size(nLinks);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(nLinks),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(nLinks(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(nLinks(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
    end
    %xlabel('$t$','interpreter','latex')
    ylabel('Link density (per fiber)')
    xlim([0 min(tmaxes)])

    subplot(3,2,2)
    for iP=1:parSet-1
        nBund = nBundAll{iP};
        [nTri,nSteps] = size(nBund);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(nBund),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(nBund(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(nBund(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
        %plot((0:plotEvery(iP):nSteps-1)*dtsave-shift(iP),mean(nBund(:,1:plotEvery(iP):end)),'LineWidth',2.0)
    end
    ylabel('Bundle density')
    box on

    subplot(3,2,3)
    for iP=1:parSet-1
        nInBun = nInBundAll{iP};
        [nTri,nSteps] = size(nInBun);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(nInBun),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(nInBun(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(nInBun(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
        %plot((0:plotEvery(iP):nSteps-1)*dtsave-shift(iP),mean(nInBun(:,1:plotEvery(iP):end)),'LineWidth',2.0)
    end
    %xlabel('$t$','interpreter','latex')
    ylabel('\% in bundles')
    xlim([0 min(tmaxes)])
    
    subplot(3,2,4)
    hold on
    box on
    for iP=1:parSet-1
        meanBO = MBAlignAll{iP};
        [nTri,nSteps] = size(meanBO);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(meanBO),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(meanBO(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(meanBO(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
    end
    %xlabel('$t/\tau_f$','interpreter','latex')
    ylabel('Mean Bundle alignment')
    xlim([0 max(tmaxes)])

    subplot(3,2,5)
    hold on
    box on
    for iP=1:parSet-1
        meanDispThis = MeanDispAll{iP};
        [nTri,nSteps] = size(meanDispThis);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(meanDispThis),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps)*dtsave,mean(meanDispThis(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(meanDispThis(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
    end
    xlabel('$t$','interpreter','latex')
    ylabel('Mean displacement $(\Delta t = 0.02)$')
    xlim([0 max(tmaxes)])
% % 
    subplot(3,2,6)
    hold on
    box on
    for iP=1:parSet-1
        maxDispThis = MaxDispAll{iP};
        [nTri,nSteps] = size(maxDispThis);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(maxDispThis),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps)*dtsave,mean(maxDispThis(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(maxDispThis(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
    end
    xlabel('$t$','interpreter','latex')
    ylabel('Max displacement $(\Delta t = 0.02)$')
    xlim([0 max(tmaxes)])

%     subplot(3,2,5)
%     hold on
%     box on
%     for iP=1:parSet-1
%         meanInBun = meanBundAll{iP};
%         [nTri,nSteps] = size(meanInBun);
%         dtsave=dtsaves(iP);
%         plot((0:nSteps-1)*dtsave,mean(meanInBun),'LineWidth',2.0)
%         hold on
%         set(gca,'ColorOrderIndex',iP)
%         errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(meanInBun(:,ErrStart*iP:ErrorEvery(iP):end)),...
%             std(meanInBun(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
%             'o','MarkerSize',0.5,'LineWidth',1.0)
%     end
%     xlabel('$t$','interpreter','latex')
%     ylabel('Mean bundle size')
%     xlim([0 max(tmaxes)])
% % % 
%     subplot(3,2,6)
%     hold on
%     box on
%     for iP=1:parSet-1
%         maxInBun = MaxBundAll{iP};
%         [nTri,nSteps] = size(maxInBun);
%         dtsave=dtsaves(iP);
%         plot((0:nSteps-1)*dtsave,mean(maxInBun),'LineWidth',2.0)
%         hold on
%         set(gca,'ColorOrderIndex',iP)
%         errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(maxInBun(:,ErrStart*iP:ErrorEvery(iP):end)),...
%             std(maxInBun(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
%             'o','MarkerSize',0.5,'LineWidth',1.0)
%     end
%     xlabel('$t$','interpreter','latex')
%     ylabel('Max bundle size')
%     xlim([0 max(tmaxes)])

figure;
dts=[2e-5 2e-5 1e-5 1e-5 1e-4];%2e-5 1e-5];
%ErrorEvery=[500 1000 2500 5000];
ErrorEvery=[500 500 500 500 500 500];
for iP=1:parSet-2
    skip1 = dts(end)/dts(iP);
    nContactsThis = nContactsAll{iP}(:,1:skip1:end)./nContactsAll{1,end};
    [nTri,nSteps] = size(nContactsThis);
    dtsave=1e-4;%dts(iP);
    plot((0:ErrorEvery(iP)/10:nSteps-1)*dtsave,...
        mean(nContactsThis(:,1:ErrorEvery(iP)/10:nSteps)),'LineWidth',2.0)
    hold on
    set(gca,'ColorOrderIndex',iP)
    errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,...
        mean(nContactsThis(:,ErrStart*iP:ErrorEvery(iP):end)),...
        std(nContactsThis(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
        'o','MarkerSize',0.5,'LineWidth',1.0)
end
box on
end

% MDBinEdges = (0:dispwidth:2*freefibval)/freefibval;
% MDCenters = (MDBinEdges(1:end-1)+MDBinEdges(2:end))/2;
% for iP=1:parSet-1
%     EEBinEdges = 0:eebinwidths(iP):L;
%     EECenters = (EEBinEdges(1:end-1)+EEBinEdges(2:end))/2;
%     subplot(1,2,1)
%     hold on
%     errorbar(EECenters,mean(meanEndEndAll{iP}),std(meanEndEndAll{iP})*2/sqrt(seedmax),...
%         'LineWidth',2.0)
%     subplot(1,2,2)
%     hold on
%     errorbar(MDCenters,mean(MeanDispAll{iP}),std(MeanDispAll{iP})*2/sqrt(seedmax),...
%         'LineWidth',2.0)
% end