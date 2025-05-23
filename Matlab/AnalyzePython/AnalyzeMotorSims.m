% Analyze simulations of "cortical flows"
clear
addpath(genpath('../../Python'))
names=["ConfinedLocQFlowLdx5Mot0.2CL0.3_Dt2.5e-05_" ...
    "ConfinedLocQFlowLdx5Mot0.2CL0.3_Dt5e-05_"];
tmaxes = [15 30];
dtsaves = 5e-2*ones(1,length(names));
nSeeds=[2 2];
if (exist('parSet','var'))
else
    parSet=1;
end
F = 500;
L = 1;
plots = 1;
Ld = 2;
Ldx = 5;
N=13;
for iName=1:length(names)
seedmax=nSeeds(iName);
[s,w,b]=chebpts(N,[0 L],2);
[sD,wD,bD]=chebpts(2*N,[0 L],2);
Ext = barymat(sD,s,b);
COMsample = barymat(L/2,s,b);
for seed=1:seedmax
FileName = strcat(names(iName),num2str(seed),'.txt'); % ALWAYS LOOK @ MOVIE!
Locs=load(strcat('Locs',FileName));
% Separate bundle
BundleOrders_Sep = load(strcat('BundleOrderParams_Sep',FileName));
NPerBundle_Sep = load(strcat('NFibsPerBundle_Sep',FileName));
NBundlesPerstep_Sep = load(strcat('NumberOfBundles_Sep',FileName));
BundDensity=NBundlesPerstep_Sep/(Ld^2*Ldx);
BundStart = [0; cumsum(NBundlesPerstep_Sep)];
% Fiber info
try 
nContacts = load(strcat('nContacts',FileName));
catch
nContacs = 0;
end
LocalFibAlignment = load(strcat('LocalAlignmentPerFib',FileName));
nFibsConnected = load(strcat('NumFibsConnectedPerFib',FileName));
% Link info
NLinksPerFib = load(strcat('nLinksPerFib',FileName));
nLinksPerT = sum(NLinksPerFib')/2;
[~,numTs]=size(nLinksPerT);
startind = [0 cumsum(nLinksPerT)];
% Labels for visualization
labels = load(strcat('FinalLabels_Sep',FileName));
[nts,~] = size(labels);
InBundle = zeros(F,nts);
for iT=1:nts
    lab=1;
    newlabels = zeros(F,1);
    for iFib=1:F
        if (newlabels(iFib)==0)
            iFiblab = labels(iT,iFib);
            if (sum(labels(iT,:)==iFiblab)==1)
                newlabels(iFib)=-1;
            else
                newlabels(labels(iT,:)==iFiblab,iT)=lab;
                lab=lab+1;
                InBundle(iFib,iT)=1;
            end
        else
            InBundle(iFib,iT)=1;
        end
    end
end

StepDisplacements = zeros(numTs-1,F);
EndEndDistances = zeros(numTs-1,F);
MeanXFromCenter = zeros(numTs-1,F);
step=N*F;
for iT=1:numTs-1
    XThisT = Locs((iT-1)*step+1:iT*step,:);
    XNextT = Locs(iT*step+1:(iT+1)*step,:);
    for iF=1:F
        ThisX=XThisT((iF-1)*N+1:iF*N,:);
        MeanX = mod(w*ThisX(:,1)/L,Ldx);
        MeanXFromCenter(iT,iF) = abs(MeanX/Ldx-1/2);
        NextX=XNextT((iF-1)*N+1:iF*N,:);
        D = NextX-ThisX;
        NormD = wD*sum((Ext*D).*(Ext*D),2);
        StepDisplacements(iT,iF) = sqrt(NormD);
        EndEndDistances(iT,iF)=norm(barymat(0,s,b)*ThisX-barymat(L,s,b)*ThisX);
    end
end
max(StepDisplacements(:))

meanBundleOrder = zeros(numTs,1);
meanPerBundle= zeros(numTs,1);
maxPerBundle = zeros(numTs,1);
AtLeast5InBundle = zeros(numTs,1);
for iT=1:numTs
    if (NBundlesPerstep_Sep(iT) > 0)
        start = BundStart(iT)+1;
        meanPerBundle(iT) = sum(NPerBundle_Sep(start:BundStart(iT+1)))/NBundlesPerstep_Sep(iT);
        maxPerBundle(iT) = max(NPerBundle_Sep(start:BundStart(iT+1)));
        meanBundleOrder(iT)=sum(BundleOrders_Sep(start:BundStart(iT+1)).*NPerBundle_Sep(start:BundStart(iT+1)))/...
            sum(NPerBundle_Sep(start:BundStart(iT+1)));
    end
end
% Motor calculations
AvgMotorSpeed = zeros(numTs,1);
PctStuckMotors = zeros(numTs,1);
NMotorsPerFib = load(strcat('nMotorsPerFib',FileName));
MotorSpeeds = load(strcat('MotorSpeeds',FileName));
NumMotorsPerT = sum(NMotorsPerFib,2);
NumMotorsPerFib = mean(NMotorsPerFib,2);
startind = [0; cumsum(NumMotorsPerT)];
for iT=1:numTs
    MotSpeeds = MotorSpeeds(startind(iT)+1:startind(iT+1));
    AvgMotorSpeed(iT)=mean(MotSpeeds);
    PctStuckMotors(iT)=sum(MotSpeeds==0)/length(MotSpeeds);
end

% Collect statistics
AvgMotSpeeds(seed,:)=AvgMotorSpeed;
PctStuckMotorsss(seed,:)=PctStuckMotors;
NumMotorsPerFibs(seed,:)=NumMotorsPerFib;
nLinks(seed,:)=nLinksPerT*2/(F*L);
nBund(seed,:)=BundDensity;
nContactsTrials(seed,:)=nContacts;
meanBund(seed,:)=meanPerBundle;
nInBund(seed,:)=meanPerBundle.*NBundlesPerstep_Sep/F*100;
MBAlign(seed,:)=meanBundleOrder;
MaxBundSize(seed,:)=maxPerBundle;
meanEndEnd(seed,:)=mean(EndEndDistances');
meanStepDisplacementsP(seed,:)=mean(StepDisplacements');
maxStepDisplacementsP(seed,:)=max(StepDisplacements');
meanXLoc(seed,:)=mean(MeanXFromCenter');
end
AvgMotSpeedAll{parSet}=AvgMotSpeeds;
PctStuckMotorsAll{parSet}=PctStuckMotorsss;
nMotsAll{parSet}=NumMotorsPerFibs;
nLinksAll{parSet}=nLinks;
nBundAll{parSet}=nBund;
nInBundAll{parSet}=nInBund;
nContactsAll{parSet}=nContactsTrials;
MBAlignAll{parSet}=MBAlign;
MaxBundAll{parSet}=MaxBundSize;
meanBundAll{parSet}=meanBund;
MeanDispAll{parSet}=meanStepDisplacementsP;
MaxDispAll{parSet}=maxStepDisplacementsP;
meanEndEndAll{parSet}=meanEndEnd;
meanXLocs{parSet} = meanXLoc;
parSet=parSet+1;
clear nLinks nBund nInBund MBAlign MaxBundSize meanDispP meanCurvatures meanBund meanEndEnd
clear AllFibsInBundlesS MeanInBundleDispP MeanOutBundleDispP meanStepDisplacementsP
clear maxStepDisplacementsP nContactsTrials
clear AvgMotSpeeds PctStuckMotorsss NumMotorsPerFibs meanXLoc
end

if (plots)
    % We want to plot the X position of the fibers, the flow speed
    % (um/min), the end to end distances, and statistics about the CLs and motors
    tiledlayout(3,1,'Padding', 'none', 'TileSpacing', 'compact');
    nexttile
    ErrorEvery = 20*ones(1,length(names));
    ErrStart=5;
    % X Position of fibers
    for iP=1:parSet-1
        MeanX = meanXLocs{iP};
        [nTri,nSteps] = size(MeanX);
        dtsave=dtsaves(iP);
        if (iP<parSet)
        plot((0:nSteps-1)*dtsave,mean(MeanX,'omitnan'),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps)*dtsave,...
            mean(MeanX(:,ErrStart*iP:ErrorEvery(iP):end),'omitnan'),...
            std(MeanX(:,ErrStart*iP:ErrorEvery(iP):end),'omitnan')*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
        else
        for j=1:nSeeds(iP)
        plot((0:nSteps-1)*dtsave,MeanX(j,:),'LineWidth',2.0)
        end
        end
    end
    %xlabel('$t$','interpreter','latex')
    ylabel('$\bar x$ (from center)')
    xlim([0 max(tmaxes)])
    
%     % Fiber end to end distances
%     nexttile
%     for iP=1:parSet-1
%         meanEE = meanEndEndAll{iP};
%         [nTri,nSteps] = size(meanEE);
%         dtsave=dtsaves(iP);
%         if (iP<parSet)
%         plot((0:nSteps-1)*dtsave,mean(meanEE),'LineWidth',2.0)
%         hold on
%         set(gca,'ColorOrderIndex',iP)
%         BarInds=ErrStart*iP:ErrorEvery(iP):nSteps-1;
%         errorbar((BarInds-1)*dtsave,mean(meanEE(:,BarInds)),...
%             std(meanEE(:,BarInds))*2/sqrt(nTri),...
%             'o','MarkerSize',0.5,'LineWidth',1.0)
%         else
%         for j=1:nSeeds(iP)
%         plot((0:nSteps-1)*dtsave,meanEE(j,:),'LineWidth',2.0)
%         end
%         end
%     end
%     ylabel('Mean end-to-end distance')
%     xlim([0 max(tmaxes)])
%     
    nexttile
    for iP=1:parSet-1
        hold on
        nLinks = nLinksAll{iP};
        [nTri,nSteps] = size(nLinks);
        dtsave=dtsaves(iP);
        if (iP<parSet)
        plot((0:nSteps-1)*dtsave,mean(nLinks),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(nLinks(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(nLinks(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
        else
        for j=1:nSeeds(iP)
        plot((0:nSteps-1)*dtsave,nLinks(j,:),'LineWidth',2.0)
        end
        end
    end
    %xlabel('$t$','interpreter','latex')
    ylabel('Link density (per fiber)')
    xlim([0 max(tmaxes)])


    nexttile
    ErrorEvery = 20*ones(1,length(names));
    ErrStart=5;
    for iP=1:parSet-1
        hold on
        nMots = nMotsAll{iP};
        [nTri,nSteps] = size(nMots);
        dtsave=dtsaves(iP);
        if (iP<parSet)
        plot((0:nSteps-1)*dtsave,mean(nMots),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(nMots(:,ErrStart*iP:ErrorEvery(iP):end)),...
            std(nMots(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
        else
        for j=1:nSeeds(iP)
        plot((0:nSteps-1)*dtsave,nMots(j,:),'LineWidth',2.0)
        end
        end
    end
    %xlabel('$t$','interpreter','latex')
    ylabel('Motor density (per fiber)')
    xlim([0 max(tmaxes)])

%     nexttile
%     ErrorEvery = 20*ones(1,length(names));
%     ErrStart=5;
%     for iP=1:parSet-1
%         hold on
%         avgspeed = AvgMotSpeedAll{iP};
%         [nTri,nSteps] = size(avgspeed);
%         dtsave=dtsaves(iP);
%         if (iP<parSet)
%         plot((0:nSteps-1)*dtsave,mean(avgspeed),'LineWidth',2.0)
%         hold on
%         set(gca,'ColorOrderIndex',iP)
%         errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(avgspeed(:,ErrStart*iP:ErrorEvery(iP):end)),...
%             std(avgspeed(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
%             'o','MarkerSize',0.5,'LineWidth',1.0)
%         else
%         for j=1:nSeeds(iP)
%         plot((0:nSteps-1)*dtsave,avgspeed(j,:),'LineWidth',2.0)
%         end
%         end
%     end
%     xlabel('$t$','interpreter','latex')
%     ylabel('Avg motor speed')
%     xlim([0 max(tmaxes)])
% 
%     nexttile
%     ErrorEvery = 20*ones(1,length(names));
%     ErrStart=5;
%     for iP=1:parSet-1
%         hold on
%         pctstuck = PctStuckMotorsAll{iP};
%         [nTri,nSteps] = size(pctstuck);
%         dtsave=dtsaves(iP);
%         if (iP<parSet)
%         plot((0:nSteps-1)*dtsave,mean(pctstuck),'LineWidth',2.0)
%         hold on
%         set(gca,'ColorOrderIndex',iP)
%         errorbar((ErrStart*iP:ErrorEvery(iP):nSteps-1)*dtsave,mean(pctstuck(:,ErrStart*iP:ErrorEvery(iP):end)),...
%             std(pctstuck(:,ErrStart*iP:ErrorEvery(iP):end))*2/sqrt(nTri),...
%             'o','MarkerSize',0.5,'LineWidth',1.0)
%         else
%         for j=1:nSeeds(iP)
%         plot((0:nSteps-1)*dtsave,pctstuck(j,:),'LineWidth',2.0)
%         end
%         end        
%     end
%     xlabel('$t$','interpreter','latex')
%     ylabel('Frac motors stuck')
%     xlim([0 max(tmaxes)])

    figure;
    % Contacts
    for iP=1:parSet-1
        nContactsThis = nContactsAll{iP};
        [nTri,nSteps] = size(nContactsThis);
        dtsave=dtsaves(iP);
        plot((0:nSteps-1)*dtsave,mean(nContactsThis),'LineWidth',2.0)
        hold on
        set(gca,'ColorOrderIndex',iP)
        BarInds=ErrStart*iP:ErrorEvery(iP):nSteps-1;
        errorbar((BarInds-1)*dtsave,...
            mean(nContactsThis(:,BarInds)),...
            std(nContactsThis(:,BarInds))*2/sqrt(nTri),...
            'o','MarkerSize',0.5,'LineWidth',1.0)
    end
    xlim([0 max(tmaxes)])
    ylabel('Mean contacts')
end