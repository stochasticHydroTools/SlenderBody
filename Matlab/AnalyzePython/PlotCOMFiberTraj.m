% Look at some filament COM trajectories
F = 500;
N = 12;
Nx = N+1;
L = 1;
Ld = 2;
Ldx = 5;
%ConfinedNoStFlowLdx5Mot0.3Turn10_Dt0.0001_
Name="ConfinedLocQFlowLdx5Mot0.2CL0.3_Dt5e-05_";
clear Density MeanFlows AlignParameters
c=[0    0.4470    0.7410];
c=[0.8500    0.3250    0.0980];
%c=[0.9290    0.6940    0.1250];
%c=[0 0 0];
nSeed=2;
for seed=1:nSeed
%seed=1;
Locs=load(strcat("Locs",Name,num2str(seed),".txt"));
nT = length(Locs)/(Nx*F);
[sx,wx,bx]=chebpts(Nx,[0 L],2);
D=diffmat(Nx,[0 L]);
Movie=0;
if (Movie)
f=figure;
end
nBinsForFlow=10;
dBin=Ldx/nBinsForFlow;
nObsPerBin = zeros(1,nBinsForFlow);
MeanFlowPerBin = zeros(1,nBinsForFlow);
BinMatrixAlign = zeros(3,3,nBinsForFlow);
dtsave=5e-2;
WindowTime=1;
RecordEveryT=0.5;
WindowAvg = WindowTime/dtsave;
RecordEvery = RecordEveryT/dtsave;
for iF=1:F
    COM = zeros(nT,3);
    TauBar = zeros(nT,3);
    for iT=1:nT
        %nexttile
        X = Locs((iT-1)*F*Nx+1:iT*F*Nx,:);
        Xf = X((iF-1)*Nx+1:iF*Nx,:);
        Tauf = D*Xf;
        COM(iT,:) = wx/L*Xf;
        TauBar(iT,:)=wx/L*Tauf;
        TauBar(iT,:)=TauBar(iT,:)/norm(TauBar(iT,:));
    end
    % Break into sets of trajectories
    Disp=COM(1:end-1,:)-COM(2:end,:);
    NormDisp = sqrt(sum(Disp.*Disp,2));
    Jumps = find(NormDisp>0.3);
    Jumps=[Jumps;nT];
    TrajIndex=ones(nT,1);
    for k=1:length(Jumps)-1
        TrajIndex(Jumps(k)+1:Jumps(k+1))=k+1;
    end
    for k=1:max(TrajIndex)
        TrajPlot = COM(TrajIndex==k,:);
        TauPlot = TauBar(TrajIndex==k,:);
        % Average over 1 s windows
        MovAvg=movmean(TrajPlot,WindowAvg);
        MovAvg=MovAvg(1:RecordEvery:end,:);
        MovAvgTau=movmean(TauPlot,WindowAvg);
        MovAvgTau=MovAvgTau(1:RecordEvery:end,:);
        MovAvgTau=MovAvgTau./sqrt(sum(MovAvgTau.*MovAvgTau,2));
        [nTPts,~]=size(TrajPlot);
        [nAPts,~]=size(MovAvg);
        % Plot any periodic copies that fall in the box
        if (Movie)
        Shifts = unique([Ldx Ld Ld].*floor(TrajPlot./[Ldx Ld Ld]),'Rows');
        nS=size(Shifts,1);
        for iS=1:nS
            % c1=sky(nTPts);
            % scatter3(TrajPlot(:,1)-Shifts(iS,1),TrajPlot(:,2)-Shifts(iS,2),...
            %     TrajPlot(:,3)-Shifts(iS,3),20,c1,'filled')
            % hold on
            c2=summer(nAPts);
            scatter3(MovAvg(:,1)-Shifts(iS,1),MovAvg(:,2)-Shifts(iS,2),...
                MovAvg(:,3)-Shifts(iS,3),50,c2,'filled')
            hold on
        end
        end
        % From the moving average read off displacements as proxy for
        % cortical flows
        MovAvgDispX=MovAvg(2:end,1)-MovAvg(1:end-1,1);
        SpeedDisp = MovAvgDispX/RecordEveryT;
        LocsX = 1/2*(MovAvg(2:end,1)+MovAvg(1:end-1,1));
        TauX = 1/2*(MovAvgTau(2:end,:)+MovAvgTau(1:end-1,:));
        TauX = TauX./sqrt(sum(TauX.*TauX,2));
        LocsX = LocsX-Ldx.*floor(LocsX/Ldx);
        BinNum = ceil(LocsX/dBin);
        for j=1:nAPts-1
            nObsPerBin(BinNum(j))=nObsPerBin(BinNum(j))+1;
            MeanFlowPerBin(BinNum(j))=MeanFlowPerBin(BinNum(j))+SpeedDisp(j);
            BinMatrixAlign(:,:,BinNum(j))=BinMatrixAlign(:,:,BinNum(j))...
                +TauX(j,:)'*TauX(j,:);
        end
    end
    if (Movie)
    view(2)
    xlim([0 Ldx])
    ylim([0 Ld])
    zlim([0 Ld])
    PlotAspect;
    movieframes(iT)=getframe(f);
    hold off
    end
end
Density(seed,:)=nObsPerBin/(dBin*sum(nObsPerBin));
MeanFlows(seed,:)=MeanFlowPerBin./nObsPerBin*60;
% Process alignment matrix
AlignParams = zeros(1,nBinsForFlow);
for j=1:nBinsForFlow
    Mat=BinMatrixAlign(:,:,j)/nObsPerBin(j);
    % Compute eigenvalues
    AlignParams(j)=max(eig(Mat));
end
AlignParameters(seed,:)=AlignParams;
% subplot(1,3,1)
% plot((1/2:nBinsForFlow)*dBin,Density(seed,:),'Color',c,'LineWidth',1)
% hold on
% subplot(1,3,2)
% plot((1/2:nBinsForFlow)*dBin,MeanFlows(seed,:),'Color',c,'LineWidth',1)
% hold on
% subplot(1,3,3)
% plot((1/2:nBinsForFlow)*dBin,AlignParameters(seed,:),'Color',c,'LineWidth',1);
% hold on
end
subplot(1,3,1)
errorbar((1/2:nBinsForFlow)*dBin,mean(Density),2*std(Density)/sqrt(nSeed),...
    'Linewidth',2,'Color',c)
hold on
subplot(1,3,2)
errorbar((1/2:nBinsForFlow)*dBin,mean(MeanFlows),2*std(MeanFlows)/sqrt(nSeed),...
    'Linewidth',2,'Color',c)
hold on
subplot(1,3,3)
errorbar((1/2:nBinsForFlow)*dBin,mean(AlignParameters),2*std(AlignParameters)/sqrt(nSeed),...
    'Linewidth',2,'Color',c)
hold on
