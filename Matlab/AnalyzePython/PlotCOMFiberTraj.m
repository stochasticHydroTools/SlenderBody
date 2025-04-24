% Look at some filament COM trajectories
F = 300;
N = 12;
Nx = N+1;
L = 1;
Ld = 2;
Ldx = 5;
%ConfinedNoStFlowLdx5Mot0.3Turn10_Dt0.0001_
Name="ConfinedNoStFlowLdx5Mot0.3Turn10_Dt0.0001_";
for seed=1:2
%seed=1;
Locs=load(strcat("Locs",Name,num2str(seed),".txt"));
nT = length(Locs)/(Nx*F);
[sx,wx,bx]=chebpts(Nx,[0 L],2);
Movie=0;
if (Movie)
f=figure;
end
nBinsForFlow=10;
dBin=Ldx/nBinsForFlow;
nObsPerBin = zeros(1,nBinsForFlow);
MeanFlowPerBin = zeros(1,nBinsForFlow);
dtsave=5e-2;
WindowTime=1;
RecordEveryT=0.5;
WindowAvg = WindowTime/dtsave;
RecordEvery = RecordEveryT/dtsave;
for iF=1:F
    COM = zeros(nT,3);
    for iT=1:nT
        %nexttile
        X = Locs((iT-1)*F*Nx+1:iT*F*Nx,:);
        Xf = X((iF-1)*Nx+1:iF*Nx,:);
        COM(iT,:) = wx/L*Xf;
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
        % Average over 1 s windows
        MovAvg=movmean(TrajPlot,WindowAvg);
        MovAvg=MovAvg(1:RecordEvery:end,:);
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
        LocsX = LocsX-Ldx.*floor(LocsX/Ldx);
        BinNum = ceil(LocsX/dBin);
        for j=1:nAPts-1
            nObsPerBin(BinNum(j))=nObsPerBin(BinNum(j))+1;
            MeanFlowPerBin(BinNum(j))=MeanFlowPerBin(BinNum(j))+SpeedDisp(j);
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
figure(2);
plot((0:nBinsForFlow-1)*dBin,MeanFlowPerBin./nObsPerBin*60)
hold on
end
