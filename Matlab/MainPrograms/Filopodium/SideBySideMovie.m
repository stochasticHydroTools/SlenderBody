f=figure;
for iT=2:400
    clf;
    tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');
    nexttile
    Xinds = (iT-1)*Nx+1:iT*Nx;
    FrInds = (iT-1)*N+1:iT*N;
    X2Now = RplNp1*X_2Turn(Xinds,:);
    plot3(X2Now(:,1),X2Now(:,2),X2Now(:,3));
    hold on
    FramePts = RplN*RNp1ToN*X_2Turn(Xinds,:);
    FrameToPlot = RplN*D1_2Turn(FrInds,:);
    quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
        FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),'LineWidth',1.0);
    xlim([-0.62 0.52])
    ylim([-0.62 0.52])
    zlim([0 2])
    PlotAspect
    title(strcat('2-turns, $t=$',num2str(iT*1e-2)))
    nexttile
    X2Now = RplNp1*X_3Turn(Xinds,:);
    plot3(X2Now(:,1),X2Now(:,2),X2Now(:,3));
    hold on
    FramePts = RplN*RNp1ToN*X_3Turn(Xinds,:);
    FrameToPlot = RplN*D1_3Turn(FrInds,:);
    quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
        FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),'LineWidth',1.0);
    xlim([-0.62 0.52])
    ylim([-0.62 0.52])
    zlim([0 2])
    PlotAspect
    title(strcat('3-turns, $t=$',num2str(iT*1e-2)))
    %xlim([-0.2 0.2])
    %ylim([-0.2 0.2])
    movieframes(iT-1)=getframe(f);
end