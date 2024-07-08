%% Plotting a movie
iTrial=1;
AllLfacs = AllExtensions{iTrial};
Xpts = AllPositions{iTrial};
nLinksT = AllnLinks{iTrial};
AllLinks = AllLinksConfs{iTrial};
Thetass = AllAngles{iTrial};
D1s = AllMatFrames{iTrial};
%f=figure;
%f.Position = [100 100 1000 1000];
savedt=saveEvery*dt;
tmovies = 0:0.01:2;
%tmovies = [0 0.01 0.02 0.04 0.06 0.08 0.1 0.15 0.2 0.3];
nSaves = length(nLinksT);
for iT=nSaves
    %subplot(2,3,iT)
    nexttile
    saveIndex=iT;%tmovies(iT)/(savedt)+1;
    linkEnd = sum(nLinksT(1:saveIndex));
    nLinks = nLinksT(saveIndex);
    links=AllLinks(linkEnd-nLinks+1:linkEnd,:);
    PtsThisT = Xpts((saveIndex-1)*nFib*Nx+1:saveIndex*nFib*Nx,:);
    D1sThisT = D1s((saveIndex-1)*nFib*N+1:saveIndex*nFib*N,:);
    for iFib=1:nFib
        fibInds = (iFib-1)*Nx+1:iFib*Nx;
        plot3(RplNp1*PtsThisT(fibInds,1),RplNp1*PtsThisT(fibInds,2),...
            RplNp1*PtsThisT(fibInds,3));
        hold on
        FramePts = RNp1ToN*PtsThisT(fibInds,:);
        FrameToPlot = D1((iFib-1)*N+1:iFib*N,:);
        %set(gca,'ColorOrderIndex',iFib)
        %quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
        %    FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),0.25,'LineWidth',1.0);
    end
    [nLinks,~]=size(links);
    [~,X1stars,X2stars] = getCLforceEn(links,PtsThisT,Runi,KCL, zeros(nLinks,1),0,0);
    for iLink=1:nLinks
        linkPts = [X1stars(iLink,:); X2stars(iLink,:)];
        plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko');
    end
    %xlim([-0.25 0.5])
   % ylim([-0.6 0.4])
    %zlim([-0.5 0.9])
    PlotAspect
    %view([ -59.1403    5.5726])
    view([-109.5316   19.7340])
    title(strcat('$t=$',num2str(tmovies(iT))))
    hold off
    %movieframes(iT)=getframe(f);
end
