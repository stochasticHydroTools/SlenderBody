maxTrial = 10;
nErrorBars = 2;
nPerErrorBar = maxTrial/nErrorBars;
nOuter=12;
AllRVals=zeros(nErrorBars,nPerErrorBar*nOuter);
for iError=1:nErrorBars
for jTrial=1:nPerErrorBar
iTrial=(iError-1)*nPerErrorBar+jTrial;
AllLfacs = AllExtensions{iTrial};
Xpts = AllPositions{iTrial};
nLinksT = AllnLinks{iTrial};
AllLinks = AllLinksConfs{iTrial};
Thetass = AllAngles{iTrial};
D1s = AllMatFrames{iTrial};
% %% Statistics about bundling
% Information about the bundle as a whole
nPerSave = Nx*nFib;
nSaves = length(Xpts)/nPerSave;
for iT=1:nSaves
    PtsThisT = Xpts((iT-1)*nPerSave+1:iT*nPerSave,:);
    meanEP = zeros(1,3);
    meanSP = zeros(1,3);
    for iFib=1:nFib
        fibInds = (iFib-1)*Nx+1:iFib*Nx;
        XFib = PtsThisT(fibInds,:);
        CurvMatrix = 1/AllLfacs(iT,iFib)^2*DNp1^2;
        WTilde_Np1_L = AllLfacs(iT,iFib)*WTilde_Np1;
        FibCurv = reshape((CurvMatrix*XFib)',[],1);
        AllMeanCurves(iT,iFib)=sqrt(FibCurv'*WTilde_Np1_L*FibCurv);
        EPthis = barymat(L,sNp1,bNp1)*XFib;
        MPthis = barymat(L/2,sNp1,bNp1)*XFib;
        Zerothis = barymat(0,sNp1,bNp1)*XFib;
        % Closest point on central fiber is the origin
        if (iFib==1)
            CentralFiber=RplNp1*XFib;
        else
            EffOrigin=zeros(Nx,3);
%             for iPt=1:Nx
%                 diff = XFib(iPt,:)-CentralFiber;
%                 [~,ind]=min(sum(diff.*diff,2));
%                 EffOrigin(iPt,:)=CentralFiber(ind,:);
%             end
%             EffCurve = XFib-EffOrigin;
%             [RotAngles, RVals] = cart2pol(EffCurve(:,1),EffCurve(:,2));
%             sgns = sign(RotAngles(1:end-1).*RotAngles(2:end));
%             diff = (RotAngles(1:end-1) - RotAngles(2:end)).*sign(fmot0);
%             nschanges = cumsum(sgns < 1 & diff > pi);
%             RotAngles = RotAngles+[0;nschanges*2*pi];
%             AllDeltaThetas(iT,iFib)=barymat(L/2,sNp1,bNp1)*RotAngles...
%                 -barymat(0,sNp1,bNp1)*RotAngles;
%             AllDeltaThetas(iT,nFib+iFib)=barymat(L,sNp1,bNp1)*RotAngles...
%                 -barymat(0,sNp1,bNp1)*RotAngles;
            % Easier procedure always taking angle on [0,2pi]
            diff = EPthis-CentralFiber;
            [~,ind]=min(sum(diff.*diff,2));
            EPAxis=diff(ind,:);
            [rotangEP,~]=cart2pol(EPAxis(1),EPAxis(:,2));
            diff = MPthis-CentralFiber;
            [~,ind]=min(sum(diff.*diff,2));
            MPAxis=diff(ind,:);
            [rotangMP,~]=cart2pol(MPAxis(1),MPAxis(:,2));
            rot0 = cart2pol(Zerothis(1),Zerothis(2));
            AllDeltaThetas(iT,iFib)=rotangMP-rot0;
            AllDeltaThetas(iT,nFib+iFib)=rotangEP-rot0;
        end
        % Compute angle with initial tangent axis
        AllXVals(iT,iFib)=EPthis(1);
        AllYVals(iT,iFib)=EPthis(2);
        AllZVals(iT,iFib)=EPthis(3);
        AllAxisAngles(iT,iFib)=acos((EPthis(3)-Zerothis(3))/norm(EPthis-Zerothis));
    end
end
inds=1:nSaves;
AllDeltaThetas(AllDeltaThetas < -1)=AllDeltaThetas(AllDeltaThetas<-1)+2*pi;
AllDeltaThetas(AllDeltaThetas > 5)=AllDeltaThetas(AllDeltaThetas>5)-2*pi;
%plot(AllXVals(end,11:22),AllYVals(end,11:22),'b>')
%hold on
AllRVals(iError,nOuter*(jTrial-1)+1:jTrial*nOuter) = ...
    sqrt(AllXVals(end,11:22).^2+AllYVals(end,11:22).^2);
% Statistics by circle
%plot(AllDeltaThetas)
%drawnow
CircleInds{1}=1;
CircleInds{2}=2:4;
CircleInds{3}=5:10;
CircleInds{4}=11:22;
for iCirc=1:nCircles
    if (length(CircleInds{iCirc})==1)
        MeanThisCirc=AllMeanCurves(:,CircleInds{iCirc})/(2*pi);
        MeanBendAngle=AllAxisAngles(:,CircleInds{iCirc})*180/pi;
        MeanZ = AllZVals(:,CircleInds{iCirc});
        MeanEPNumRots = 0;
        MeanMPNumRots = 0;
    else
        MeanThisCirc=mean(AllMeanCurves(:,CircleInds{iCirc})')'/(2*pi);
        MeanBendAngle=mean(AllAxisAngles(:,CircleInds{iCirc})')'*180/pi;
        MeanMPNumRots=mean(AllDeltaThetas(:,CircleInds{iCirc})')'/pi;
        MeanEPNumRots=mean(AllDeltaThetas(:,nFib+CircleInds{iCirc})')'/pi;
        MeanZ = mean(AllZVals(:,CircleInds{iCirc})')';
    end
    if (jTrial==1)
        CurvesByCirc{iCirc}(iError,:)=zeros(1,nSaves);
        AnglesByCirc{iCirc}(iError,:)=zeros(1,nSaves);
        MPRotsByCirc{iCirc}(iError,:)=zeros(1,nSaves);
        EPRotsByCirc{iCirc}(iError,:)=zeros(1,nSaves);
        ZByCirc{iCirc}(iError,:)=zeros(1,nSaves);
    end
    CurvesByCirc{iCirc}(iError,:)=CurvesByCirc{iCirc}(iError,:)+MeanThisCirc'/nPerErrorBar;
    AnglesByCirc{iCirc}(iError,:)=AnglesByCirc{iCirc}(iError,:)+MeanBendAngle'/nPerErrorBar;
    MPRotsByCirc{iCirc}(iError,:)=MPRotsByCirc{iCirc}(iError,:)+MeanMPNumRots'/nPerErrorBar;
    EPRotsByCirc{iCirc}(iError,:)=EPRotsByCirc{iCirc}(iError,:)+MeanEPNumRots'/nPerErrorBar;
    ZByCirc{iCirc}(iError,:)=ZByCirc{iCirc}(iError,:)+MeanZ'/nPerErrorBar;
end
end
end

ts=0:saveEvery*dt:tf;
for iCirc=1:nCircles
    MeanCurvesByCirc(iCirc,:) = mean(CurvesByCirc{iCirc});
    MeanAnglesByCirc(iCirc,:) = mean(AnglesByCirc{iCirc});
    MeanMPRotsByCirc(iCirc,:) = mean(MPRotsByCirc{iCirc});
    MeanEPRotsByCirc(iCirc,:) = mean(EPRotsByCirc{iCirc});
    MeanZByCirc(iCirc,:) = mean(ZByCirc{iCirc});
    StdCurvesByCirc(iCirc,:) = 2*std(CurvesByCirc{iCirc})/sqrt(nErrorBars);
    StdAnglesByCirc(iCirc,:) = 2*std(AnglesByCirc{iCirc})/sqrt(nErrorBars);
    StdMPRotsByCirc(iCirc,:) = 2*std(MPRotsByCirc{iCirc})/sqrt(nErrorBars);
    StdEPRotsByCirc(iCirc,:) = 2*std(EPRotsByCirc{iCirc})/sqrt(nErrorBars);
    StdZByCirc(iCirc,:) = 2*std(ZByCirc{iCirc})/sqrt(nErrorBars);
    plot(ts,MeanMPRotsByCirc(iCirc,:),':')
    hold on
    errorEvery=100;
    startval=iCirc*errorEvery/nCircles;
    set(gca,'ColorOrderIndex',iCirc)
    errorbar(ts(startval:errorEvery:end),MeanMPRotsByCirc(iCirc,startval:errorEvery:end),...
        StdMPRotsByCirc(iCirc,startval:errorEvery:end),'o','MarkerSize',0.01,'LineWidth',2)
    set(gca,'ColorOrderIndex',iCirc)
    plot(ts,MeanEPRotsByCirc(iCirc,:),'-')
    hold on
    startval=iCirc*errorEvery/nCircles;
    set(gca,'ColorOrderIndex',iCirc)
    errorbar(ts(startval:errorEvery:end),MeanEPRotsByCirc(iCirc,startval:errorEvery:end),...
        StdEPRotsByCirc(iCirc,startval:errorEvery:end),'o','MarkerSize',0.01,'LineWidth',2)
end
return
% 

% % %% Plotting the statistics
% subplot(3,2,1)

% ts=0:saveEvery*dt:tf;
% % plot(ts,AllZVals(:,2))
% % hold on
% % set(gca,'ColorOrderIndex',5)
% % plot(ts,AllZVals(:,1))
% % plot(ts,mean(AllZVals')',':k')
% % title('$z$ coordinate')
% subplot(3,2,5.5)
nT = length(AllXVals(:,1));
for iT=1:nT
    for iFib=1:nFib
        set(gca,'ColorOrderIndex',iFib)
        scatter(AllXVals(iT,iFib),AllYVals(iT,iFib),'filled','MarkerFaceAlpha',0.1+0.9*iT/nT);
        hold on
    end
end
% box on
% title('Trajectories in $(x,y)$ plane')
% xlabel('$x$')
% ylabel('$y$')
% subplot(3,2,3)

title('Normalized $L^2$ curvature')
xlabel('$t$ (s)','interpreter','latex')
subplot(3,2,4)
plot(ts,AllAxisAngles(:,1)*180/pi)
hold on
set(gca,'ColorOrderIndex',5)
plot(ts,AllAxisAngles(:,5)*180/pi)
plot(ts,AllAxisAngles(:,6)*180/pi,':k')
title('Angle from straight (deg)')
xlabel('$t$ (s)','interpreter','latex')
legend('Exterior','Central','Average','Location','Northeast')
subplot(3,2,2)
for iFib=1:nFib-1
    plot(ts,AllDeltaThetas(:,iFib)/(2*pi),':')
    hold on
    set(gca,'ColorOrderIndex',iFib)
    plot(ts,AllDeltaThetas(:,nFib+iFib)/(2*pi))
end
title('Rotations around central fiber')
legend('Midpoint','Endpoint')
return