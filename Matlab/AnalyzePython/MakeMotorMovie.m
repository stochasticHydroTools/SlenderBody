F = 400;
N = 8;
Nx = N+1;
L = 0.5;
Nu = 40;
Npl = 100;
Ld = 2;
f=figure;
%tiledlayout(3,3,'Padding', 'none', 'TileSpacing', 'compact');
Name="ConfinedFlowMot2Turn_Dt5e-05_";
seed=2;
Locs=load(strcat("Locs",Name,num2str(seed),".txt"));
nT = length(Locs)/(Nx*F);
[sx,wx,bx]=chebpts(Nx,[0 L],2);
Rpl = barymat((0:1/Npl:L)',sx,bx);
Ru = barymat((0:1/(Nu-1):L)',sx,bx);
nLTime=zeros(nT,1);
X0 = Locs(1:F*Nx,:);
for iT=1:nT
    %nexttile
    X = Locs((iT-1)*F*Nx+1:iT*F*Nx,:);
    if (iT>1)
        Xp=Locs((iT-2)*F*Nx+1:(iT-1)*F*Nx,:);
        change(iT)=max(max(abs(X-Xp)));
    end
    % Plot fibers
    for iF=1:F
        PltPts=Rpl*X((iF-1)*Nx+1:iF*Nx,:);
        % Plot any periodic copies that fall in the box
        Shifts = unique(Ld*floor(PltPts/Ld),'Rows');
        nS=size(Shifts,1);
        for iS=1:nS
            plot3(PltPts(:,1)-Shifts(iS,1),PltPts(:,2)-Shifts(iS,2),...
                PltPts(:,3)-Shifts(iS,3),'-','Color',[0 0.4470 0.7410])
            hold on
        end
%         if (iF>2)
%             PltPts=Rpl*X0((iF-1)*Nx+1:iF*Nx,:);    
%             plot3(PltPts(:,1),PltPts(:,2),PltPts(:,3),':','Color',[0.8500 0.3250 0.0980])
%         end
    end
    view(3)
    % Plot links
%     for LinkMot=[0 1]
%         if (LinkMot==0)
%             Links = load(strcat(Name,'_MotStep',num2str(iT-1),'_',num2str(seed),'.txt'));
%             pChars="-ko";
%         else
%             Links = load(strcat(Name,'_CLStep',num2str(iT-1),'_',num2str(seed),'.txt'));
%             pChars="-rs";
%         end
%         nL=Links(1,1);
%         nLTime(iT)=nL;
%         for iL=1:nL
%             iPt = Links(iL+1,1);
%             jPt = Links(iL+1,2);
%             shift = Links(iL+1,3:end);
%             iFib = floor(iPt/Nu+1e-6);
%             jFib = floor(jPt/Nu+1e-6);
%             iSite = iPt-iFib*Nu;
%             jSite = jPt-jFib*Nu;
%             X1 = Ru*X(iFib*Nx+1:(iFib+1)*Nx,:);
%             X2 = Ru*X(jFib*Nx+1:(jFib+1)*Nx,:)+shift;
%             PltPts = [X1(iSite+1,:); X2(jSite+1,:)];
%             % Plot any periodic copies that fall in the box
%             Shifts = unique(Ld*floor(PltPts/Ld),'Rows');
%             nS=size(Shifts,1);
%             for iS=1:nS
%                 plot3(PltPts(:,1)-Shifts(iS,1),PltPts(:,2)-Shifts(iS,2),...
%                     PltPts(:,3)-Shifts(iS,3),pChars)
%                 hold on
%             end
%         end
%     end
    xlim([0 Ld])
    ylim([0 Ld])
    zlim([0 Ld])
    PlotAspect;
    movieframes(iT)=getframe(f);
    hold off
end
%MFreeBd = load('Mot_FreeLinkBound_1.txt');
%CFreeBd = load('CL_FreeLinkBound_1.txt');