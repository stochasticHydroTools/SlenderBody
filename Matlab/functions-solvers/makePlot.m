% Make the plot for the SBT simulations
% Plot the fiber first
hold on
if exist('nSpecies')
    nColors = max(allnewlabels);
    Corder = jet(nColors);
else
    nColors = max(newlabels);
    Corder =  jet(nColors);%
%     Corder = [0    0.4470    0.7410;
%     0.8500    0.3250    0.0980;
%     0.9290    0.6940    0.1250];
end
% Corder = get(gca, 'ColorOrder');
% Plot the fibers
plotfibs=1:nFib;
%plotlinks=(1:2)';    
thk = 1; % line thickness
%CoeffsToValsCheb = cos(acos(2*(s0/Lf)-1).*(0:N-1));
%CoeffstoValsUniform = cos(acos(1).* (0:N-1));
%ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);
ends=zeros(nFib,3);
for iFib=plotfibs    
%     if (length(links) > 0)
%         plotlinks = [plotlinks; find(links(:,1)==iFib); find(links(:,3)==iFib)];
%     end
    inds=(iFib-1)*3*N+1:3:iFib*3*N;
    scinds = (iFib-1)*N+1:iFib*N;
    fp=[Xt(inds) Xt(inds+1) Xt(inds+2)];
    if (newlabels(iFib)==-1)
        line=':';
        fibcolor='k';
        thk=0.5;
    else
        fibcolor = Corder(mod(newlabels(iFib),nColors)+1,:);
        if (exist('nSpecies'))
            %fibcolor = Corder(iSpecies,:);
            if (iSpecies==1)
                line='-';
                thk=1.0;
            else
                line='--';
                thk=2.0;
            end
        else
            fibcolor = Corder(mod(newlabels(iFib),nColors)+1,:);
            line='-';
        end
        %line='-';
    end
%      fibcolor = Corder(mod(iFib,7)+(mod(iFib,7)==0)*7,:);
%      line='-';
    scatter3(Rpl*fp(:,1),Rpl*fp(:,2),Rpl*fp(:,3), 12,Rpl*theta_s(scinds),'filled')
%     plot3(Rpl*fp(:,1),Rpl*fp(:,2),Rpl*fp(:,3))
    caxis([-2*pi 2*pi])
    hold on
    indices=count*N+1:(count+1)*N;
    %plot3(Rpl*X2(indices,1),Rpl*X2(indices,2),Rpl*X2(indices,3))
    %plot3(Rpl*X1(indices,1),Rpl*X1(indices,2),Rpl*X1(indices,3))
    [~,shifts] = calcShifted(fp,gn,Ld,Ld,Ld);
    [sx,sy,sz]=meshgrid(unique(shifts(:,1)),unique(shifts(:,2)),unique(shifts(:,3)));
    sx=sx(:); sy=sy(:); sz=sz(:);
%     sx = 0; sy=0; sz =0;
%     for iS=1:length(sx)
%         PlotLocs = [Xt(inds)-Ld*(sx(iS)+gn*sy(iS)) Xt(inds+1)-Ld*sy(iS) Xt(inds+2)-Ld*sz(iS)];
%         PlotLocsPrime = ([1 -gn 0; 0 1 0; 0 0 1]*PlotLocs')'; 
%         PlotInds = (PlotLocsPrime(:,1) > -Ld/2 & PlotLocsPrime(:,1) < Ld/2 & ...
%             PlotLocsPrime(:,2) > -Ld/2 & PlotLocsPrime(:,2) < Ld/2 & ...
%             PlotLocsPrime(:,3) > -Ld/2 & PlotLocsPrime(:,3) < Ld/2);
%         PlotInds=1:N; % all
%         plot3(PlotLocs(PlotInds,1),PlotLocs(PlotInds,2),PlotLocs(PlotInds,3),...
%         line,'LineWidth',thk,'Color',fibcolor)
%     end
end
%scatter3(X2mp(1),X2mp(2),X2mp(3),'x','MarkerEdgeColor',XColor,'LineWidth',2)
% % Plot the cross linkers
[nLinks,~]=size(links);
if (nLinks > 0)
for iL=1:nLinks
    iFib=links(iL,1);
    jFib=links(iL,3);
    inds1=(iFib-1)*3*N+1:3:iFib*3*N;
    inds2=(jFib-1)*3*N+1:3:jFib*3*N;
    shift2 = Ld*[links(iL,5)+gn*links(iL,6) links(iL,6) links(iL,7)];
    fp=[Xt(inds1) Xt(inds1+1) Xt(inds1+2); ...
        Xt(inds2)-shift2(1) Xt(inds2+1)-shift2(2) Xt(inds2+2)-shift2(3)];
    [~,shifts] = calcShifted(fp,gn,Ld,Ld,Ld);
    [sx,sy,sz]=meshgrid(unique(shifts(:,1)),unique(shifts(:,2)),unique(shifts(:,3)));
    sx=sx(:); sy=sy(:); sz=sz(:);
%     [~,n1] = min(abs(s0-links(iL,2)));
%     ind1 = (iFib-1)*3*N+3*n1-2;
%     [~,n2] = min(abs(s0-links(iL,4)));
%     ind2 = (jFib-1)*3*N+3*n2-2;
    sx=0; sy=0; sz=0;
%     for iS=1:length(sx)
%         if (thk==0.5)
%             plot3(Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds1+1)-Ld*sy(iS),Xt(inds1+2)-Ld*sz(iS),...
%                 '-','LineWidth',thk,'Color','g')
%             plot3(Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds2+1)-shift2(2)-Ld*sy(iS),...
%                 Xt(inds2+2)-shift2(3)-Ld*sz(iS),'-','LineWidth',thk,...
%                 'Color','g')
%         else
%             plot3(Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds1+1)-Ld*sy(iS),Xt(inds1+2)-Ld*sz(iS),...
%                 '-','LineWidth',thk,'Color',Corder(mod(iFib,7)+(mod(iFib,7)==0)*7,:))
%             plot3(Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds2+1)-shift2(2)-Ld*sy(iS),...
%                 Xt(inds2+2)-shift2(3)-Ld*sz(iS),'-','LineWidth',thk,...
%                 'Color',Corder(mod(jFib,7)+(mod(jFib,7)==0)*7,:))
%         end
        iS=1;
        X1 = [Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)) Xt(inds1+1)-Ld*sy(iS) Xt(inds1+2)-Ld*sz(iS)];
        X2 = [Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)) Xt(inds2+1)-shift2(2)-Ld*sy(iS) Xt(inds2+2)-shift2(3)-Ld*sz(iS)];
        % Find points on fibers where CL is bound
        th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
        Lmatn = (cos((0:N-1).*th));
        % Calculate the force density on fiber 1
        s1star=links(iL,2);
        th1 = acos(2*s1star/Lf-1);%(links(iL,1))-1)';
        U1 = (cos((0:N-1).*th1));
        X1star = U1*(Lmatn \ X1);
        s2star=links(iL,4);
        th2 = acos(2*s2star/Lf-1);%(links(iL,3))-1)';
        U2 = (cos((0:N-1).*th2));
        X2star = U2*(Lmatn \ X2);
        Linkpts = [X1star;X2star];
        plot3(Linkpts(:,1),Linkpts(:,2),Linkpts(:,3),'-k','LineWidth',1)
%     end
end
end
str=sprintf('$t=$ %1.4f s', t);
title(str,'Interpreter','latex')
view([-0.2682   49.0867])
%view(2)
%view([12 40])
%view([30 15])
%view([60 30])
% xlim([-Ld/2 Ld/2])
% ylim([-Ld/2 Ld/2])
% zlim([-Ld/2 Ld/2])
% pbaspect([1 1 2])
% view([-46.9081   30.2241])
xlim([-L/2 L/2])
ylim([-L/2 L])
zlim([-L/2 L/2])
% pbaspect([0.6 0.6 2])
% xlim([-L L])
% ylim([-L L])
pbaspect([1 1.5 1])
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
zlabel('$z$','interpreter','latex')
set(gca,'FontName','times New roman','FontSize',14)
hold off