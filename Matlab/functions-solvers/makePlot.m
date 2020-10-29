% Make the plot for the SBT simulations
% Plot the fiber first
clf;
hold on
Corder =  get(gca, 'ColorOrder');
% Plot the fibers
plotfibs=1:nFib;
plotlinks=[];    
thk = 1; % line thickness
CoeffsToValsCheb = cos(acos(2*(s0/Lf)-1).*(0:N-1));
CoeffstoValsUniform = cos(acos(1).* (0:N-1));
ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);
ends=zeros(nFib,3);
for iFib=plotfibs    
    if (length(links) > 0)
        plotlinks = [plotlinks; find(links(:,1)==iFib); find(links(:,3)==iFib)];
    end
    inds=(iFib-1)*3*N+1:3:iFib*3*N;
    fp=[Xt(inds) Xt(inds+1) Xt(inds+2)];
    plot3(fp(:,1),fp(:,2),fp(:,3), '-','LineWidth',thk,'Color',Corder(mod(iFib,7)+(mod(iFib,7)==0)*7,:))
    [~,shifts] = calcShifted(fp,gn,Ld,Ld,Ld);
    [sx,sy,sz]=meshgrid(unique(shifts(:,1)),unique(shifts(:,2)),unique(shifts(:,3)));
    sx=sx(:); sy=sy(:); sz=sz(:);
    for iS=1:length(sx)
        PlotLocs = [Xt(inds)-Ld*(sx(iS)+gn*sy(iS)) Xt(inds+1)-Ld*sy(iS) Xt(inds+2)-Ld*sz(iS)];
        PlotLocsPrime = ([1 -gn 0; 0 1 0; 0 0 1]*PlotLocs')'; 
        PlotInds = (PlotLocsPrime(:,1) > -Ld/2 & PlotLocsPrime(:,1) < Ld/2 & ...
            PlotLocsPrime(:,2) > -Ld/2 & PlotLocsPrime(:,2) < Ld/2 & ...
            PlotLocsPrime(:,3) > -Ld/2 & PlotLocsPrime(:,3) < Ld/2);
        PlotInds=1:N; % all
        if (thk==0.5)
            plot3(PlotLocs(PlotInds,1),PlotLocs(PlotInds,2),PlotLocs(PlotInds,3),...
            '-','LineWidth',thk,'Color','g')
        else 
            plot3(PlotLocs(PlotInds,1),PlotLocs(PlotInds,2),PlotLocs(PlotInds,3),...
            '-','LineWidth',thk,'Color',Corder(mod(iFib,7)+(mod(iFib,7)==0)*7,:))
        end
    end
end
% % Plot the cross linkers
[nLinks,~]=size(links);
if (nLinks > 0)
for iL=unique(plotlinks')
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
    [~,n1] = min(abs(s0-links(iL,2)));
    ind1 = (iFib-1)*3*N+3*n1-2;
    [~,n2] = min(abs(s0-links(iL,4)));
    ind2 = (jFib-1)*3*N+3*n2-2;
    sx=0;
    sy=0;
    sz=0;
    for iS=1:length(sx)
        if (thk==0.5)
            plot3(Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds1+1)-Ld*sy(iS),Xt(inds1+2)-Ld*sz(iS),...
                '-','LineWidth',thk,'Color','g')
            plot3(Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds2+1)-shift2(2)-Ld*sy(iS),...
                Xt(inds2+2)-shift2(3)-Ld*sz(iS),'-','LineWidth',thk,...
                'Color','g')
        else
            plot3(Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds1+1)-Ld*sy(iS),Xt(inds1+2)-Ld*sz(iS),...
                '-','LineWidth',thk,'Color',Corder(mod(iFib,7)+(mod(iFib,7)==0)*7,:))
            plot3(Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds2+1)-shift2(2)-Ld*sy(iS),...
                Xt(inds2+2)-shift2(3)-Ld*sz(iS),'-','LineWidth',thk,...
                'Color',Corder(mod(jFib,7)+(mod(jFib,7)==0)*7,:))
        end
        plot3([Xt(ind1) Xt(ind2)-shift2(1)]-Ld*(sx(iS)+gn*sy(iS)),...
            [Xt(ind1+1) Xt(ind2+1)-shift2(2)]-Ld*(sy(iS)), ...
            [Xt(ind1+2) Xt(ind2+2)-shift2(3)]-Ld*(sz(iS)),...
            '--k','LineWidth',0.1)
    end
end
end
str=sprintf('$t=$ %1.2f s', t);
title(str,'Interpreter','latex')
%view(3)
%view([48.86 14.73])
% view([60 30])
% xlim([-Ld/2 Ld/2])
% ylim([-Ld/2 Ld/2])
% zlim([-Ld/2 Ld/2])
% xlim([-1.2 1])
% ylim([-1.2 1])
% zlim([-1.2 1])
% view(2)
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
zlabel('$z$','interpreter','latex')
pbaspect([1 1 1])
set(gca,'FontName','times New roman','FontSize',14)
movieframes(length(movieframes)+1)=getframe(f);
hold off