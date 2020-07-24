% Make the plot - color code based on lambdas
% Plot the fiber first
hold on
[s0,wts]=chebpts(N,[0 Lf],1);
lambdas=lambdasH;
Xt = reshape(HydroLocs((1:step)+step*iT,:)',3*step,1);
strains = HStrains;
% lambdas=lambdasL;
% Xt = reshape(LocalLocs((1:step)+step*iT,:)',3*step,1);
% strains = LocStrains;
lamnorms = zeros(nFib,1);
for iFib=1:nFib
    fiblams = reshape(lambdas((iFib-1)*48+1:iFib*48,:),3,16)';
    normlamsSq = sum(fiblams.*fiblams,2);
    lamnorms(iFib)=1/Lf*sqrt(wts*normlamsSq);
end
Corder =  get(gca, 'ColorOrder');
% cm = colormap(hsv(1250));
% cm=cm(251:end,:);
% cli = colormap(hsv(1250));
% cli=cli(251:end,:);
cm =colormap(hsv(1000));
% Plot the fibers
plotfibs=57;
plotlinks=[];    
for iFib=plotfibs
    plotlinks = [plotlinks; find(links(:,1)==iFib); find(links(:,3)==iFib)];
end
linkedfibs = links(unique(plotlinks),3);
% % Plot the cross linkers
[nLinks,~]=size(links);
if (nLinks > 0)
for iL=unique(plotlinks')
    iFib=links(iL,1);
    jFib=links(iL,3);
    cmindex = ceil(lamnorms(jFib)/0.1795*1000);
    cellindex = ceil(abs(strains(iL))/0.125*1000);
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
    if (jFib==236)
    for iS=1:length(sx)
%         plot3(Xt(inds1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds1+1)-Ld*sy(iS),Xt(inds1+2)-Ld*sz(iS),...
%             '-','LineWidth',thk,'Color',cm(cmindex,:))
        plot3(Xt(inds2)-shift2(1)-Ld*(sx(iS)+gn*sy(iS)),Xt(inds2+1)-shift2(2)-Ld*sy(iS),...
            Xt(inds2+2)-shift2(3)-Ld*sz(iS),'-','LineWidth',thk,...
            'Color',cm(cmindex,:))
        plot3([Xt(ind1) Xt(ind2)-shift2(1)]-Ld*(sx(iS)+gn*sy(iS)),...
            [Xt(ind1+1) Xt(ind2+1)-shift2(2)]-Ld*(sy(iS)), ...
            [Xt(ind1+2) Xt(ind2+2)-shift2(3)]-Ld*(sz(iS)),'--k','LineWidth',0.2);
        %'Color',cli(cellindex,:),'LineWidth',1.0)
    end
    end
end
end
str=sprintf('$t=$ %1.2f s',t);
%title(str,'Interpreter','latex')
view(3)
view([70 20])
xlim([-Ld/2 Ld/2])
ylim([-Ld/2 Ld/2])
zlim([-Ld/2 Ld/2])
xlim([-3 1])
ylim([-1.5 2.5])
zlim([-2.5 1.5])
xlabel('$x$','interpreter','latex')
ylabel('$y$','interpreter','latex')
zlabel('$z$','interpreter','latex')
pbaspect([1 1 1])
set(gca,'FontName','times New roman','FontSize',14)
% hold off