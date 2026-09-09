names = ["MCMC2_Nx8_Lp1.mat" "MCMC2_Nx16_Lp1.mat" "MCMC2_Nx24_Lp1.mat"];
%tiledlayout(2,2,'Padding', 'none', 'TileSpacing', 'compact');
for iName = 1:length(names)
load(names(iName))
MC = mean(AllTanVecDots);
SC = 2*std(AllTanVecDots)/sqrt(nTrial);
% Tangent vector correlations
nexttile(3) 
Colors=get(gca,'ColorOrder');
fill([Deltas', fliplr(Deltas')], [MC-SC, fliplr(MC+SC)],...
    Colors(iName,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(Deltas,MC,'-','Color',Colors(iName,:),'LineWidth',2)
pbaspect([1 1 1])

% End to end distance
nBee = size(AllEndToEndDists,2);
emp = (0.5:nBee)/size(AllEndToEndDists,2);
AllEndToEndDists=AllEndToEndDists./(sum(AllEndToEndDists')'*1/nBee);
MC = mean(AllEndToEndDists);
SC = 2*std(AllEndToEndDists)/sqrt(nTrial);
nexttile(4)
fill([emp, fliplr(emp)], [MC-SC, fliplr(MC+SC)],...
    Colors(iName,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(emp,MC,'-','Color',Colors(iName,:),'LineWidth',2)
hold on
xlabel('$r/L$','interpreter','latex')
ylabel('PDF')
title('End-to-end distance')
pbaspect([1 1 1])

end
nexttile(3)
diffc=(0:0.001:L);
plot(diffc,exp(-diffc/lp),':k')
xlabel('$\Delta s/L$','interpreter','latex')
ylabel('$\langle \tau(s+\Delta s) \cdot \tau(s) \rangle$','interpreter','latex')
title('Tangent vector correlation')
nexttile(4)
dr=1e-5;
r = (0.5:1/dr)'*dr;
G = zeros(length(r),1);
for ell=1:3
    G=G+1./(lpstar*(1-r)).^(3/2).*exp(-(ell-1/2)^2./(lpstar*(1-r))).*...
        (4*((ell-1/2)./sqrt(lpstar*(1-r))).^2-2);
end
% Estimate integral of G, normalize to 1
G=G.*r.^2;
G = G/sum(G*dr);
plot(r,G,':k');
xlabel('$r/L$','interpreter','latex')
ylabel('PDF')
title('End-to-end distance')
pbaspect([1 1 1])