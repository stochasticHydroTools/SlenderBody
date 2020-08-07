close all;
clear vels fibmaxes meanstrains meancurves
nFib=700;
N=16;
Ld=4;
step=N*nFib;
rl=0.5;
Lf = 2.0;
nCL = 8400;
for omHz =[1]
% strains=load(strcat('DynamicLinkStrainsHYDROOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% curves=load(strcat('DynamicFibCurvesHYDROOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
CLstressL=load(strcat('CLStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
LamstressL=load(strcat('LambdaStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
ElstressL=load(strcat('ElasticStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% strains=load(strcat('DynamicLinkStrainsOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% curves=load(strcat('DynamicFibCurvesOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% CLstress=load(strcat('CLStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% Lamstress=load(strcat('LambdaStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
% Elstress=load(strcat('ElasticStressOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'.txt'));
stress = CLstress+Lamstress+Elstress;
s0=chebpts(N,[0 Lf],1);
for iT=1:length(strains)/nCL
    meanstrains(iT) = mean(abs(strains((1:nCL)+nCL*(iT-1))));
    meancurves(iT) = mean(curves((1:nFib)+nFib*(iT-1)));
end
% eval(strcat('meanstrainsOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'=meanstrains;'))
% eval(strcat('meancurvesOm',num2str(omHz),'F',num2str(nFib),'C',num2str(nCL),'=meancurves;'))
end
stress=CLstress+Lamstress+Elstress;

% for iCyc=1:100
% mcurves_Om01(iCyc)=mean(meancurvesOm1em1F150C1200((1:20)+(iCyc-1)*20));
% mstrains_Om01(iCyc)=mean(meanstrainsOm1em1F150C1200((1:20)+(iCyc-1)*20));
% end
% for iCyc=1:100
% mcurves_Om1(iCyc)=mean(meancurvesOm10em1F150C1200((1:4)+(iCyc-1)*4));
% mstrains_Om1(iCyc)=mean(meanstrainsOm10em1F150C1200((1:4)+(iCyc-1)*4));
% end
% for iCyc=1:100
% mcurves_Om10(iCyc)=mean(meancurvesOm100em1F150C1200((1:10)+(iCyc-1)*10));
% mstrains_Om10(iCyc)=mean(meanstrainsOm100em1F150C1200((1:10)+(iCyc-1)*10));
% end
% plot(mcurves_Om10(1:100),'LineWidth',2.0)
% hold on
% plot(mcurves_Om1(1:100),'LineWidth',2.0)
% plot(mcurves_Om01(1:100),'LineWidth',2.0)
% legend({'$\omega=20\pi$','$\omega = 2\pi$','$\omega=0.2\pi$'},'interpreter','latex')
% xlabel('Cycle number','interpreter','latex')
% ylabel('Mean curvature over cycle','interpreter','latex')
% set(gca,'FontName','times New roman','FontSize',14)
% 
% plot((1:100)*0.1,mstrains_Om10(1:100),'LineWidth',2.0)
% hold on
% plot(1:100,mstrains_Om1(1:100),'LineWidth',2.0)
% plot((1:100)*10,mstrains_Om01(1:100),'LineWidth',2.0)
% legend({'$\omega=20\pi$','$\omega = 2\pi$','$\omega=0.2\pi$'},'interpreter','latex')
% xlabel('Cycle number','interpreter','latex')
% ylabel('Mean curvature over cycle','interpreter','latex')
% set(gca,'FontName','times New roman','FontSize',14)
