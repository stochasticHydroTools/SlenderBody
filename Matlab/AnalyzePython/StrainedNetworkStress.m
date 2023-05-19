addpath(genpath('../../Python'))
StressAcor = 0;
GetStats = 1;
BatchPlot = 0;
F = 200;
N = 13;
omHzs = [1];
dtsave=2.5e-2;
gams=[0.25];
midpointStress = 0;
microdt = 1e-4;
for iOm = 1:length(omHzs)
OmHz = omHzs(iOm);
nSaves = 100;
for iGam = 1:length(gams)
maxstr = gams(iGam);
for seed = 1:5
OmHz = omHzs(iOm);
%FileName = strcat('FluctOm1Lp17.0Ld2Turn2.5_',num2str(seed),'.txt');
FileName = strcat('NLOm1DetLp1.7Ld2Turn10_',num2str(seed),'.txt');
% Locs=load(strcat('Locs',FileName));
CLstress=load(strcat('CLStress',FileName));
try
Driftstress=load(strcat('DriftStress',FileName));
catch
Driftstress = 0*CLstress;
end
Lamstress=load(strcat('LamStress',FileName));
Elstress=load(strcat('ElStress',FileName));
str = Driftstress+CLstress+Lamstress+Elstress;
Tf = 10/OmHz;
dtstr = 1/(OmHz)/nSaves;
tarray=(dtstr:dtstr:Tf)'-0.5*microdt;
if (~midpointStress)
    tarray=(dtstr:dtstr:Tf)'-microdt;
end
[Gp(seed,iOm),Gdp(seed,iOm),er(seed,:)] =getModuli(str,Tf,tarray,2*pi*OmHz,maxstr);
%allstress(seed,:)=str;
if (StressAcor)
    fittime = 5;
    if (maxstr==0)
        er(seed,:)=str;
        OmHz=0;
        fittime = 0.5;
    elseif (maxstr==0.01 || OmHz < 0.3)
        fittime = 0.5;
        OmHz=0;
    end
    [a(seed,:),fitparams(seed,:),fitvals(seed,:)]=fitStressEr(er(seed,:),fittime/dtstr,dtstr,OmHz);
end
% tarray=(dtstr:dtstr:1/OmHz)'-0.5*microdt;
% if (~midpointStress)
%     tarray=(dtstr:dtstr:1/OmHz)'-microdt;
% end
% nStat = 16;
% for iCyc=1:Tf*OmHz
%     [AllGp(seed,iCyc),AllGdp(seed,iCyc),~]=getModuli(str(nSaves*(iCyc-1)+1:nSaves*iCyc),1/OmHz,tarray,2*pi*OmHz,maxstr);
%     %AllLinkStr(seed,iCyc) = mean(LinkStrains(nStat*(iCyc-1)+1:nStat*iCyc));
% end
if (BatchPlot)
% Blocked version
    nBatch=2;
    nCyc = Tf*OmHz/nBatch;
    step = max(1,floor(nCyc/120));
    GpC1 = zeros(Tf*OmHz/nBatch,1); GdpC1 = zeros(Tf*OmHz/nBatch,1); 
    GpC2 = zeros(Tf*OmHz/nBatch,1); GdpC2 = zeros(Tf*OmHz/nBatch,1);
    for nCyc=1:step:Tf*OmHz/nBatch
        newTf=nCyc/OmHz;
        dtstr = 1/(OmHz)/nSaves;
        tarray=(dtstr:dtstr:newTf)'-0.5*microdt;
        if (~midpointStress)
            tarray=(dtstr:dtstr:newTf)'-microdt;
        end
        [GpC1(nCyc),GdpC1(nCyc)]=getModuli(str(1:length(tarray)),newTf,tarray,2*pi*OmHz,maxstr);
        [GpC2(nCyc),GdpC2(nCyc)]=getModuli(str(1*length(str)/nBatch+1:length(str)/nBatch+length(tarray)),...
            newTf,tarray,2*pi*OmHz,maxstr);
    end
    plotindex = mod(indexTrial,4)+1;
    subplot(2,2,plotindex)
    title(strcat('$\omega=$',num2str(OmHz)),'interpreter','latex')
    set(gca,'ColorOrderIndex',1)
    cyctimes = (1:Tf*OmHz/nBatch)/OmHz;
    plot(cyctimes(GpC1~=0),GpC1(GpC1~=0),'-')
    hold on
    set(gca,'ColorOrderIndex',1)
    plot(cyctimes(GpC1~=0),GpC2(GpC1~=0),'--')
    set(gca,'ColorOrderIndex',2)
    plot(cyctimes(GpC1~=0),GdpC1(GpC1~=0),'-')
    hold on
    set(gca,'ColorOrderIndex',2)
    plot(cyctimes(GpC1~=0),GdpC2(GpC1~=0),'--')
    xlabel('$t$ (s)','interpreter','latex')
    ylabel('$G''$','interpreter','latex')
end
end
if (StressAcor)
    tableline=zeros(1,8);
    tableline(1:2:end)=mean(fitparams);
    tableline(2:2:end)=2*std(fitparams)/sqrt(5);
%     subplot(1,2,1)
%     hold on
    errorbar(0:dtstr:fittime,mean(a),2*std(a)/sqrt(5))
    plot(0:dtstr:fittime,mean(fitvals))
    xlim([0 fittime])
    title(strcat('$\gamma=$',num2str(maxstr)),'interpreter','latex')
end
%meanstress(iGam,:)=mean(allstress);
%meanstressHat(iGam,:)=fftshift(fft(mean(allstress)));
clear er fitparams a fitvals
end
end

   
%tableline
%Code for stress spectrum
% every = 120;
% ks=(-length(str)/2:length(str)/2-1)*nSaves/length(str);
% plot(ks(1:every:end),abs(meanstressHat(5,1:every:end))/0.15,':o')
% hold on
% plot(ks(1:every:end),abs(meanstressHat(4,1:every:end))/0.10,':o')
% set(gca, 'YScale', 'log')
% plot(ks(1:every:end),abs(meanstressHat(3,1:every:end))/0.05,':o')
% plot(ks(1:every:end),abs(meanstressHat(2,1:every:end))/0.025,':o')
% plot(ks(1:every:end),abs(meanstressHat(1,1:every:end))/0.01,':o')
% xlim([-10 10])
% xlabel('$k$','interpreter','latex')
% ylabel('$||\hat{\sigma}(k)||/\gamma$','interpreter','latex')
% legend('$\gamma=0.15$','$\gamma=0.1$','$\gamma=0.05$','$\gamma=0.025$','$\gamma=0.01$','interpreter','latex')
