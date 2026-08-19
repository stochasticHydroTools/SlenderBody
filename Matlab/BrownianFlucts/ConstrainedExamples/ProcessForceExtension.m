% Langevin dynamics
nTrial=100;
Nx=16;
L=1;
lp=10;
Force = 0;
FSqRts = Force;
dts = [1e-3 1e-4 1e-5];
MeanExtension = zeros(nTrial,length(dts));
Npl=101;
MeanCorSize = zeros(nTrial,length(dts));
MeanTauSq=zeros(Nx-1,nTrial,length(dts));
[sX,wX,bX]=chebpts(Nx,[0 L]);
DX = diffmat(Nx,1,[0 L],'chebkind2');
spl=(0:Npl-1)'/(Npl-1);
Rpl = barymat(spl,sX,bX);

for iDT=1:length(dts)
dt=dts(iDT);
for jj=1:nTrial
    load(strcat('Clamped1_Lp',num2str(lp),'_F',num2str(Force),...
        '_Nx',num2str(Nx),'_Dt',num2str(dt),'_Seed',num2str(jj),'.mat'))
    sC=s;
    sC(1)=0;
    sC(end)=L;
    ToTauGrid=barymat(sC,sX,bX);
    nT = length(Xpts)/Nx-1;
    BurnIn=0.2*nT;
    nObs = nT-BurnIn;
    for jT=BurnIn+1:nT
        ThesePts = Xpts((jT-1)*Nx+1:jT*Nx,:);
        TauThis= ToTauGrid*DX*ThesePts;
        TauThis = TauThis.^2;
        TauSqExt = 1/2*(TauThis(:,1)+TauThis(:,3));
        MeanTauSq(:,jj,iDT)=MeanTauSq(:,jj,iDT)+TauSqExt/nObs;
        MeanExtension(jj,iDT) = MeanExtension(jj,iDT) + ThesePts(end,2)/nObs;
    end
    %MeanCorSize(jj,iDT)=MeanOmTurn;
end
end
nError = 5;
nPerError = nTrial/nError;
MeanMeanExtension = zeros(nError,length(dts));
MeanMeanTau = zeros(Nx-1,nError,length(dts));
MeanMeanEr = zeros(nError,length(dts));
PlTau = zeros(length(spl),nError,length(dts));
for k = 1:nError
    MeanMeanExtension(k,:)=mean(MeanExtension((k-1)*nPerError+1:k*nPerError,:));
    MeanMeanTau(:,k,:) = mean(MeanTauSq(:,(k-1)*nPerError+1:k*nPerError,:),2);
    MeanMeanEr(k,:)=mean(MeanCorSize((k-1)*nPerError+1:k*nPerError,:));
end
ConstrToPl = barymat(spl,s,b)*ConstrToCheb;

% Theory
% figure(1)
% errorbar(FSqRts,mean(MeanMeanExtension),2*std(MeanMeanExtension)/sqrt(nError),...
%     'o','LineWidth',2.0)
AllFs=1:10;
ThExt = 1-(AllFs*L.*cosh(AllFs*L)-sinh(AllFs*L))./...
    (2*AllFs.^2*lp*L.*sinh(AllFs*L));
% hold on
% plot(AllFs,ThExt)
% xlabel('$\sqrt{F/\ell_p k_B T}$')
% ylabel('Mean extension')

% Taus 
s=0:0.001:L;
x1 = s/L - (2*FSqRts*s*cosh(FSqRts*L)-sinh(FSqRts*L)+sinh(FSqRts*(L-2*s)))...
    ./(4*FSqRts.^2*lp*L*sinh(FSqRts*L));
xTrans = sqrt((2*FSqRts*s*sinh(FSqRts*L) - 3*cosh(FSqRts*L)+4*cosh(FSqRts*(L-s))...
    -cosh(FSqRts*(L-2*s)))./(2*FSqRts.^3*L^2*lp*sinh(FSqRts*L)));
TauTrans = sinh(FSqRts*(L-s)).*sinh(FSqRts*s)/(FSqRts*lp*sinh(FSqRts*L));
if (FSqRts==0)
    TauTrans=((L - s).*s)/(L*lp);
end
if (1)
plot(s,TauTrans,'-k')
hold on
set(gca,'ColorOrderIndex',1)
end
DefColors=get(gca,'ColorOrder');

if (1)
for iT=1:length(dts)
AvgVec = MeanMeanTau(:,:,iT)';
AvgPl = AvgVec*ConstrToPl';
fill([spl', fliplr(spl')], [mean(AvgPl)-2*std(AvgPl)/sqrt(nError),...
    fliplr(mean(AvgPl)+2*std(AvgPl)/sqrt(nError))],DefColors(iT,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(spl,mean(AvgPl),':','Color',DefColors(iT,:),'LineWidth',2)
plot(sC,mean(AvgVec),'o','Color',DefColors(iT,:),'LineWidth',2)
end
end

% MCMC
Nxs = [Nx];
cls=['1'];
for iT=1
DefColors(iT,:)=[0 0 0];    
load(strcat('MCMC',cls(iT),'_Clamp_Nx',num2str(Nxs(iT)),'_Lp',num2str(lp),'.mat'))
MMTau=mean(MeanTauSq(:,[1 3],:),2);
MTau = barymat(spl,s,b)*ConstrToCheb*mean(MMTau,3);
STau=2*barymat(spl,s,b)*std(MMTau,[],3)/sqrt(nTrial);
fill([spl', fliplr(spl')], [MTau'-2*STau',...
    fliplr(MTau'+2*STau')],DefColors(iT,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(spl,MTau,':','Color',DefColors(iT,:),'LineWidth',2)
plot(sC,mean(MMTau,3),'o','Color',DefColors(iT,:),'LineWidth',2)
end