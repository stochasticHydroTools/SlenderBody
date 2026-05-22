% Langevin dynamics
nTrial=100;
Ns=[8 16];
L=1;
lp=2;
Force = 3;
FSqRts = Force;
dts = [1e-3 1e-5];
MeanExtension = zeros(nTrial,length(dts));
MeanTay = zeros(nTrial,length(dts));
Npl=101;
MeanCorSize = zeros(nTrial,length(dts));
MeanTauSq=zeros(Npl,3,nTrial,length(dts));
for iDT=1:length(dts)
dt=dts(iDT);
Nx = Ns(iDT);
[sX,wX,bX]=chebpts(Nx,[0 L]);
DX = diffmat(Nx,1,[0 L],'chebkind2');
spl=(0:Npl-1)'/(Npl-1);
Rpl = barymat(spl,sX,bX);
for jj=1:nTrial
    % try
    load(strcat('Clamped_Nx',num2str(Nx),...
        '_Dt',num2str(dt),'_Seed',num2str(jj),'.mat'))
    nT = length(Xpts)/Nx-1;
    BurnIn=0;%0.5*nT;
    nObs = nT-BurnIn;
    for jT=BurnIn+1:nT
        ThesePts = Xpts((jT-1)*Nx+1:jT*Nx,:);
        TauThis= Rpl*DX*ThesePts;
        MeanTauSq(:,:,jj,iDT)=MeanTauSq(:,:,jj,iDT)+(TauThis.^2)/nObs;
        MeanExtension(jj,iDT) = MeanExtension(jj,iDT) + ThesePts(end,2)/nObs;
    end
    MeanCorSize(jj,iDT)=MeanOmTurn;
end
end
nError = 5;
nPerError = nTrial/nError;
MeanMeanExtension = zeros(nError,length(dts));
MeanMeanTau = zeros(Npl,3,nError,length(dts));
MeanMeanEr = zeros(nError,length(dts));
PlTau = zeros(length(spl),nError,length(dts));
for k = 1:nError
    MeanMeanExtension(k,:)=mean(MeanExtension((k-1)*nPerError+1:k*nPerError,:));
    MeanMeanTau(:,:,k,:) = mean(MeanTauSq(:,:,(k-1)*nPerError+1:k*nPerError,:),3);
    MeanMeanEr(k,:)=mean(MeanCorSize((k-1)*nPerError+1:k*nPerError,:));
    for iT=1:length(dts)
        PlTau(:,k,iT)=mean(MeanMeanTau(:,[1 3],k,iT),2);
    end
end

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
plot(s,TauTrans,'-k')
hold on
set(gca,'ColorOrderIndex',1)
DefColors=get(gca,'ColorOrder');
for iT=1:length(dts)
AvgVec = PlTau(:,:,iT)';
fill([spl', fliplr(spl')], [mean(AvgVec)-2*std(AvgVec)/sqrt(nError),...
    fliplr(mean(AvgVec)+2*std(AvgVec)/sqrt(nError))],DefColors(iT,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(spl,mean(AvgVec),'-','Color',DefColors(iT,:),'LineWidth',2)
end

