% Langevin dynamics
nTrial=30;
Ns=[16 16];
Force = 3;
FSqRts = Force;
dts = [1e-3 2.5e-5];
MeanExtension = zeros(nTrial,length(dts));
MeanCorSize = zeros(nTrial,length(dts));
MeanTauSq=zeros(Npl,3,nTrial,length(dts));
for iDT=1:length(dts)
dt=dts(iDT);
N = Ns(iDT);
for j=1:nTrial
    % try
    load(strcat('Type1_N',num2str(N),...
        '_Dt',num2str(dt),'_Seed',num2str(j),'.mat'))
    % catch
    % load(strcat('Lp3_N',num2str(N),...
    %     '_Dt',num2str(dt),'_Seed',num2str(j),'.mat'))
    % end
    nT = length(Xpts)/Nx-1;
    BurnIn=0.5*nT;
    nObs = nT-BurnIn;
    for jT=BurnIn+1:nT
        ThesePts = Xpts((jT-1)*Nx+1:jT*Nx,:);
        TauThis= RNp1ToN*DNp1*ThesePts;
        MeanTauSq(:,:,j,iDT)=MeanTauSq(:,:,j,iDT)+barymat(spl,s,b)*TauThis.^2/nObs;
        MeanExtension(j,iDT) = MeanExtension(j,iDT) + ThesePts(end,2)/nObs;
    end
    MeanCorSize(j,iDT)=mean(MeanOmTurn);
end
end
nError = 3;
nPerError = nTrial/nError;
MeanMeanExtension = zeros(nError,length(dts));
MeanMeanTau = zeros(Npl,3,nError,length(dts));
MeanMeanCor = zeros(nError,length(dts));
PlTau = zeros(length(spl),nError,length(dts));
for k = 1:nError
    MeanMeanExtension(k,:)=mean(MeanExtension((k-1)*nPerError+1:k*nPerError,:));
    MeanMeanCor(k,:) = mean(MeanCorSize((k-1)*nPerError+1:k*nPerError,:));
    MeanMeanTau(:,:,k,:) = mean(MeanTauSq(:,:,(k-1)*nPerError+1:k*nPerError,:),3);
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
for iT=1:length(dts)
plot(spl,mean(PlTau(:,:,iT),2));
hold on
set(gca,'ColorOrderIndex',get(gca,'ColorOrderIndex')-1)
errorbar(spl(iT:5:end),mean(PlTau(iT:5:end,:,iT),2),...
    2*std(PlTau(iT:5:end,:,iT),[],2)/sqrt(nError),'o','LineWidth',2.0,...
    'MarkerSize',0.5)
end

