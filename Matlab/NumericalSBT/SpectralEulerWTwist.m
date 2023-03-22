% % % Compute refined solution via Richardson
eps = 1e-3;
delta = 0;
N = 1/eps;
KirchoffFiberRectangularKeav_Twist;
lambdaEps = lambda;
MomentsEps = RectMoments;
u1OverEps = RectuEuler;
Om1OverEps = OmegaPar_Euler;
if (eps==1e-3)
    u1=u1OverEps; 
    Moments1 = MomentsEps;
    lambda1=lambdaEps;
    Omega1 = Om1OverEps;
    sRect=s_u;
else
    N = 2/eps;
    KirchoffFiberRectangularKeav_Twist;
    u1=RectuEuler;
    Moments1=RectMoments;
    lambda1 = lambda;
    sRect=s_u;
    Omega1 = OmegaPar_Euler;
end
if (eps==1e-3)
    N = 2/eps;
else
    N = 4/eps;
end
KirchoffFiberRectangularKeav_Twist;
u2=1/2*(RectuEuler(1:2:end,:)+RectuEuler(2:2:end,:));
Moments2=RectMoments;
lambda2 = 1/2*(lambda(1:2:end,:)+lambda(2:2:end,:));
Omega2 = 1/2*(OmegaPar_Euler(1:2:end)+OmegaPar_Euler(2:2:end));
uExtrap = (4*u2-u1)/3;
MomentsExtrap=(4*Moments2-Moments1)/3;
lambdaExtrap = (4*lambda2-lambda1)/3;
OmegaParExtrap = (4*Omega2-Omega1)/3;
s_u = sRect;
clearvars -except s_u MomentsExtrap uExtrap RectMoments eps delta u1 Moments1 lambdaExtrap lambda2 ...
     OmegaParExtrap lambdaEps MomentsEps u1OverEps lambdaExtrap Om1OverEps
strongthetaBC=0;
index=1;
nFib = 1;
NForSmalls = [4];
if (eps > 2e-3)
    NForSmalls = [12];
end
Ns = 8:8:48;
for NForSmall=NForSmalls
for N=Ns
L = 2;
mu = 1;
Eb = 1;
twmod = 1;
a = eps*L;
RectangularCollocation = 1;
[s,w,b] = chebpts(N,[0 L],1);
D = diffmat(N,[0 L],'chebkind1');
%warning('q=7 and twist mod is zero!')
q=1;
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
XMP=[0;0;0];
deltaLocal=0;
dt = 0;
clamp0 = 0;
TorqBC = 0;
strongthetaBC = 0;
TurnFreq=0;
clampL = 0;
initZeroTheta =1;
NupsampleHydro = 32;
InitFiberVarsNew;
theta_s = getTheta(sPsi);
X = reshape(Xt,3,[])';

XBC = UpsampleXBCMat*Xt + BCShift;
fE = FE*XBC;
% Calculate twisting force according to BC
theta_s_Psip2= UpsampleThetaBCMat \ [theta_s; PsiBC0; 0];
Theta_ss = DPsip2*theta_s_Psip2;
XBC3 = reshape(XBC,3,[])';
% Twist force computed on common N+5 grid, then downsampled to Nx
fTw3 = twmod*((RPsiToNp5*Theta_ss).*cross(DNp5*XBC3,DNp5^2*XBC3)+...
    (RPsiToNp5*theta_s_Psip2).*cross(DNp5*XBC3,DNp5^3*XBC3));
fTw = reshape((RNp5ToNp1*fTw3)',3*Nx,1);
nparTw = twmod*RPsiToNp1*Theta_ss;

% Compute mobilities 
Mtt = TransTransMobilityMatrix(X,a,L,mu,sNp1,bNp1,DNp1,AllbS_Np1,AllbD_Np1,NForSmall,0,0);
UTorq = UFromN(X,nparTw,DNp1,AllbS_Np1,a,L,mu,sNp1,bNp1,1,NForSmall);
UTorq = reshape(UTorq',[],1);

K = KonNp1(X_s,XonNp1Mat,I);
if (clamp0)
    K = K(:,1:3*N);
end
Kt = K'*WTilde_Np1;
RHS = Kt*(fE+fTw+(Mtt \ UTorq));
alphaU = lsqminnorm(Kt*Mtt^(-1)*K,RHS);
Omega = reshape(alphaU(1:3*N),3,N)';
uEuler = K*alphaU;
lambda = Mtt \ (uEuler-UTorq)-fE-fTw;

f = reshape(lambda+fE+fTw,3,[])';
OmegaRotTrans = OmegaFromF(X,f,DNp1,AllbS_Np1,a,L,mu,sNp1,bNp1,1,NForSmall);
OmegaRotTrans = RPsip2ToN*OmegaRotTrans;
OmegaRotRot = RPsip2ToN*Mrr_Psip2*twmod*DPsip2*theta_s_Psip2;
OmegaPar_Euler = OmegaRotRot+OmegaRotTrans;

[spl,wpl,bpl]=chebpts(1000,[0 L]);
Rpl = barymat(spl,sNp1,bNp1);
sv{index}=sNp1;
ws{index}=wNp1;
us{index}=Rpl*reshape(uEuler,3,Nx)';
lams{index}=reshape(lambda,3,Nx)';
%LamPerps{index}=sqrt(sum(LamPerps{index}.*LamPerps{index},2));
%plot(s,lambda(1:3:end))
% hold on
Moments{index}=zeros(3,N);
[s2,w2,b2]=chebpts(2*N,[0 L]);
R1To2 = barymat(s2,s,b);
R1To2s = stackMatrix(R1To2);
R2To1 = barymat(s,s2,b2);
R2To1s = stackMatrix(R2To1);
Xup = Rpl*X;
for iMoment=0:N-1
    Tk = cos(iMoment*acos(2*spl/L-1));
    Moments{index}(:,iMoment+1)=wpl*((Rpl*lams{index}).*Tk);
end

RplOm = barymat(spl,s,b);
Oms{index}=RplOm*OmegaPar_Euler;
index=index+1;
end
end

%% Convergence plot (U Omega)
subplot(1,2,1)
hold on
box on
set(gca,'YScale','Log')
uerrorsSpec=zeros(1,length(Ns)-1);
%ds = L/length(uExtrap);
nzation=sqrt(wpl*(sum(us{end}.*us{end},2)));
%RToUni = barymat(s_u,spl,bpl);
for iU=1:length(uerrorsSpec)
    %diffRect = RToUni*us{iU}-uExtrap;
    %uerrorsRect(iU)=sqrt(ds*sum(sum(diffRect.*diffRect,2)))/nzation;
    diffSpec = us{iU}-us{iU+1};
    uerrorsSpec(iU)=sqrt(wpl*sum(diffSpec.*diffSpec,2))/nzation;
    %boundary = spl/L < 0.05 | spl/L > 0.95;
    %uerrorsNoBd(iU)=sqrt(wpl(~boundary)*sum(diffSpec(~boundary).*diffSpec(~boundary),2))/nzation;
end
% 
try
    diff=uExtrap-u1OverEps;
catch
    diff=1/2*(uExtrap(1:2:end,:)+uExtrap(2:2:end,:))-u1OverEps;
end
normdiff=sum(diff.*diff,2);
EpsEr=sqrt(L/length(u1OverEps)*sum(normdiff))/nzation;
semilogy(Ns(1:end-1),uerrorsSpec,'-.o')
set(gca,'ColorOrderIndex',2)
plot(xlim,EpsEr*[1 1],'--')
xlabel('$N$','interpreter','latex')
ylabel('Relative $U$ error','interpreter','latex')
% %legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
% 
subplot(1,2,2)
Omerrors=zeros(1,length(Ns)-1);
nzation=sqrt(wpl*(sum(Oms{end}.*Oms{end},2)));
for iU=1:length(Omerrors)
    OmdiffSpec = Oms{iU}-Oms{iU+1};
    OmerrorsSpec(iU)=sqrt(wpl*sum(OmdiffSpec.*OmdiffSpec,2))/nzation;
end
try
    diff=OmegaParExtrap-Om1OverEps;
catch
    diff=1/2*(OmegaParExtrap(1:2:end,:)+OmegaParExtrap(2:2:end,:))-Om1OverEps;
end
normdiff=sum(diff.*diff,2);
EpsEr=sqrt(L/length(Om1OverEps)*sum(normdiff))/nzation;
semilogy(Ns(1:end-1),OmerrorsSpec,'-o')
hold on
% %semilogy(Ns(1:end-1),OmerrorsRect,'--o')
% %hold on
% %semilogy(Ns(1:end-1),OmerrorsNoBd,'-.s')
set(gca,'ColorOrderIndex',2)
plot(xlim,EpsEr*[1 1],'--')
xlabel('$N$','interpreter','latex')
ylabel('Relative $\Omega^\parallel$ error','interpreter','latex')
return

%% Lambda plots
% subplot(1,2,1)
% for ip=1:index-1
% plot(sv{ip}/L,lams{ip}(:,1))
% hold on
% end
% plot(s_u/L,lambdaExtrap(:,1),'--')
% % ylim([-10 10])
% xlabel('$s/L$','interpreter','latex')
% ylabel('$\lambda^{(x)}(s)$','interpreter','latex')
% return
% legend('$N=16$','$N=24$','$N=32$','$N=40$',...
%     'Richardson','2nd order $1/\epsilon$','interpreter','latex','numColumns',1)
%legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
% subplot(1,2,2)
% hold on
% box on
% set(gca,'YScale','Log')
% lamerrors=zeros(1,length(Ns)-1);
% nzation=sqrt(wpl*(sum(lams{end}.*lams{end},2)));
% %RToUni = barymat(s_u,spl,bpl);
% for iU=1:length(lamerrors)
%     diffSpec = lams{iU}-lams{iU+1};
%     lamerrors(iU)=sqrt(wpl*sum(diffSpec.*diffSpec,2))/nzation;
%     %diffRect = RToUni*lams{iU}-lambdaExtrap;
%     %lamerrorsRect(iU)=sqrt(ds*sum(sum(diffRect.*diffRect,2)))/nzation;
%     %boundary = s_u/L < 0.05 | s_u/L > 0.95;
%     %lamerrorsNoBd(iU)=sqrt(ds*sum(sum(diffRect(~boundary).*diffRect(~boundary),2)))/nzation;
% end
% try
%     diff=lambdaExtrap-lambdaEps;
% catch
%     diff=1/2*(lambdaExtrap(1:2:end,:)+lambdaExtrap(2:2:end,:))-lambdaEps;
% end
% normdiff=sum(diff.*diff,2);
% EpsEr=sqrt(L/length(lambdaEps)*sum(normdiff))/nzation;
% semilogy(Ns(1:end-1),lamerrors,'-o')
% % hold on
% % semilogy(Ns,lamerrorsNoBd,'-.s')
% % plot(xlim,EpsEr*[1 1],':k')
% xlabel('$N$','interpreter','latex')
% ylabel('Relative $\lambda$ error','interpreter','latex')
% legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
for iU=1:length(Ns)
plot(1:2:Ns(iU),(abs(Moments{iU}(1,2:2:end))),'-o')
hold on
end
plot(1:2:40,(abs(MomentsExtrap(1,2:2:end))),'-.s')
plot(1:2:40,(abs(MomentsEps(1,2:2:end))),':d')
set(gca,'YScale','Log')
xlabel('$k$','interpreter','latex')
ylabel('Moment','interpreter','latex')
legend('$N=8$','$N=16$','$N=24$','$N=32$','$N=40$',...
    'Richardson','2nd order $1/\epsilon$','interpreter','latex','numColumns',1,'Location','Northeast')
box on
% axes('Position',[0.2 0.7 0.2 0.2]);
% hold on
% box on
% plot(Ns(1:end-1),uerrorsSpec,'-o')
% hold on
% plot(Ns(1:end-1),uerrorsNoBd,'--s')
% xlabel('$N$','interpreter','latex')
% ylabel('$U$ error','interpreter','latex')
% legend('All','Interior')
set(gca,'YScale','Log')
return
hold on;
Momenterrors={};
nzation=max(abs(Moments{end}(1,:)));
for iU=1:length(Ns)
    Momenterrors{iU} = (Moments{iU}-Moments{end}(:,1:Ns(iU)))/nzation;
    Momenterrors2{iU} = (Moments{iU}-RectMoments(:,1:Ns(iU)))/nzation;
    Momenterrors3{iU} = (Moments{iU}-MomentsExtrap(:,1:Ns(iU)))/nzation;
end

figure;
hold on
set(gca,'ColorOrderIndex',1)
for iU=1:length(Ns)
plot(1:2:Ns(iU),(abs(Momenterrors3{iU}(1,2:2:end))),'-o')
end
MomentErrorsRect = (MomentsEps-MomentsExtrap)/nzation;
set(gca,'ColorOrderIndex',7)
plot(1:2:Ns(iU),(abs(MomentErrorsRect(1,2:2:end))),':d')
ylim([1e-4 10])
xlabel('$k$','interpreter','latex')
ylabel('Relative moment error','interpreter','latex')
legend('$N=8$','$N=16$','$N=24$','$N=32$','$N=40$','2nd order $1/\epsilon$','interpreter','latex')
set(gca,'YScale','Log')
box on
% axes('Position',[0.2 0.7 0.2 0.2]);
% hold on
% box on
% for index=1:length(Ns)
%     plot(spl,us{index}(:,2),linestys(index))
%     xlim([0 0.2])
% end
% axes('Position',[0.7 0.2 0.2 0.2]);
% hold on
% box on
% for index=1:length(Ns)
%     plot(spl,lams{index}(:,1),linestys(index))
%     xlim([0 0.2])
% end

function theta_s = getTheta(s)
    theta_s = sin(2*pi*s);
end