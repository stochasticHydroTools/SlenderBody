% Compute refined solution via Richardson
eps = 1e-3;
delta = 0;
N = 1/eps;
KirchoffFiberRectangularKeav_Twist;
lambdaEps = lambda;
MomentsEps = RectMoments;
u1OverEps = RectuEuler;
Om1OverEps = OmegaPar_Euler;
N = 1/eps;
KirchoffFiberRectangularKeav_Twist;
u1=RectuEuler;
Moments1=RectMoments;
lambda1 = lambda;
sRect=s_u;
Omega1 = OmegaPar_Euler;
N = 2/eps;
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
index=1;
NForSmalls = 4;
if (eps > 2e-3)
    NForSmalls = [8];
end
Ns = 8:8:40;
pinvtol = 1e-10;
for N=Ns
for NForSmall=NForSmalls
for ptol = pinvtol
% delta = 0.1;
L = 2;
mu = 1;
local = 1;
Eb = 1;
twmod = 1;
a = eps*L;
[s,w,b] = chebpts(N,[0 L],1);
D = diffmat(N,[0 L],'chebkind1');
Lmat = cos(acos(2*s/L-1).*(0:N-1));
q=1; 
Xs = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
X = pinv(D)*Xs;
X=X-barymat(0,s,b)*X;
deltaLocal=delta;
clamp0 = 0;
clampL = 0;
dt = 0; tf = 1;
makeMovie = 0;
nFib=1;
fibpts = X; X_s=Xs;
theta_s = getTheta(s);
theta0 = pinv(D)*theta_s;
theta0 = theta0-barymat(L/2,s,b)*theta0;
SBTMain;

% Compute force
XBC = UpsampleXBCMat2*Xt + BCShift;
Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
Xsss = stackMatrix(R4ToN*D_sp4^3)*XBC;
Xss = reshape(Xss,3,N)';
Xsss = reshape(Xsss,3,N)';
fE = FE*XBC;
% Calculate twisting force according to BC
theta_s_sp2 = UpsampleThetaBCMat \ [theta_s; ThetaBC0; 0];
Theta_ss = D_sp2*theta_s_sp2;
XBC3 = reshape(XBC,3,N+4)';
fTw3 = twmod*R4ToN*((R2To4*Theta_ss).*cross(D_sp4*XBC3,D_sp4^2*XBC3)+...
    (R2To4*theta_s_sp2).*cross(D_sp4*XBC3,D_sp4^3*XBC3));
fTw = reshape(fTw3',3*N,1);
npar = twmod*R2ToN*Theta_ss;

% Compute mobilities 
if (local==0) % all integrals by upsampling
    Mtt = zeros(3*N);
    Mrt = zeros(3*N);
    Mtr = zeros(3*N);
    Mrr = zeros(3*N);
    for iC=1:3*N
        fIn = zeros(3*N,1);
        fIn(iC)=1;
        fInput = reshape(fIn,3,N)';
        % Subtract singular part for each s
        Nups=200;
        Utt = 1/(8*pi*mu)*upsampleRPY(X,s,X,fInput,s,b,Nups,L,a);
        Utr = 1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,fInput,s,b,Nups,L,a);
        Urt = 1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,fInput,s,b,Nups,L,a);
        Urr = 1/(8*pi*mu)*upsampleRPYRotRot(X,s,X,fInput,s,b,Nups,L,a);
        Mtt(:,iC) = reshape(Utt',3*N,1);
        Mtr(:,iC) = reshape(Utr',3*N,1);
        Mrt(:,iC) = reshape(Urt',3*N,1);
        Mrr(:,iC) = reshape(Urr',3*N,1);
    end
else % Our asymptotic mobility
    chebForInts=1;
    AllbS = precomputeStokesletInts(s,L,a,N,chebForInts);
    Allb_trueFP = precomputeStokesletInts(s,L,0,N,chebForInts);
    AllbD = precomputeDoubletInts(s,L,a,N,chebForInts);
    [~, MtrLoc, MrtLoc, Mrr,~] = getGrandMloc(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,delta);
    % Translational mobility
    %NForSmall = 8;
    Mtt = ExactRPYSpectralMobility(N,X,Xs,Xss,Xsss,a,L,mu,s,b,D,AllbS,AllbD,NForSmall);
%     Mtt = getGrandMloc(N,Xs_stack,Xss_stack,a,L,mu,s,0);
%     Mtt= Mtt+StokesletFinitePartMatrix(X,Xs,D,s,L,N,mu,Allb_trueFP);
end

% Solve problem
if (local==1)
    UFromN = upsampleRPYTransRotSmall(X,Xs,npar,s,b,NForSmall,L,a,mu);
    UFromN = UFromN + UFromNFPIntegral(X,Xs,Xss,Xsss,s,N,L,npar,D*npar,AllbS,mu);
    UFromN = UFromN + reshape(getMlocRotlet(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,0)'*npar,3,N)';
    UFromN = reshape(UFromN',3*N,1);
else
    UFromN = Mtr*reshape((npar.*Xs)',3*N,1);
end
Mttinv = Mtt^(-1);
[K,Kt]=getKMats3D(Xst,Lmat,w,N,I,wIt,'U',0);
RHS = Kt*(fE+fTw+Mttinv*UFromN);
alphaU = lsqminnorm(Kt*Mttinv*K,RHS,ptol);
uEuler = K*alphaU;
lambda = Mttinv*(uEuler-UFromN) - fE - fTw;
f = fE+lambda+fTw;
% Compute parallel vel in post-processing step
if (local==0)
    OmegaPar_Euler = sum(reshape(Mrt*f+Mrr*reshape((npar.*Xs)',3*N,1),3,N)'.*Xs,2);
else
    f = reshape(f,3,N)';
    OmegaPar_Euler = upsampleRPYRotTransSmall(X,Xs,f,s,b,NForSmall,L,a,mu);
    OmegaPar_Euler = OmegaPar_Euler + OmFromFFPIntegral(X,Xs,Xss,Xsss,s,N,L,f,D*f,AllbS,mu); 
    OmegaPar_Euler = OmegaPar_Euler + getMlocRotlet(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),...
        a,L,mu,s,0)*reshape(f',3*N,1);
    [~, ~, ~,Mrr,~] = getGrandMloc(N,zeros(3*N,1),zeros(3*N),a,L,mu,s,deltaLocal);
    OmegaPar_Euler = OmegaPar_Euler+Mrr*npar;
end

[spl,wpl,bpl]=chebpts(1000,[0 L]);
Rpl = barymat(spl,s,b);
us{index}=Rpl*reshape(uEuler,3,N)';
lams{index}=Rpl*reshape(lambda,3,N)';
%plot(s,lambda(1:3:end))
% hold on
stress{index}=zeros(3);
Moments{index}=zeros(3,N);
[s2,w2,b2]=chebpts(2*N,[0 L]);
R1To2 = barymat(s2,s,b);
R1To2s = stackMatrix(R1To2);
R2To1 = barymat(s,s2,b2);
R2To1s = stackMatrix(R2To1);
regLoc_Slt2 = getMlocStokeslet(2*N,R1To2*Xs,a,L,mu,s2,0.1);
Mtt2 = regLoc_Slt2+4*eye(3*2*N)/(8*pi*mu);
Mf1 = Rpl*R2To1*reshape(Mtt2*(R1To2s*fE),3,2*N)';
Mf2 = Rpl*reshape(Mtt*fE,3,N)';
Mf{index}=Rpl*reshape(Mtt*ones(3*N,1),3,N)';
Xup = Rpl*X;
for iPt=1:1000
    stress{index}=stress{index}+lams{index}(iPt,:)'*Xup(iPt,:)*wpl(iPt);
end
for iMoment=0:N-1
    Tk = cos(iMoment*acos(2*spl/L-1));
    Moments{index}(:,iMoment+1)=wpl*(lams{index}.*Tk);
end
    
Oms{index}=Rpl*OmegaPar_Euler;
index=index+1;
end
end
end
linestys=["-","-","-","-","-","-"];

% figure;
% hold on
% box on
% for index=1:2
% % subplot(1,3,1)
% % hold on
% % box on   
% % plot(spl/L,us{index}(:,2),linestys(index))
% % if (index==length(us))
% %     plot(s_u/L,uExtrap(:,2),'--')
% % end
% % xlabel('$s/L$','interpreter','latex')
% % ylabel('$U^{(y)}(s)$','interpreter','latex')
% plot(spl/L,lams{index}(:,1),linestys(index))
% if (index==2)
%     plot(s_u/L,lambdaExtrap(:,1),'--')
% end
% %ylim([-10 10])
% xlabel('$s/L$','interpreter','latex')
% ylabel('$\lambda^{(x)}(s)$','interpreter','latex')
% % subplot(1,3,3)
% % hold on
% % box on
% % plot(spl/L,Oms{index},linestyleges(index))
% % if (index==length(us))
% %     plot(s_u/L,OmegaParExtrap,'--')
% % end
% % xlabel('$s/L$','interpreter','latex')
% % ylabel('$\Omega^\parallel(s)$','interpreter','latex')
% end
% 
% subplot(1,2,1)
% uerrors=zeros(1,length(Ns)-1);
% ds = L/length(uExtrap);
% nzation=sqrt(wpl*(sum(us{end}.*us{end},2)));
% RToUni = barymat(s_u,spl,bpl);
% for iU=1:length(uerrors)
%     diffRect = RToUni*us{iU}-uExtrap;
%     uerrorsRect(iU)=sqrt(ds*sum(sum(diffRect.*diffRect,2)))/nzation;
%     diffSpec = us{iU}-us{iU+1};
%     uerrorsSpec(iU)=sqrt(wpl*sum(diffSpec.*diffSpec,2))/nzation;
%     boundary = spl/L < 0.05 | spl/L > 0.95;
%     uerrorsNoBd(iU)=sqrt(wpl(~boundary)*sum(diffSpec(~boundary).*diffSpec(~boundary),2))/nzation;
% end
% try
%     diff=uExtrap-u1OverEps;
% catch
%     diff=1/2*(uExtrap(1:2:end,:)+uExtrap(2:2:end,:))-u1OverEps;
% end
% normdiff=sum(diff.*diff,2);
% EpsEr=sqrt(L/length(u1OverEps)*sum(normdiff))/nzation;
% semilogy(Ns(1:end-1),uerrorsSpec,'-o')
% set(gca,'ColorOrderIndex',1)
% hold on
% %semilogy(Ns(1:end-1),uerrorsRect,'--o')
% %hold on
% %semilogy(Ns(1:end-1),uerrorsNoBd,'-.s')
% plot(xlim,EpsEr*[1 1],':')
% xlabel('$N$','interpreter','latex')
% ylabel('Relative $U$ error','interpreter','latex')
% legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
% 
% subplot(1,2,2)
% Omerrors=zeros(1,length(Ns)-1);
% nzation=sqrt(wpl*(sum(Oms{end}.*Oms{end},2)));
% for iU=1:length(Omerrors)
%     diffRect = RToUni*Oms{iU}-OmegaParExtrap;
%     OmerrorsRect(iU)=sqrt(ds*sum(sum(diffRect.*diffRect,2)))/nzation;
%     OmdiffSpec = Oms{iU}-Oms{iU+1};
%     OmerrorsSpec(iU)=sqrt(wpl*sum(OmdiffSpec.*OmdiffSpec,2))/nzation;
%     boundary = spl/L < 0.05 | spl/L > 0.95;
%     OmerrorsNoBd(iU)=sqrt(wpl(~boundary)*sum(OmdiffSpec(~boundary).*OmdiffSpec(~boundary),2))/nzation;
% end
% try
%     diff=OmegaParExtrap-Om1OverEps;
% catch
%     diff=1/2*(OmegaParExtrap(1:2:end,:)+OmegaParExtrap(2:2:end,:))-Om1OverEps;
% end
% normdiff=sum(diff.*diff,2);
% EpsEr=sqrt(L/length(Om1OverEps)*sum(normdiff))/nzation;
% semilogy(Ns(1:end-1),OmerrorsSpec,'-o')
% set(gca,'ColorOrderIndex',1)
% hold on
% %semilogy(Ns(1:end-1),OmerrorsRect,'--o')
% %hold on
% %semilogy(Ns(1:end-1),OmerrorsNoBd,'-.s')
% plot(xlim,EpsEr*[1 1],':')
% xlabel('$N$','interpreter','latex')
% ylabel('Relative $U$ error','interpreter','latex')
% %legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
% % subplot(1,2,2)
% % lamerrors=zeros(1,length(Ns));
% % nzation=sqrt(sum(ds*sum(lambdaExtrap.*lambdaExtrap,2)));
% % RToUni = barymat(s_u,spl,bpl);
% % for iU=1:length(lamerrors)
% %     diffRect = RToUni*lams{iU}-lambdaExtrap;
% %     lamerrorsRect(iU)=sqrt(ds*sum(sum(diffRect.*diffRect,2)))/nzation;
% %     boundary = s_u/L < 0.05 | s_u/L > 0.95;
% %     lamerrorsNoBd(iU)=sqrt(ds*sum(sum(diffRect(~boundary).*diffRect(~boundary),2)))/nzation;
% % end
% % try
% %     diff=lambdaExtrap-lambdaEps;
% % catch
% %     diff=1/2*(lambdaExtrap(1:2:end,:)+lambdaExtrap(2:2:end,:))-lambdaEps;
% % end
% % normdiff=sum(diff.*diff,2);
% % EpsEr=sqrt(L/length(lambdaEps)*sum(normdiff))/nzation;
% % semilogy(Ns,lamerrorsRect,'-o')
% % hold on
% % semilogy(Ns,lamerrorsNoBd,'-.s')
% % plot(xlim,EpsEr*[1 1],':k')
% % xlabel('$N$','interpreter','latex')
% % ylabel('Relative $\lambda$ error','interpreter','latex')
% % legend('Error','$0.05 \leq s/L \leq 0.95$','2nd order $1/\epsilon$','interpreter','latex')
% return;
figure;
box on
hold on;
Momenterrors={};
nzation=max(abs(Moments{end}(:)));
for iU=1:length(Ns)
    Momenterrors{iU} = (Moments{iU}-Moments{end}(:,1:Ns(iU)))/nzation;
    Momenterrors2{iU} = (Moments{iU}-RectMoments(:,1:Ns(iU)))/nzation;
    Momenterrors3{iU} = (Moments{iU}-MomentsExtrap(:,1:Ns(iU)))/nzation;
end
hold on
for iU=1:length(Ns)
plot(1:2:Ns(iU),max(abs(Moments{iU}(:,2:2:end))),'-o')
end
plot(1:2:40,max(abs(MomentsExtrap(:,2:2:end))),'-.s')
plot(1:2:40,max(abs(MomentsEps(:,2:2:end))),':d')
set(gca,'YScale','Log')
xlabel('$k$','interpreter','latex')
ylabel('Moment','interpreter','latex')
legend('$N=8$','$N=16$','$N=24$','$N=32$','$N=40$',...
    'Richardson','2nd order $1/\epsilon$','interpreter','latex','numColumns',2)

figure;
hold on
set(gca,'ColorOrderIndex',1)
for iU=1:length(Ns)
plot(1:2:Ns(iU),max(abs(Momenterrors3{iU}(:,2:2:end))),'-o')
end
MomentErrorsRect = (MomentsEps-MomentsExtrap)/nzation;
set(gca,'ColorOrderIndex',7)
plot(1:2:Ns(iU),max(abs(MomentErrorsRect(:,2:2:end))),':d')
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