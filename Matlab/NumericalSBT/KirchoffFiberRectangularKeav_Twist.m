localdrag = 0;
delta = 0;
L = 2;
% eps = 0.0025;
kappa = 1;
twistmod = 1;
linesty='-';
% if (twistmod > 0)
%      N = 4/eps;
% else
%      N =1/eps;
% end
mu = 1;
a = eps*L;
ds = L/N;
s_u = ((1/2:1:N)*ds)';
s_uep = (0:N)*ds;
DCToC = DerivMat(N,ds);
[X_u,Xs_u,Xss_u,Xsss_u,Xs4_u] = FlorenFiber(s_u,L,DCToC);
XsCross = zeros(3*N);
XsXsAll = zeros(3*N);
D = zeros(N,N+1); % maps ends -> centers
Avg = zeros(N,N+1); % ends -> centers (adjoint is centers -> ends)
IntMat = zeros(N+1,N); % centers -> ends
for iP=1:N % Looping over segments / centers
    XsCross(3*iP-2:3*iP,3*iP-2:3*iP) = CPMatrix(Xs_u(iP,:));
    D(iP,iP) = -1/ds;
    D(iP,iP+1) = 1/ds;
    Avg(iP,iP) = 1/2;
    Avg(iP,iP+1)=1/2;
    IntMat(iP,1:iP-1)=ds;
end
IntMat(N+1,:)=ds;
Ds = zeros(3*N,3*N+3);
Avgs = zeros(3*N,3*N+3);
IntMats = zeros(3*N+3,3*N);
for iD=1:3
    Ds(iD:3:end,iD:3:end)=D;
    Avgs(iD:3:end,iD:3:end)=Avg;
    IntMats(iD:3:end,iD:3:end)=IntMat;
end
Ds_sm = Ds(4:end,4:end);
Avg_sm = Avgs(4:end,4:end);

% Mobility AT SEGMENT CENTERS (maps centers -> centers)
if (localdrag)    
    [sReg,regwt] = RegularizeS(s_u,delta,L);
    [~, Mtr, Mrr,sNew] = getGrandMloc(N,reshape(Xs_u',3*N,1),reshape(Xss_u',3*N,1),...
        stackMatrix(DCToC),a,L,mu,s_u,delta);
    % Compute spectral FP matrices column by column
    SletFP = 1/(8*pi*mu)*NLMatrixSkip2a(reshape(X_u',3*N,1),...
             reshape(Xs_u',3*N,1),s_u,ds*ones(1,N),N,a,L,1);
    SletFPn = 1/(8*pi*mu)*NLMatrixSkip2a(reshape(X_u',3*N,1),...
         reshape(Xs_u',3*N,1),s_u,ds*ones(1,N),N,a,L,0);
    DbletFP = 1/(8*pi*mu)*DoubletNLMatrixSkip2a(reshape(X_u',3*N,1),...
             reshape(Xs_u',3*N,1),s_u,ds*ones(1,N),N,a,L);
    Mtr_FP = 1/(8*pi*mu)*NLMatrixRotTrans(reshape(X_u',3*N,1),...
         reshape(Xs_u',3*N,1),reshape(Xss_u',3*N,1),stackMatrix(DCToC), s_u,ds*ones(1,N),N,a);
    Mtr = Mtr+Mtr_FP;
    % Translational mobility
    % Integral of Stokeslet and doublet, leave small part asymptotic
    Loc_Slt = getMlocStokeslet(N,Xs_u,a,L,mu,s_u,0);
    regLoc_Slt = getMlocStokeslet(N,Xs_u,a,L,mu,s_u,delta);
    Loc_Dblt = getMlocDoublet(N,Xs_u,Xss_u,Xsss_u,stackMatrix(DCToC),a,L,mu,s_u,0,0);
    SmallExact = zeros(3*N);
    for iPt=1:N
        for jPt=1:N
            if (abs(s_u(iPt)-s_u(jPt)) <= 2*a)
                MRPY = calcRPYKernel(X_u(iPt,:),X_u(jPt,:),a);
                SmallExact(3*iPt-2:3*iPt,3*jPt-2:3*jPt)=1/(8*pi*mu)*MRPY*ds;
            end
        end
    end
    stackWt = reshape(repmat(regwt,1,3)',3*N,1);
    Dlet = Loc_Dblt+DbletFP;
    Block4I = eye(3*N)*4/(8*pi*mu);
    Mtt = stackWt.^2.*(Loc_Slt+SletFP+2*a^2/3*Dlet + SmallExact)+(1-stackWt.^2).*(Block4I+...
        regLoc_Slt);
    Mtr_par = Mtr;
    Mrt = Mtr;
else
    [Mtt, Mtr, Mrr] = getGrandMBlobs(N,X_u,a,mu);
    Mtt = Mtt*ds;
    Mtr = Mtr*ds;
    Mrr = Mrr*ds;
    Mtr_par = Mtr;
    Mrt = Mtr;
end
[sReg,regwt] = RegularizeS(s_u,delta,L);
AvgMat=zeros(N);
if (delta > 0)
    for iPt=1:N
    t = s_u(iPt);
    w = regwt(iPt);
    id = zeros(1,N);
    id(iPt)=1;
    if (t < L/2)
    closept=delta*L;
    else
    closept = L-delta*L;
    end
    gauss = exp(-(abs(s_u-closept)/(delta*L)).^2)';
    gauss = ds*gauss/sum(ds*gauss);
    AvgMat(iPt,:)=w^2*id+(1-w^2)*gauss;
    end
else 
    AvgMat = eye(N);
end
Mtt = stackMatrix(AvgMat)*Mtt;
% Now do the Euler method
[theta_s,theta_ss] = getTheta(s_u);
%theta_ss = DCToC*theta_s;
npar = twistmod*reshape((theta_ss.*Xs_u)',3*N,1);
K = KMatBlobs(ds,Xs_u);
Kt = K';
fE_3 = -kappa*Xs4_u+twistmod*(theta_ss.*cross(Xs_u,Xss_u)+theta_s.*cross(Xs_u,Xsss_u));
fE = reshape(fE_3',3*N,1);
RHS = Kt*(fE+Mtt^(-1)*Mtr_par*npar);
alphaU = (Kt*Mtt^(-1)*K) \RHS;
uEuler = K*alphaU;
lambda = Mtt \(uEuler-Mtr_par*npar) - fE;
mismatch = Ds_sm*(Mtt*(fE+lambda)+Mtr_par*npar) + Avg_sm*XsCross*(Mrt*(fE+lambda)+Mrr*npar);
OmegaPar_Euler = sum(reshape(Mrt*(fE+lambda)+Mrr*npar,3,N)'.*Xs_u,2);

% Compute moments of lambda against Cheb polynomials up to 40
RectMoments = zeros(3,40);
lambda = reshape(lambda,3,N)';
uEuler = reshape(uEuler,3,N)';
for iMoment=0:39
    Tk = cos(iMoment*acos(2*s_u/L-1));
    RectMoments(:,iMoment+1)=ds*sum(lambda.*Tk);
end
RectuEuler = uEuler;
% plot(s_u,uEuler(:,2))
return
% 
% % for iD=0
% subplot(1,3,2)
% hold on
% box on
% plot(s_u,lambda(1:3:end),linesty)
% ylabel('$\lambda_E^{(x)}$ (Euler)','interpreter','latex')
% subplot(1,3,1)
% hold on
% box on
% plot(s_u,uEuler(2:3:end),linesty)
% ylabel('$U_E^{(y)}$ (Euler)','interpreter','latex')
% subplot(1,3,3)
% hold on
% box on
% plot(s_u,OmegaPar_Euler,linesty)
% ylabel('$\Omega_E^\parallel$','interpreter','latex')
% end
% 
% 
Ns_3 = kappa*cross(Xs_u,Xsss_u)+twistmod*(theta_ss.*Xs_u+theta_s.*Xss_u); % torque defined at CENTERS
Ns = reshape(Ns_3',3*N,1);
USegmentsMat = Mtt*Ds+Mtr*XsCross*Avgs;  
OmSegmentsMat = XsCross*(Mrt*Ds+Mrr*XsCross*Avgs);
FMatrix = Ds_sm*USegmentsMat + Avg_sm*OmSegmentsMat;
FMatrix = [eye(3) zeros(3,3*N); FMatrix; zeros(3,3*N) eye(3)];
% RHS = [zeros(3,1); -Ds_sm*Mtr*Ns-Avg_sm*XsCross*Mrr*Ns; zeros(3,1)];
% F = FMatrix \ RHS;
% Fs = Ds*F;
% uKirch = USegmentsMat*F+Mtr*Ns;
% OmegaKirch = OmSegmentsMat*F+XsCross*Mrr*Ns;
RHSper = [zeros(3,1); -mismatch; zeros(3,1)];
deltaF = FMatrix \ RHSper;
mismatch=reshape(mismatch,3,N-1)';
uEuler = reshape(uEuler,3,N)';

% Difference
%deltaFs = reshape(Fs-(fE+lambda),3,[])';        % defined at midpoints
deltaFs = reshape(Ds*deltaF,3,[])';           % defined at midpoints
%deltaU = reshape(uKirch-uEuler,3,N)';          % defined at midpoints
deltaU = reshape(USegmentsMat*deltaF,3,[])';  % defined at midpoints
F0 = reshape(IntMats*(fE+lambda),3,[])';
deltaOmega = reshape((Mrt*Ds+Mrr*XsCross*Avgs)*deltaF,3,N)';
Omega=reshape((Mrt*Ds)*reshape(F0',[],1),3,N)';
deltaOmCrossTau = cross(Xs_u,deltaOmega);
deltaOmDotTau = sum(Xs_u.*deltaOmega,2);
deltaF = reshape(deltaF,3,[])';
deltaFPar = sum((Avg*deltaF).*Xs_u,2);
%deltaOmCrossTau = cross(deltaOm,Xs_u); % defined at centers 1, ..., N

% Estimate and plot deltaF
DDeltaU = Ds_sm(1:3:end,1:3:end)*deltaU;
line1=[sqrt(sum(ds*sum(mismatch.*mismatch,2))) ...
sqrt(sum(ds*sum(deltaU.*deltaU,2))) ...
sqrt(sum(ds*sum(deltaF.*deltaF,2))) ...
sqrt(sum(ds*sum(DDeltaU.*DDeltaU,2))) ...
sqrt(sum(ds*sum(deltaOmCrossTau.*deltaOmCrossTau,2))) ...
sqrt(sum(ds*deltaOmDotTau.*deltaOmDotTau))];
normdistance = 0.1;
inds = s_u > normdistance & s_u < L-normdistance;
minds = s_uep > normdistance & s_uep < L-normdistance;
mintinds = s_uep(2:end-1) > normdistance & s_uep(2:end-1) < L-normdistance;

line2 = [sqrt(sum(ds*sum(mismatch(mintinds,:).*mismatch(mintinds,:),2))) ...
sqrt(sum(ds*sum(deltaU(inds,:).*deltaU(inds,:),2))) ...
sqrt(sum(ds*sum(deltaF(minds,:).*deltaF(minds,:),2))) ...
sqrt(sum(ds*sum(DDeltaU(mintinds,:).*DDeltaU(mintinds,:),2))) ... 
sqrt(sum(ds*sum(deltaOmCrossTau(inds,:).*deltaOmCrossTau(inds,:),2))) ...
sqrt(sum(ds*deltaOmDotTau(inds).*deltaOmDotTau(inds)))];
DUEuler = DCToC*uEuler;
normalizations = [normalizations;  sqrt(sum(ds*sum(uEuler(inds,:).*uEuler(inds,:),2))) ...
     sqrt(sum(ds*sum(DUEuler(inds,:).*DUEuler(inds,:),2))) ...
  sqrt(sum(ds*sum(F0(minds,:).*F0(minds,:),2))) ...
  sqrt(sum(ds*OmegaPar_Euler(inds,:).*OmegaPar_Euler(inds,:)))];
 %allLine1s=[allLine1s; line1];
 allLine2s=[allLine2s; line2];
 
% % sqrt(sum(ds*sum(MismatchLocal(inds,:).*MismatchLocal(inds,:),2)))
% % Matrix to convert coordinates from (x,y,z) -> local triad
CoordinateConv_LINKS =  ChangeCoordsMat(s_u,L);
CoordinateConv_BLOBS =  ChangeCoordsMat(s_uep',L);


subplot(2,3,1)
hold on
box on
DuEuler = Ds_sm(1:3:end,1:3:end)*uEuler;
uEulerCoords = CoordinateConv_BLOBS(4:end-3,4:end-3)*reshape(DuEuler',[],1);
uEulerCoords =reshape(uEulerCoords,3,[])';
plot(s_uep(2:end-1),uEulerCoords,linesty)
ylabel('$\partial_s U_E$','interpreter','latex')
subplot(2,3,2)
hold on
box on
mismatchCoords = CoordinateConv_BLOBS(4:end-3,4:end-3)*reshape(mismatch',[],1);
mismatchCoords = reshape(mismatchCoords,3,[])';
plot(s_uep(2:end-1),-mismatchCoords,linesty)
ylabel('$-m$','interpreter','latex')
subplot(2,3,3)
hold on
box on
deltaFCoords = CoordinateConv_BLOBS*reshape(deltaF',[],1);
deltaFCoords = reshape(deltaFCoords,3,[])';
plot(s_uep,deltaFCoords,linesty)
ylabel('$\Delta F$','interpreter','latex')
subplot(2,3,4)
hold on
box on
plot(s_u,deltaOmDotTau,linesty)
ylabel('$\Delta \Omega^\parallel$','interpreter','latex')
%xlabel('$s$','interpreter','latex')
subplot(2,3,5)
hold on
box on
deltaUs = Ds_sm*reshape(deltaU',[],1);
deltaUsCoords = CoordinateConv_BLOBS(4:end-3,4:end-3)*deltaUs;
deltaUsCoords = reshape(deltaUsCoords,3,[])';
plot(s_uep(2:end-1),deltaUsCoords,linesty)
xlim([0 2])
ylabel('$\partial_s \Delta U$','interpreter','latex')
xlabel('$s$','interpreter','latex')
subplot(2,3,6)
hold on
box on
deltaOmCoords = CoordinateConv_LINKS*reshape(deltaOmCrossTau',[],1);
deltaOmCoords=reshape(deltaOmCoords,3,[])';
plot(s_u,deltaOmCoords,linesty)
ylabel('$\tau \times \Delta \Omega $','interpreter','latex')
xlabel('$s$','interpreter','latex')

function [X,Xs, Xss, Xsss, Xssss] = FlorenFiber(s,L,D)
    q=1;
    syms t
    XsSym = [cos(q*t.^3 .* (t-L).^3) sin(q*t.^3.*(t - L).^3) 1]/sqrt(2);
    Xss = double(subs(diff(XsSym,t),s));
    Xsss = double(subs(diff(XsSym,t,2),s));
    Xssss = double(subs(diff(XsSym,t,3),s));
    Xs = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(length(s),1)]/sqrt(2);
%     Xss = D*Xs;
%     Xsss = D^2*Xs;
    %Xssss = D^3*Xs;
    X = zeros(length(s),3);
    ds = s(2)-s(1);
    for iPt=1:length(s)
        for jPt=1:iPt-1
            X(iPt,:)=X(iPt,:)+ds*Xs(jPt,:);
        end
        X(iPt,:)=X(iPt,:)+ds/2*Xs(iPt,:);
    end
end

    
function [theta_s,theta_ss]= getTheta(s)
    theta_s = sin(2*pi*s);
    theta_ss = 2*pi*cos(2*pi*s);
end

function CoordinateConv = ChangeCoordsMat(s,L)
    N = length(s);
    [~,Xs_u,~,~,~] = FlorenFiber(s,L,zeros(N));
    [theta,phi,~] = cart2sph(Xs_u(:,1),Xs_u(:,2),Xs_u(:,3));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1 = [-sin(theta) cos(theta) 0*theta];
    n2 = [-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
    for iPt=1:N
        mat = [Xs_u(iPt,:)' n1(iPt,:)' n2(iPt,:)'];
        CoordinateConv(3*iPt-2:3*iPt,3*iPt-2:3*iPt)=mat^(-1);
    end
end

function DCToC = DerivMat(N,ds)
    DCToC = zeros(N);
    for iR=2:N-1
        DCToC(iR,iR-1)=-1/(2*ds);
        DCToC(iR,iR+1)=1/(2*ds);
    end
    DCToC(1,:)=[-3/2 2 -1/2 zeros(1,N-3)]/ds;
    DCToC(end,:)=[zeros(1,N-3) 1/2 -2 3/2]/ds;
end


        