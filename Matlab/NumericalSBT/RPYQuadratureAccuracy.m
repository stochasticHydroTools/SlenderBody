deltaLocal=0;
clamp0=0;
clampL=0;
Eb = 1;
twmod = 1;
dt=0;
makeMovie=0;
nFib=1;
tf=0;
for eps=[1e-2 1e-3]
if (eps > 5e-3)
    NForSmalls = [6];
else
    NForSmalls = [4];
end
for NForSmall = NForSmalls
chebers=[];
Ns = 8:8:40;
for N = Ns
L=2;            % fiber length
mu=1/(8*pi);    % fluid viscosity
a=eps*L;
% Nodes for solution, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
chebForInts=1;
AllbS = precomputeStokesletInts(s,L,a,N,chebForInts);
AllbD = precomputeDoubletInts(s,L,a,N,chebForInts);

% Fiber initialization
q=1; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
X = pinv(D)*X_s;
X=X-barymat(0,s,b)*X;
fibpts = pinv(D)*X_s;
fibpts=fibpts-barymat(0,s,b)*fibpts;
theta0 = cos(4*pi*s/L);
theta0 = theta0-barymat(L/2,s,b)*theta0;
theta_s = D*theta0;
strongthetaBC=0;
InitFiberVars;

% Compute force and torque (discrete)
XBC = UpsampleXBCMat*Xt + BCShift;
fE= reshape(FE*XBC,3,N)';
% Calculate twisting force according to BC
theta_s_sp2 = UpsampleThetaBCMat \ [theta_s; ThetaBC0; 0];
Theta_ss = D_sp2*theta_s_sp2;
XBC3 = reshape(XBC,3,N+4)';
fTw3 = twmod*R4ToN*((R2To4*Theta_ss).*cross(D_sp4*XBC3,D_sp4^2*XBC3)+...
    (R2To4*theta_s_sp2).*cross(D_sp4*XBC3,D_sp4^3*XBC3));
f = fE+fTw3;
nsc= twmod*R2ToN*Theta_ss;
Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
Xsss = stackMatrix(R4ToN*D_sp4^3)*XBC;
Xss = reshape(Xss,3,N)';
Xsss = reshape(Xsss,3,N)';

% The goal is to compute the integral of the Stokeslet on |s-s'| > 2a using
% two different techniques: 1) Upsampling, and 2) Tornberg, and see what is
% more accurate
% Generate refined answer
Nref = 200;
%U_ref=1/(8*pi*mu)*upsampleRPY(X,s,X,f,s,b,Nref,L,a);
%U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,nsc.*X_s,s,b,Nref,L,a);
U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,f,s,b,Nref,L,a);
%U_ref=1/(8*pi*mu)*upsampleRPYRotRot(X,s,X,nsc.*X_s,s,b,Nref,L,a);
U_ref = sum(U_ref.*X_s,2);

% Approach 2: singularity subtraction & Tornberg
Mtt = ExactRPYSpectralMobility(N,X,X_s,Xss,Xsss,a,L,mu,s,b,D,AllbS,AllbD,NForSmall);
%U2 = reshape(Mtt*reshape(f',3*N,1),3,N)';
%U2 = upsampleRPYTransRotSmall(X,X_s,nsc,s,b,NForSmall,L,a,mu);
%U2 = U2 + UFromNFPIntegral(X,X_s,Xss,Xsss,s,N,L,nsc,D*nsc,AllbS,mu);
%U2 = U2 + reshape(getMlocRotlet(N,reshape(X_s',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,0)'*nsc,3,N)';
U2 = upsampleRPYRotTransSmall(X,X_s,f,s,b,NForSmall,L,a,mu);
U2 = U2 + OmFromFFPIntegral(X,X_s,Xss,Xsss,s,N,L,f,D*f,AllbS,mu); 
U2 = U2 + getMlocRotlet(N,reshape(X_s',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,0)*reshape(f',3*N,1);
%[~, ~, ~,Mrr,~] = getGrandMloc(N,zeros(3*N,1),zeros(3*N),a,L,mu,s,0);
%U2 = Mrr*nsc;

% Calculate error on upsampled grid
[sup,wup] = chebpts(1000,[0 L]);
Rup = barymat(sup,s,b);
Urefup = Rup*U_ref;
nzation = sqrt(wup*sum(Urefup.*Urefup,2));
er2 = Rup*(U2-U_ref);
ner2 = sqrt(wup*sum(er2.*er2,2))/nzation;

chebers = [chebers; N ner2];
end
if (eps > 5e-3 && NForSmall < 7)
    semilogy(Ns,chebers(:,2),'-s','MarkerSize',8)
    hold on
elseif (eps < 5e-3 && NForSmall < 5)
    semilogy(Ns,chebers(:,2),'--s','MarkerSize',8)
elseif (eps > 5e-3 && NForSmall > 7)
    semilogy(Ns,chebers(:,2),'--d')
else
    semilogy(Ns,chebers(:,2),':^')
end
end
end
xlabel('$N$','interpreter','latex')
%legend('$\hat{\epsilon}=10^{-2}$, $N_2=6$','$\hat{\epsilon}=10^{-2}$, $N_2=8$','$\hat{\epsilon}=10^{-3}$, $N_2=4$',...
%    ' $\hat{\epsilon}=10^{-3}$, $N_2=6$','interpreter','latex')
%ylabel('$L^2$ error in $U$','interpreter','latex')
ylabel('$L^2$ error in $\Omega^\parallel_\textrm{f}(s)$','interpreter','latex')
%ylabel('$L^2$ error in $U_\textrm{n}(s)$','interpreter','latex')
%ylabel('$L^2$ error in $\Omega^\parallel_\textrm{n}(s)$','interpreter','latex')
%legend('$\epsilon=10^{-2}$','$\epsilon=10^{-3}$','interpreter','latex')
xlim([8 40])


