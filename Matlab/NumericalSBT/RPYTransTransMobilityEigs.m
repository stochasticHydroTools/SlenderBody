deltaLocal=0;
clamp0=0;
clampL=0;
Eb = 1;
twmod = 1;
dt=0;
makeMovie=0;
nFib=1;
tf=0;
runN2plot=0;
for eps=1e-2
chebers=[];
if (runN2plot)
    Ns = 45;
else
    Ns= [15 30 45 60];
end
for iN=1:length(Ns)
N = Ns(iN);
if (N == 45 && runN2plot)
     N2s = [6 4 8];
else
    N2s = 6;
end
for iN2 = 1:length(N2s)
NForSmall=N2s(iN2);
L=2;            % fiber length
mu=1;%/(8*pi);    % fluid viscosity
% eps=1e-2;
a=eps*L;
% Nodes for solution, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');

% Fiber initialization
q=7; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
X = pinv(D)*X_s;
X=X-barymat(0,s,b)*X;
fibpts = X;
theta0 = cos(4*pi*s/L);
theta0 = theta0-barymat(L/2,s,b)*theta0;
theta_s = D*theta0;
InitFiberVars;
% Upsampled mobility
Mtt = zeros(3*N);
for iC=1:3*N
    fIn = zeros(3*N,1);
    fIn(iC)=1;
    fInput = reshape(fIn,3,N)';
    % Subtract singular part for each s
    Utt = 1/(8*pi*mu)*upsampleRPY(X,s,X,fInput,s,b,200,L,a);
    Mtt(:,iC) = reshape(Utt',3*N,1);
end

% Approach 2: singularity subtraction & Tornberg
XBC = UpsampleXBCMat*Xt + BCShift;
fE= reshape(FE*XBC,3,N)';
% Calculate twisting force according to BC
Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
Xsss = stackMatrix(R4ToN*D_sp4^3)*XBC;
Xss = reshape(Xss,3,N)';
Xsss = reshape(Xsss,3,N)';
Mrpy = ExactRPYSpectralMobility(N,X,X_s,Xss,Xsss,a,L,mu,s,b,D,AllbS,AllbD,NForSmall);

% SBT mobility
Msbt = getGrandMloc(N,reshape(X_s',3*N,1),Xss,a,L,mu,s,0);
Msbt = Msbt+StokesletFinitePartMatrix(X,X_s,Xss,D,s,L,N,mu,Allb_trueFP);

% Calculate error on upsampled grid
nzation = max(real(eig(Mrpy)));
if (runN2plot)
    nzation=1;
    startplotind=2*iN2-1;
    set(gca,'ColorOrderIndex',iN2)
    h(startplotind)=semilogy(3*N-1:-1:0,sort(real(eig(Mrpy)))/nzation,'-');
    hold on
    set(gca,'ColorOrderIndex',iN2)
    h(startplotind+1) =semilogy(3*N-1:-1:0,-sort(real(eig(Mrpy)))/nzation,'-.o');
    xlabel('$k$','interpreter','latex')
    ylabel('$\nu_k$','interpreter','latex')
    if (iN==length(Ns) && iN2==length(N2s))
        h(2*length(N2s)+1) =semilogy(3*N-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
    end
else
startplotind=5*iN-4;
if (N > 45)
    nzation=1;
    h(end+1) =semilogy(3*N-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
else
    nzation=1;
    h(startplotind) =semilogy(3*N-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
    hold on
    set(gca,'ColorOrderIndex',iN)
    h(startplotind+1) = semilogy(3*N-1:-1:0,sort(real(eig(Mrpy)))/nzation,'-');
    set(gca,'ColorOrderIndex',iN)
    h(startplotind+2) =semilogy(3*N-1:-1:0,-sort(real(eig(Mrpy)))/nzation,'-.o');
    set(gca,'ColorOrderIndex',iN)
    h(startplotind+3) =semilogy(3*N-1:-1:0,sort(real(eig(Msbt)))/nzation,'--');
    set(gca,'ColorOrderIndex',iN)
    h(startplotind+4) =semilogy(3*N-1:-1:0,-sort(real(eig(Msbt)))/nzation,':s');
    xlabel('$k$','interpreter','latex')
    ylabel('$\nu_k$','interpreter','latex')
end
end
end
end
if (runN2plot)
    legend(h(1:2:end),'$N_2=6$','$N_2=4$','$N_2=8$','Oversampled RPY','interpreter','latex')
else
legend(h([2 4 7 9 12 14 16]),'RPY, $N=15$','SBT, $N=15$','RPY, $N=30$','SBT, $N=30$','RPY, $N=45$','SBT, $N=45$',...
    'Over., $N=60$','interpreter','latex')
end
uistack(h(end),'bottom')
end
