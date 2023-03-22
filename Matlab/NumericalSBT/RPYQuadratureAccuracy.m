for eps=[1e-2 1e-3]
if (eps > 5e-3)
    NForSmalls = [6 10];
else
    NForSmalls = [4 8];
end
for NForSmall = NForSmalls
chebers=[];
Ns = 8:8:56;
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
X = [cos(3*s) sin(3*s) cos(s)];
X_s = D*X;
f = [cos(2*s.^3) s.^2 sin(2*s)];
nsc = cos(4*s.^2);

% The goal is to compute the integral of the Stokeslet on |s-s'| > 2a using
% two different techniques: 1) Upsampling, and 2) Tornberg, and see what is
% more accurate
% Generate refined answer
Nref = 1600;
%U_ref=1/(8*pi*mu)*upsampleRPY(X,s,X,f,s,b,Nref,L,a);
%U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,nsc.*X_s,s,b,Nref,L,a);
%U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,f,s,b,Nref,L,a);
U_ref=1/(8*pi*mu)*upsampleRPYRotRot(X,s,X,nsc.*X_s,s,b,Nref,L,a);
U_ref = sum(U_ref.*X_s,2);

% Approach 2: singularity subtraction & Tornberg
asymp=1; delta=0;
%Mtt = TransTransMobilityMatrix(X,a,L,mu,s,b,D,AllbS,AllbD,NForSmall,asymp,delta);
%U2 = reshape(Mtt*reshape(f',3*N,1),3,N)';
%U2 = UFromN(X,nsc,D,AllbS,a,L,mu,s,b,1,NForSmall);
%U2 = OmegaFromF(X,f,D,AllbS,a,L,mu,s,b,1,NForSmall);
Mrr = RotRotMobilityMatrix(X,a,L,mu,s,b,D,AllbD,NForSmall,asymp,delta);
U2 = Mrr*nsc;

% Calculate error on upsampled grid
[sup,wup] = chebpts(1000,[0 L]);
Rup = barymat(sup,s,b);
Urefup = Rup*U_ref;
nzation = sqrt(wup*sum(Urefup.*Urefup,2));
er2 = Rup*(U2-U_ref);
ner2 = sqrt(wup*sum(er2.*er2,2))/nzation;

chebers = [chebers; N ner2];
end
semilogy(Ns,chebers(:,2),'-o')
hold on
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


