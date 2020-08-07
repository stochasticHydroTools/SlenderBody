% File to solve for alpha and lambda for a single fiber
% i.e., solve system
% [-M K; K^T 0]*[alpha;lambda]=[M*fE;0]
% Using a spectral discretization
% Important variables
N=16;
Lf=2;   
mu=1;
eps=1e-3;
Eb=1;

% Chebyshev initialization
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 Lf], 2);
[s0,w0,~] = chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.

% Fiber initialization (Floren's configuration)
X_s = [cos(s0.^3 .* (s0-Lf).^3) sin(s0.^3.*(s0 - Lf).^3) ones(N,1)]/sqrt(2);
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[~,xp] = ode45(@(s,X) cos(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
[~,yp]= ode45(@(s,X) sin(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
zp = s0/sqrt(2);
fibpts=[xp(2:end) yp(2:end) zp];

D = diffmat(N, 1, [0 Lf], 'chebkind1');
[Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf);
FE = -Eb*Rs*D4s;

% Take one step
Xt= reshape(fibpts',3*N,1);
Xtm1 = Xt;
Xst = reshape(X_s',3*N,1);
Xstm1=Xst;
Mloc = getMlocRPY(N,Xst,eps,Lf,mu,s0);
[K,Kt]=getKMats3D(Xst,chebyshevmat,w0,N);
fE=FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt)); % elastic force
fEcols = reshape(fE,3,N);
reler=10;
lambda = zeros(3*N,1); % 0 initial guess for lambda
iters=0;
while (reler > 1e-6)
    lamCols = reshape(lambda,3,N);
    [~,Oone] = calcSelf(N,s0,Lf,eps,fibpts,X_s,fEcols+lamCols,D); % finite part integral
    nLvel = 1/(8*pi*mu)*reshape(Oone',3*N,1);
    B=[K I];
    C=[Kt; wIt];
    RHS = C*fE+ C*Mloc^(-1)*nLvel;
    alphaU = lsqminnorm(C*Mloc^(-1)*B,RHS);
    alphas=alphaU(1:2*N-2);
    Urigid=alphaU(2*N-1:2*N+1);
    ut = K*alphas+I*Urigid;
    OmegaCrossTau = Ds*(K*alphas);
    l_m1 = lambda;
    lambda = Mloc \ (K*alphas+I*Urigid-nLvel)-fE;
    reler = norm(lambda-l_m1)/(max([1 norm(lambda)])); % lambda convergence check
    iters=iters+1;
end
