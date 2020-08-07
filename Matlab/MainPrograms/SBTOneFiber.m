% File to solve for alpha and lambda for a single fiber
% i.e., solve system
% [-M K; K^T 0]*[alpha;lambda]=[M*fE;0]
% Using a spectral discretization
% Important variables
N=20;       % number of Chebyshev pts
Nuni = 1000; % number of uniform points (blobs)
L=2;        % fiber length
mu=1;       % fluid viscosity
eps=5e-3;   % r/L aspect ratio
Eb=1;       % bending constant 
delta =0.2; % fraction of length over which to taper the fiber

% Chebyshev initialization
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 L], 2);
[s0,w0,~] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
[Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,L);
FE = -Eb*Rs*D4s;

% Fiber initialization (Floren's configuration)
X_s = [cos(s0.^3 .* (s0-L).^3) sin(s0.^3.*(s0 - L).^3) ones(N,1)]/sqrt(2);
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[~,xp] = ode45(@(s,X) cos(s^3 * (s-L)^3)/sqrt(2),[0; s0],0,opts);
[~,yp]= ode45(@(s,X) sin(s^3 * (s-L)^3)/sqrt(2),[0; s0],0,opts);
zp = s0/sqrt(2);
fibpts=[xp(2:end) yp(2:end) zp];

% Compute matrices K and M
Xt= reshape(fibpts',3*N,1);
Xst = reshape(X_s',3*N,1);
Mloc = getMlocRPY(N,Xst,eps,L,mu,s0,delta);
MFP = FinitePartMatrix(fibpts,X_s,D,Ds,s0,L,N,mu);
M = MFP+Mloc;
[K,Kt]=getKMats3D(Xst,chebyshevmat,w0,N);
fE=FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt)); % elastic force

% Schur complement solve
B=[K I];
C=[Kt; wIt];
RHS = C*fE;
alphaU = lsqminnorm(C*(M \ B),RHS);
alphas=alphaU(1:2*N-2);
Urigid=alphaU(2*N-1:2*N+1);
ut = K*alphas+I*Urigid;
OmegaCrossTau = reshape(Ds*ut,3,N)';
lambda = reshape(M \ ut-fE,3,N)';

% Resample omega,Xs, and lambda on the uniform grid
CoeffsToValsCheb = cos(acos(2*(s0/L)-1).*(0:N-1));
ds = L/(Nuni-1);
sUni = (0:ds:L)';
CoeffstoValsUniform = cos(acos(2*sUni/L-1).* (0:N-1));
ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);
OmegaCrossTau_Uniform = ChebtoUniform*OmegaCrossTau;
Lambda_Uniform = ChebtoUniform*lambda;
% Compute Omega on the uniform grid by crossing with tau on the uniform grid
tau_Uniform = ChebtoUniform*X_s;
Omega_Uniform = cross(tau_Uniform,OmegaCrossTau_Uniform);

