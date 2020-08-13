% File to solve for alpha and lambda for a single fiber
% i.e., solve system
% [-M K; K^T 0]*[alpha;lambda]=[M*fE;0]
% Using a spectral discretization
% Important variables
rng(0);
N=24;       % number of Chebyshev pts
Nuni = 100; % number of uniform points (blobs)
L=2;        % fiber length
mu=1;       % fluid viscosity
eps=1e-3;   % r/L aspect ratio
Eb=1;       % bending constant 
% delta = 0.2; % fraction of length over which to taper the fiber

% Chebyshev initialization
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 L], 2);
[s0,w0,b0] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
[Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,L);
W = diag(reshape([w0; w0; w0],3*N,1));
FE = -Eb*Rs*D4s;

% Fiber initialization (Floren's configuration)
% X_s = [cos(s0.^3 .* (s0-L).^3) sin(s0.^3.*(s0 - L).^3) ones(N,1)]/sqrt(2);
% opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
% [~,xp] = ode45(@(s,X) cos(s^3 * (s-L)^3)/sqrt(2),[0; s0],0,opts);
% [~,yp]= ode45(@(s,X) sin(s^3 * (s-L)^3)/sqrt(2),[0; s0],0,opts);
% zp = s0/sqrt(2);
% fibpts=[xp(2:end) yp(2:end) zp];
X_s = [ones(N,1) zeros(N,2)];
fibpts = [s0 zeros(N,2)];

% Compute matrices K and M
Xt= reshape(fibpts',3*N,1);
Xst = reshape(X_s',3*N,1);
%Mloc = getMlocRPY(N,Xst,eps,L,mu,s0,delta);
Mloc = getMlocRPYUnreg(N,Xst,eps,L,mu,s0);
MFP = FinitePartMatrix(fibpts,X_s,D,Ds,s0,L,N,mu);
% Upsample to 2000 pts and do finite part by skipping sing pt
% Nmax=512;
% [sLg, wLg, bLg] = chebpts(Nmax, [0 L], 1);
% Rdwn = barymat(s0, sLg, bLg); % Resampling matrix.
% Rdwns=zeros(3*N,3*Nmax);
% Rdwns(1:3:3*N,1:3:3*Nmax)=Rdwn;
% Rdwns(2:3:3*N,2:3:3*Nmax)=Rdwn;
% Rdwns(3:3:3*N,3:3:3*Nmax)=Rdwn;
% Rup = barymat(sLg, s0, b0); % Resampling matrix.
% Rups=zeros(3*Nmax,3*N);
% Rups(1:3:3*Nmax,1:3:3*N)=Rup;
% Rups(2:3:3*Nmax,2:3:3*N)=Rup;
% Rups(3:3:3*Nmax,3:3:3*N)=Rup;
% FPMatSkip = 1/(8*pi*mu)*Rdwns*NLMatrix(Rups*Xt,Rups*Xst,sLg,wLg,Nmax,10*eps*L)*Rups;
% MFP = FPMatSkip;

M = Mloc;%+MFP;
disp('No finite part!')
[K,Kt]=getKMats3D(Xst,chebyshevmat,w0,N);
%fb=FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt)); % elastic force
fb = reshape([ones(N,1) zeros(N,2)]',3*N,1);

% Eigenvalue estimates
% gamn=2*sum(1./(1:N-1)); % bound on finite part eigenvalue
% aI = 47/12-log(24);
% atau = 1/4-log(24);
% cmin = log(4*delta/2*(1-delta/2)/eps^2);
% lampar = 2*cmin+atau+aI-2*gamn;
% lamperp = cmin+aI-gamn;
% if (lampar < 0 || lamperp < 0)
%     warning('Solvability condition violated - expect oscillations in solution')
%     if (firstviolation==-1)
%         firstviolation=N;
%     end
% end

% Schur complement solve
B=[K I];
C=[Kt; wIt];
RHS = C*fb;
alphaU = lsqminnorm(C*(M \ B),RHS);
alphas=alphaU(1:2*N-2);
Urigid=alphaU(2*N-1:2*N+1);
ut = K*alphas+I*Urigid;
OmegaCrossTau = reshape(Ds*ut,3,N)';
lambda = reshape(M \ut -fb,3,N)';

% Resample omega,Xs, and lambda on the uniform grid
CoeffsToValsCheb = cos(acos(2*(s0/L)-1).*(0:N-1));
ds = L/(Nuni-1);
sUni = (0:ds:L)';
CoeffstoValsUniform = cos(acos(2*sUni/L-1).* (0:N-1));
ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);

% Resample X on the grid
OmegaCrossTau_Uniform = ChebtoUniform*OmegaCrossTau;
Lambda_Uniform = ChebtoUniform*lambda;
utilde_Uniform = ChebtoUniform*reshape(ut,3,N)';
% Compute Omega on the uniform grid by crossing with tau on the uniform grid
tau_Uniform = ChebtoUniform*X_s;
Omega_Uniform = cross(tau_Uniform,OmegaCrossTau_Uniform);
% Alternative: compute Omega by going through upsampled Cheb grid
[s2N,w2N,b2N] = chebpts(2*N, [0 L], 1); % 1st-kind grid for ODE.
Rup = barymat(s2N,s0,b0);
Rdwn = barymat(s0,s2N,b2N);
Omega_Cheb = Rdwn*cross(Rup*X_s,Rup*OmegaCrossTau);
Omega_UniformFromCheb = ChebtoUniform*Omega_Cheb;