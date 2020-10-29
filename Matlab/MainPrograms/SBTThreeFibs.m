% This is three fibers in a shear flow, example in the paper
close all;
global Periodic doFP doSpecialQuad;
Periodic=1;
flowtype = 'S'; % S for shear, Q for quadratic
doFP = 1; % no finite part integral
doSpecialQuad=1; % no special quad
deltaLocal = 0.1; % part of the fiber to make ellipsoidal
makeMovie=1;
nFib=3;
nCL = 0;
N=32;
Lf=2;   % microns
nonLocal = 1; % whether to do the nonlocal terms 
maxiters = 1;
Ld = 2.4; % periodic domain size (microns)
xi = 3; % Ewald parameter
mu=1;
eps=1e-3;
Eb=0.01;
dt=0.005;
omega=0;
gam0=1;
t=0;
tf=2.4;
grav=0; % for falling fibers, value of gravity
Kspring = 10; % spring constant for cross linkers
rl = 0.25; % rest length for cross linkers
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 Lf], 2);
[s0,w0,~] = chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.
% 3 fibers for nonlocal test
fibpts = [s0-1 zeros(N,2)-[0.6 0.03]; zeros(N,1) s0-1 zeros(N,1); s0-1 zeros(N,1)+[0.6 0.05]];
D = diffmat(N, 1, [0 Lf], 'chebkind1');
Lmat = cos(acos(2*s0/Lf-1).*(0:N-1));
% max(abs(D*[xp(2:end) yp(2:end) zp]-X_s)) % error in X_s
[Rs,Ls,Ds,Dinv,D4BC,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf);
FE = -Eb*Rs*D4BC;
if (makeMovie)
    f=figure;
    movieframes=getframe(f);
end
Xpts=[];
forces=[];
links=[];
lambdas=zeros(3*N*nFib,1);
lambdalast=zeros(3*N*nFib,1);
fext=zeros(3*N*nFib,1);
Xt= reshape(fibpts',3*N*nFib,1);
Xtm1 = Xt;
Xst=[];
for iFib=1:nFib
    inds=(iFib-1)*N+1:iFib*N;
    Xst=[Xst;reshape((D*fibpts(inds,:))',3*N,1)];
end
Xstm1=Xst;
stopcount=floor(tf/dt+1e-5);
saveEvery=1;%floor(0.1/dt+1e-5);%max(1,floor(1e-4/dt+1e-5));
fibstresses=zeros(stopcount,1);
gn=0;
SBTMain;