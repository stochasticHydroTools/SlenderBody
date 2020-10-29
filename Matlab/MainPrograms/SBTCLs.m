% Main file for a simulation of 4 cross-linked fibers 
% With enough CLs this is an elastic solid
addpath('../chebfun-master')
addpath('../functions-solvers')
close all;
global Periodic doFP doSpecialQuad;
Periodic=1;
flowtype = 'S'; % S for shear, Q for quadratic
doFP = 1;
doSpecialQuad=1;
deltaLocal = 0.1; % part of the fiber to make ellipsoidal
makeMovie=1;
nFib=25;
nCL = 0;%nFib*(nFib-1)/2;
N=16;
Lf=2;   % microns
nonLocal = 1; % whether to do the nonlocal terms 
maxiters = 1;
Ld = 5; % periodic domain size (microns)
xi = 1; % Ewald parameter
mu=1;%0.01;  % 10 x water
eps=1e-3;%2e-3;  % slenderness
Eb=0.01;%0.0411; % 10*kT
omega=2*pi;
T = 2*pi/omega;
gam0=0.1*omega;
dt = 5e-2;
t=0;
tf=5;
grav=0; % for falling fibers, value of gravity
Kspring = 1; % spring constant for cross linkers
rl = 0.3+1e-5; % rest length for cross linkers
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 Lf], 2);
[s0,w0,~] = chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.
PyLocs=load('PyLocs.txt');
%fibpts=PyLocs(1:nFib*N,:);
%fibpts=PyLocs(997*N+1:998*N,:);
% fibpts = [zeros(N,1) s0-0.5 zeros(N,1); zeros(N,1)+0.3 s0-0.5 zeros(N,1);...
%     zeros(N,1)+0.6 s0-0.5 zeros(N,1); zeros(N,1)+0.9 s0-0.5 zeros(N,1)];
fibpts = PyLocs(1:N*nFib,:);
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