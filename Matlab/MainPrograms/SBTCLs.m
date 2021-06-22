% Main file for a simulation of 4 cross-linked fibers 
% With enough CLs this is an elastic solid
addpath('../chebfun-master')
addpath('../functions-solvers')
close all;
global Periodic doFP doSpecialQuad;
Periodic=0;
flowtype = 'S'; % S for shear, Q for quadratic
doFP = 1;
doSpecialQuad=1;
deltaLocal = 0.1; % part of the fiber to make ellipsoidal
makeMovie=1;
nFib=2;
nCL = 1;%nFib*(nFib-1)/2;
N=64;
Lf=[10 2];   % microns
nonLocal = 0; % whether to do the nonlocal terms 
maxiters = 1;
Ld = 11; % periodic domain size (microns)
xi = 1; % Ewald parameter
mu=1;%0.01;  % 10 x water
eps=[1e-2 0.1];  % slenderness
Eb=[0 8e-3];%0.0411; % 10*kT
omega=0;%2*pi;
T = 2*pi/omega;
gam0=0*omega;
dt = 5e-3;
t=0;
tf=5;
grav=0; % for falling fibers, value of gravity
Kspring = 10; % spring constant for cross linkers
rl = 0.05; % rest length for cross linkers
% Nodes for solution, plus quadrature and barycentric weights:
for iFib=1:2
    [s{iFib}, ~, b{iFib}] = chebpts(N+4, [0 Lf(iFib)], 2);
    [s0{iFib},w0{iFib},~] = chebpts(N, [0 Lf(iFib)], 1); % 1st-kind grid for ODE.
    if (iFib==1)
        X_s(1:N,:) = [ones(N,1) zeros(N,2)]; 
    else
        X_s(N+1:2*N,:)=[zeros(N,1) ones(N,1) zeros(N,1)];
    end
    D{iFib} = diffmat(N, 1, [0 Lf(iFib)], 'chebkind1');
    Lmat{iFib} = cos(acos(2*s0{iFib}/Lf(iFib)-1).*(0:N-1));
    [Rs{iFib},Ls{iFib},Ds{iFib},Dinv{iFib},D4BC{iFib},I{iFib},wIt{iFib}]=...
        stackMatrices3D(s0{iFib},w0{iFib},s{iFib},b{iFib},N,Lf(iFib));
    FE{iFib} = -Eb(iFib)*Rs{iFib}*D4BC{iFib};
end
fibpts = [pinv(D{1})*X_s(1:N,:); pinv(D{2})*X_s(N+1:end,:)+[0 Lf(2)/2+rl 0]];
if (makeMovie)
    f=figure;
    movieframes=getframe(f);
end
Xpts=[];
forces=[];
links = [1 Lf(1)/2 2 0 0 0 0];
lambdas=zeros(3*N*nFib,1);
lambdalast=zeros(3*N*nFib,1);
fext=zeros(3*N*nFib,1);
Xt= reshape(fibpts',3*N*nFib,1);
Xtm1 = Xt;
Xst=[];
for iFib=1:nFib
    inds=(iFib-1)*N+1:iFib*N;
    Xst=[Xst;reshape((D{iFib}*fibpts(inds,:))',3*N,1)];
end
Xstm1=Xst;
stopcount=floor(tf/dt+1e-5);
saveEvery=2;%floor(0.1/dt+1e-5);%max(1,floor(1e-4/dt+1e-5));
fibstresses=zeros(stopcount,1);
gn=0;
SBTMain;