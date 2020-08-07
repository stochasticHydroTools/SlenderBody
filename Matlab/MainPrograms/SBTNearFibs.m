% Initialization file for the slender body simulatons
close all;
global Periodic flowtype doFP doSpecialQuad deltaLocal;
Periodic=0;
flowtype = 'Q';
doFP = 1;
doSpecialQuad=0;
deltaLocal = 0.2;
N=16;
makeMovie=1;
nFib=1;
nCL = 0;
Lf=1.8;   % microns
nonLocal = 1; % whether to do the nonlocal terms 
LMMlam = 1;
maxiters = 1;
Ld = 5; % periodic domain size (microns)
xi = 2; % Ewald parameter
mu=1;%0.01;  % 10 x water
eps=1e-3;%2e-3;  % slenderness
Eb=0.1;%0.0411; % 10*kT
dt=1e-2;
omega=0;%2*pi;
gam0=1;%0.2*pi;
t=0;
tf=0.5;
grav=0; % for falling fibers, value of gravity
Kspring = 10; % spring constant for cross linkers
rl = 0.25; % rest length for cross linkers
% Nodes for solution, plus quadrature and barycentric weights:
[s, ~, b] = chebpts(N+4, [0 Lf], 2);
[s0,w0,~] = chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.
% 2 fibers for nonlocal test
fibpts = [-0.05*ones(N,1) s0-1 zeros(N,1)];
D = diffmat(N, 1, [0 Lf], 'chebkind1');
[Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf);
FE = -Eb*Rs*D4s;
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
    Xst=[Xst;reshape((D*fibpts((iFib-1)*N+1:iFib*N,:))',3*N,1)];
end
Xstm1=Xst;
stopcount=floor(tf/dt+1e-5);
saveEvery=max(floor(0.01/dt+1e-5),1);
tic
fibstresses=zeros(stopcount,1);
gn=0;
SBTMain;
%% Restart
nFib=2;
omega=0;%2*pi;
gam0=0;%0.2*pi;
fibpts = Xpts(end-N+1:end,:)-[min(Xpts(end-N+1:end,1)) 0 0]; % center at 0
fibpts = [fibpts; 0.11*ones(N,1) zeros(N,1) s0-1];
Xt= reshape(fibpts',3*N*nFib,1);
Xtm1 = Xt;
Xst=[];
lambdas=zeros(3*N*nFib,1);
lambdalast=zeros(3*N*nFib,1);
fext=zeros(3*N*nFib,1);
for iFib=1:nFib
    Xst=[Xst;reshape((D*fibpts((iFib-1)*N+1:iFib*N,:))',3*N,1)];
end
Xstm1=Xst;
t=0;
tf=3;
stopcount=floor(tf/dt+1e-5);
saveEvery=max(floor(0.01/dt+1e-5),1);
Xpts=[];
forces=[];
links=[];
tic
SBTMain;
eval(strcat('N',num2str(N),'NoSQ_dt1e2=Xpts;'))
