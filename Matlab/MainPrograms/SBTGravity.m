% Main file for 4 falling fibers (example in our paper)
close all;
Periodic=0;
flowtype = 'S'; % S for shear, Q for quadratic
doSpecialQuad=0; % no special quad
newlabels = 1:4;
deltaLocal = 0.5; % part of the fiber to make ellipsoidal
makeMovie=0;
nFib=4;
nCL = 0;%nFib*(nFib-1)/2;
strongthetaBC = 0;
N=16; 
NupsampleforNL=N;
L=2;   % microns
interFiberHydro = 1; % whether to do the nonlocal terms 
maxiters = 1;
Ld = 4; % periodic domain size (microns) not relevant here
xi = 0; % Ewald parameter not relevant here
mu=1;
eps=1e-3;
Temporal_order = 2;
exactRPY = 0;
includeFPonLHS = 0;
includeFPonRHS = 0;
if (exactRPY)
    includeFPonRHS=0;
    includeFPonLHS=0;
end
a = exp(3/2)/4*eps*L; % match SBT
noRotTransAtAll = 0;
nonLocalRotTrans = 1;
bottomwall = 0;
Eb=1;
dt=1.25e-3/2;
clamp0 = 0;
clampL = 0;
twmod = 0;
omega=0;
gam0=0; % no shear
t=0;
tf=0.25;
grav=-10; % for falling fibers, value of gravity
Kspring = 10; % spring constant for cross linkers
rl = 0.25; % rest length for cross linkers
% Nodes for solution, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
d=0.2;
fibpts=[d*ones(N,1) 0*ones(N,1) (s-1); 0*ones(N,1) d*ones(N,1) (s-1); ...
    -d*ones(N,1) 0*ones(N,1) (s-1); 0*ones(N,1) -d*ones(N,1) (s-1)];
D = diffmat(N, 1, [0 L], 'chebkind1');
Lmat = cos(acos(2*s/L-1).*(0:N-1));
EvalMat=(vander(s-L/2));
theta0 = zeros(4*N,1);
theta_s = zeros(4*N,1);
X_s = zeros(nFib*N,3);
for iFib=1:nFib
    inds = (iFib-1)*N+1:iFib*N;
    X_s(inds,:)=D*fibpts(inds,:);
end
saveEvery = 1;
InitFiberVars;
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            clf;
            makePlot; 
            movieframes(length(movieframes)+1)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
    end
    % Evolve system
    % Background flow, strain, external force
    [U0,strain] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1,flowtype);
    fCL = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',N,sp4,w,...
        L, Kspring,rl,strain,Ld)',3*N*nFib,1);
    gravF = reshape(([0 0 grav/L(1)].*ones(N*nFib,3))',3*N*nFib,1);
    lamguess = lambdas;
    if (t > 1.6*dt)
        lamguess = 2*lambdas-lambdalast;
    end
    lambdalast=lambdas;
    fext = fCL+gravF;
    TemporalIntegrator;
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xp1;
    Xst = Xsp1;
    theta_s = theta_sp1;
%     Ders=[Ders;max([max(abs(D1next-D1next2));max(abs(D2next-D2next2))])];
%     OmegaPar0s = [OmegaPar0s; OmegaPar0_re];
%     D1 = D1next2;
%     D2 = D2next2;
    %XBCMat_low*reshape(Xt,3,N)'-reshape(BCanswers,3,4)'
    %ThetaBCMat_low*theta
    % Compute total force (per length) at time n+1/2
%     totforce = lambdas+fEstar;
%     totStress = 1/Ld^3*getStress(Xstar,totforce,w0,N,nFib,links,nCL,rl,Kspring,gn,Ld,s0,L);
%     fibstresses(count+1)=totStress(2,1);
end
Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
if (makeMovie)
    movieframes(1)=[];
end