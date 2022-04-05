% Main program for simulation of a twirling fiber that goes unstable
% (whirling instability)
% This Section 5.3 in the paper Maxian et al. "The hydrodynamics of a
% twisting, bending, inextensible filament in Stokes flow"
% The variables below are documented in SBTRelaxingFib.m. Here I just
% clarify the ones that are new/different. 
%close all;
addpath('../functions-solvers')
newlabels = 1;
Periodic=0; 
flowtype = 'S'; % S for shear, Q for quadratic, not relevant here
clamp0 = 1;
clampL = 0;
TorqBC = 0;
strongthetaBC = 0;
exactRPY = 1;
epshat=1e-2;
bottomwall = 0;
L=2;   % microns
Eb=1;
mu=1;
twmod=1;
a=epshat*L;
% Compute the critical frequency according to theory, then set the actual
% frequency based on that
xir=32/9*pi*mu*a^2;
wcrit = 8.9*Eb/(xir*L^2);
xic = 22.9*Eb*(log(epshat^(-2)/16)+4)/(8*pi*mu*L^4);
period = 2*pi/xic;
omegafac = 0.8;
TurnFreq = omegafac*wcrit;
smperiod = 2*pi/TurnFreq;
deltaLocal = 0; % part of the fiber to make ellipsoidal
makeMovie = 1;
nFib=1;
nCL = 0;%nFib*(nFib-1)/2;
N = 40;
interFiberHydro = 0;
includeFPonRHS = 0;
includeFPonLHS = 0;
noRotTransAtAll = 0;
if (exactRPY)
    deltaLocal=0;
else
    nonLocal = 0; % whether to do the nonlocal terms 
    includeFPonRHS = 0;
    includeFPonLHS = 0;
    nonLocalRotTrans = 0;
    doSpecialQuad=0; % no special quad
end
%disp('Finite part OFF')
%disp('Doing backward Euler & first order rotation')
maxiters = 1;
Temporal_order=1;
Ld = 8; % periodic domain size (microns) not relevant here
xi = 0; % Ewald parameter not relevant here
dtfactor = 5e-4;
dt=dtfactor*period;
%dt = 0.1*smperiod;
omega=0;
gam0=0; % no shear
t=0;
tf=5*period;
grav=0; % for falling fibers, value of gravity
Kspring = 0; % spring constant for cross linkers
rl = 0; % rest length for cross linkers
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
% Fiber initialization (Floren's configuration)
D = diffmat(N, 1, [0 L], 'chebkind1');
[~, ~, ~,Mrr,~] = getGrandMloc(N,zeros(3*N,1),zeros(3*N),a,L,mu,s,deltaLocal);
theta_s=TurnFreq/twmod*pinv(D)*(Mrr \ ones(N,1));
theta_s = theta_s-barymat(L,s,b)*theta_s;
if (twmod==0)
    theta0 = zeros(N,1);
end
theta0 = pinv(D)*theta_s;
theta0 = theta0-barymat(L/2,s,b)*theta0;
deflect = 0.01;
%X_s = [deflect*cos(s.* (s-L).^3) deflect*sin(s.*(s - L).^3) ones(N,1) ]/sqrt(1+deflect^2);
X_s = [deflect*cos(s.* (s-L).^3) ones(N,1) deflect*sin(s.*(s - L).^3) ]/sqrt(1+deflect^2);
% Apply rotation matrix
p = atan(deflect);
%R = [cos(p) 0 -sin(p); 0 1 0; sin(p) 0 cos(p)];
R = [cos(p) -sin(p) 0 ;sin(p) cos(p) 0;0 0 1 ];
X_s = (R*X_s')';
fibpts = pinv(D)*X_s;
fibpts=fibpts-barymat(0,s,b)*fibpts;%+[0 0 2*a];
saveEvery=1e-2/dtfactor;
endPoints=[];
InitFiberVars;
% Parameters for the bottom wall, where we take upsampled integrals 
Nups=100;
[sup,wup,bup]=chebpts(Nups,[0 L],1);
Rglobalupsample = barymat(sup, s, b);
Rglobaldown = barymat(s,sup,bup);
Wmat=diag(reshape(repmat(wup,3,1),3*Nups,1));
D1s=[];
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
        Thetass=[Thetass; theta_s];
        endPoints = [endPoints; barymat(L,s,b)*reshape(Xt,3,N)'];
        D1s=[D1s; D1];
    end
    % Evolve system
    fext = zeros(3*N,1);
    lamguess = zeros(3*N,1);
    U0 = zeros(3*N,1);
    TemporalIntegrator;
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xp1;
    Xst = Xsp1;
    theta_s = theta_sp1;
    D1=D1next;
end
Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
D1s=[D1s; D1];
if (makeMovie)
    movieframes(1)=[];
end