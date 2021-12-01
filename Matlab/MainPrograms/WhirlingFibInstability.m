% Main file for single relaxing fiber
close all;
newlabels = 1;
Periodic=0;
flowtype = 'S'; % S for shear, Q for quadratic
clamp0 = 1;
clampL = 0;
exactRPY = 1;
eps=1e-2;
L=2;   % microns
Eb=1;
mu=1;
twmod=1;
a=eps*L;
xir=32/9*pi*mu*a^2;
wcrit = 8.9*Eb/(xir*L^2);
xic = 22.9*Eb*(log(eps^(-2)/16)+4)/(8*pi*mu*L^4);
period = 2*pi/xic;
omegafac = 0.8;
TurnFreq = omegafac*wcrit;
if (~clamp0)
    TurnFreq = 0;
end
deltaLocal = 0; % part of the fiber to make ellipsoidal
makeMovie = 0;
nFib=1;
nCL = 0;%nFib*(nFib-1)/2;
N = 40;
noRotTransAtAll = 0;
if (exactRPY)
    deltaLocal=0;
else
    nonLocal = 0; % whether to do the nonlocal terms 
    includeFP = 0;
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
if (deltaLocal < 0.5)
    t = deltaLocal*L;
    aI_MRR = 1/a^2*(5/8+t/a-27*t^2/(64*a^2)+5*t^4/(256*a^4)-1/2*(1/8-a^2/(2*(L-t)^2)));
    aTau_MRR = 1/a^2*(3/8+9/64*t^2/a^2 - 3/256*t^4/a^4+3/2*(1/8-a^2/(2*(L-t)^2)));
    ZeroRes = aI_MRR+aTau_MRR;
else
    ZeroRes = 9/(4*a^2);
end
theta0=(8*pi*mu/ZeroRes)*TurnFreq/twmod*(s.^2/2-L*s);
if (twmod==0)
    theta0 = zeros(N,1);
end
theta0 = theta0-barymat(L/2,s,b)*theta0;
deflect = 0.01;
X_s = [deflect*cos(s.* (s-L).^3) ones(N,1) deflect*sin(s.*(s - L).^3)]/sqrt(1+deflect^2);
% Apply rotation matrix
p = atan(deflect);
R = [cos(p) -sin(p) 0; sin(p) cos(p) 0; 0 0 1];
X_s = (R*X_s')';
saveEvery=1e-2/dtfactor;%max(1,floor(2e-4/dt+0.001));
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
        Thetass=[Thetass; theta_s];
        endPoints = [endPoints; barymat(L,s,b)*reshape(Xt,3,N)'];
    end
    % Evolve system
    % Background flow, strain, external force
    [U0,strain] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1,flowtype);
    fCL = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',N,s,w,...
        L, Kspring,rl,strain,Ld)',3*N*nFib,1);
    gravF = reshape(([0 0 grav/L(1)].*ones(N*nFib,3))',3*N*nFib,1);
    maxIters = 100;
    lamguess = lambdas;
    if (t > 1.6*dt)
        maxIters=maxiters;
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