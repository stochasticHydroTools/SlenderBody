% Main file for single relaxing fiber
%close all;
newlabels = 1;
Periodic=0;
flowtype = 'S'; % S for shear, Q for quadratic
clamp0 = 0;
clampL = 0;
exactRPY=1;
eps=1e-2;
L=2;   % microns
a=eps*L;
Eb=1;
mu=1;
twmod = 1;
deltaLocal = 0; % part of the fiber to make ellipsoidal
makeMovie = 0;
nFib=1;
nCL = 0;%nFib*(nFib-1)/2;
N = 30;
noRotTransAtAll=0;
if (~exactRPY)
    includeFP = 1;
    nonLocalRotTrans = 0;
    doSpecialQuad=0; % no special quad
end
%disp('Finite part OFF')
%disp('Doing backward Euler & first order rotation')
maxiters = 1;
Temporal_order=1;
Ld = 8; % periodic domain size (microns) not relevant here
xi = 0; % Ewald parameter not relevant here
dt=1e-5;
omega=0;
gam0=0; % no shear
t=0;
tf=0.01;
grav=0; % for falling fibers, value of gravity
Kspring = 0; % spring constant for cross linkers
rl = 0; % rest length for cross linkers
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
% Fiber initialization (Floren's configuration)
q=7; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
%%% loopy fiber 
% t1 = sin(3*(pi/2)*(s/2-sin(2*pi*s/2)/(2*pi))).*cos((pi)*(s/2-sin(2*pi*s/2)/(2*pi)));
% t2 = sin(3*(pi/2)*(s/2-sin(2*pi*s/2)/(2*pi))).*sin((pi)*(s/2-sin(2*pi*s/2)/(2*pi)));
% t3 = cos(3*(pi/2)*(s/2-sin(2*pi*s/2)/(2*pi)));
% X_s = [t1 t2 t3];
Lmat = cos(acos(2*s/L-1).*(0:N-1));
theta0 = 0*s;
theta0 = theta0-barymat(L/2,s,b)*theta0;
saveEvery=max(1,floor(1e-4/dt+0.001));
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