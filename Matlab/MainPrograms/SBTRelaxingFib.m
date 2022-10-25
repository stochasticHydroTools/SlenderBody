% Main file for single relaxing fiber, to test the influence of twist
% elasticity 
% This Section 5.2 in the paper Maxian et al. "The hydrodynamics of a
% twisting, bending, inextensible filament in Stokes flow"
%close all;
addpath('../functions-solvers')
newlabels = 1;
exactRPY=1;     % whether to use exact RPY integrals (1 for RPY, 0 for SBT)
N = 24;         % Number of Chebyshev collocation points
L = 2;            % fiber length=
eps = 1e-3;
a = eps*L*exp(3/2)/4;
Eb=1;           % Bending modulus
mu=1;           % Viscosity
interFiberHydro = 0;  % Inter-fiber hydrodynamics; irrelevant for a single fiber
twmod = 0;       % Twist modulus
deltaLocal = 1; % part of the fiber to make ellipsoidal/taper the radius function ove
makeMovie = 0;
nFib=1;
nCL = 0;
NupsampleforNL = N;
noRotTransAtAll=0; % 0 for including rot-trans, 1 for no rot-trans at all
if (~exactRPY)
    includeFPonRHS = 0;
    includeFPonLHS = 1;
    nonLocalRotTrans = 0;
    doSpecialQuad=0; % no special quad
else
    deltaLocal=0;
end
Temporal_order = 1;
bottomwall = 0;
maxiters = 1;
dt=1e-4;
tf = 1e-2;
t=0;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
D = diffmat(N, 1, [0 L], 'chebkind1');
% Fiber initialization 
q=1; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
Lmat = cos(acos(2*s/L-1).*(0:N-1));
theta0 = 0*s;
theta0 = theta0-barymat(L/2,s,b)*theta0;
saveEvery=1;%max(1,floor((tf/100)/dt+0.001));
theta_s = D*theta0;
fibpts = pinv(D)*X_s;
fibpts=fibpts-barymat(L/2,s,b)*fibpts;
PenaltyForceInsteadOfFlow = 0; clamp0=0; clampL=0; strongthetaBC=0;
InitFiberVars;
lambdaprev = lambdas;
D1s=[];
try
Energies=zeros(stopcount,1);
catch
Energies=[];
end
EnergiesBC=Energies;
reg=0; 
if (reg>0)
    Sm = MobilitySmoother(s,w,reg*L);
end
spl = chebpts(1000,[0 L],2);
Rpl = barymat(spl,s,b);
End0=stackMatrix(barymat(L/2,s,b))*Xt;
Xt = Xt-repmat(End0,N,1);
% XMP = XMP-End0;
%warning('Blurred temporal integrator!')
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    Energies(count+1) = Xt'*BendingEnergyMatrix*Xt;
    EnergiesBC(count+1) = Xt'*BendingEnergyMatrixBC*Xt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            clf;
            plot3(Xt(1:3:end),Xt(2:3:end),Xt(3:3:end))
            movieframes(length(movieframes)+1)=getframe(f);
        end
        Xpts=[Xpts;Rpl*reshape(Xt,3,N*nFib)'];
        Thetass=[Thetass; theta_s];
        D1s=[D1s; D1];
    end
    % Evolve system
    fext = zeros(3*N,1);
    lamguess = lambdas;
    if (Temporal_order==2)
        lamguess = 2*lambdas-lambdaprev;
    end
    U0 = zeros(3*N,1);
    lambdaprev = lambdas;
    TemporalIntegrator;
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xp1;
    Xst = Xsp1;
    theta_s = theta_sp1;
    D1= D1next;
end
Xpts=[Xpts;Rpl*reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
D1s=[D1s; D1];
if (makeMovie)
    movieframes(1)=[];
end