% Main file for single relaxing fiber, to test the influence of twist
% elasticity 
% This Section 5.2 in the paper Maxian et al. "The hydrodynamics of a
% twisting, bending, inextensible filament in Stokes flow"
%close all;
addpath('../functions-solvers')
newlabels = 1;
Periodic=0;     % Is domain periodic or free space?
flowtype = 'S'; % S for shear, Q for quadratic, not relevant here
clamp0 = 0;     % is the s=0 end clamped?
clampL = 0;     % is the s=L end clamped?
exactRPY=1;     % whether to use exact RPY integrals (1 for RPY, 0 for SBT)
TorqBC = 0;     % whether to use a BC on torque (i.e., N = twmod*theta_s) at the clamped end
TurnFreq=0;     % frequency at which the clamped end spins
if (~RandIn)
    %N = 45;         % Number of Chebyshev collocation points
    L=2;            % fiber length=
    %eps = 1e-2;
    a = eps*L;
end
%epshat=a/L;    % a/L
%a = eps*L;
%eps = a/L;
Eb=1;           % Bending modulus
mu=1;           % Viscosity
strongthetaBC = 0;  % Whether to enforce the BC for theta in the strong or weak sense
bottomwall = 0;     % Whether there is a bottom wall that influences the hydrodynamics
interFiberHydro = 0;  % Inter-fiber hydrodynamics; irrelevant for a single fiber
includeFPonRHS = 0;   % For SBT calculations, whether to include the finite part integral on the RHS (explicit)
includeFPonLHS = 0;   % For SBT calculations, whether to include the finite part integral on the LHS (implicit)
%twmod = 0;       % Twist modulus
deltaLocal = 0; % part of the fiber to make ellipsoidal/taper the radius function ove
makeMovie = 0;
nFib=1;
nCL = 0;
NupsampleforNL = N;
noRotTransAtAll=0; % 0 for including rot-trans, 1 for no rot-trans at all
if (~exactRPY)
    includeFPonRHS = 0;
    includeFPonLHS = 0;
    nonLocalRotTrans = 0;
    doSpecialQuad=0; % no special quad
else
    deltaLocal=0;
end
%disp('Finite part OFF')
%disp('Doing backward Euler & first order rotation')
maxiters = 1;
Temporal_order = 1;
Ld = 8; % periodic domain size (microns) not relevant here
xi = 0; % Ewald parameter not relevant here
%if (~RandIn)
%    dt=1e-5;
%end
omega=0;
gam0=0; % no shear
t=0;
if (~RandIn)
    tf=1e-2;
else
    tf=1e-2;
end
grav=0; % for falling fibers, value of gravity
Kspring = 0; % spring constant for cross linkers
rl = 0; % rest length for cross linkers
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
D = diffmat(N, 1, [0 L], 'chebkind1');
% Fiber initialization 
if (~RandIn)
    q=7; 
    X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
else
    % Obtain positions at Chebyshev nodes
    Tau = taus';
    X=[0 0 0; cumsum(taus'*ds)];
    XInterp = zeros(N,3);
    TauInterp = zeros(N,3);
    for iPt=1:N
        [~,xlowind] = min(1./(s_x-s(iPt)));
        % Find the tangent vector
        ThisTau = Tau(xlowind,:);
        XInterp(iPt,:) =  ThisTau*(s(iPt)-s_x(xlowind))+X(xlowind,:);
        TauInterp(iPt,:) = ThisTau;
    end
    TauHat = D*XInterp;
    X_s = TauHat./sqrt(sum(TauHat.*TauHat,2));
end
Lmat = cos(acos(2*s/L-1).*(0:N-1));
theta0 = 0*s;
theta0 = theta0-barymat(L/2,s,b)*theta0;
saveEvery=max(1,floor((tf/100)/dt+0.001));
theta_s = D*theta0;
fibpts = pinv(D)*X_s;
fibpts=fibpts-barymat(0,s,b)*fibpts;
if (RandIn)
fibpts=fibpts-barymat(L/2,s,b)*fibpts+XMP;
end
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
%warning('Blurred temporal integrator!')
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    Energies(count+1) = Xt'*BendingEnergyMatrix*Xt;
    EnergiesBC(count+1) = Xt'*BendingEnergyMatrixBC*Xt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            clf;
            makePlot; 
            movieframes(length(movieframes)+1)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
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
Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
D1s=[D1s; D1];
if (makeMovie)
    movieframes(1)=[];
end