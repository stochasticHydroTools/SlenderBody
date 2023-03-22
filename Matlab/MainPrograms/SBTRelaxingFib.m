% Main file for single relaxing fiber, to test the influence of twist
% elasticity 
% This Section 5.2 in the paper Maxian et al. "The hydrodynamics of a
% twisting, bending, inextensible filament in Stokes flow"
%close all;
addpath('../functions-solvers')
%PenaltyForceInsteadOfFlow = 1; kbT=0; ModifyBE=1; gam0=2000*kbT;
exactRPY=1; 
deltaLocal=0;
noRotTransAtAll=0;
nonLocalRotTrans=1;
%N = 39;   
RectangularCollocation = 1; 
clamp0=0; 
%twmod=0.1;
L = 2;            
eps = 1e-2;
a = eps*L;
Eb=1;           % Bending modulus
mu=1;           % Viscosity
makeMovie = 0;
if (makeMovie)
    f=figure;
    movieframes=getframe(f);
end
initZeroTheta=1;
NupsampleHydro=100;
nFib=1;
%dt = 1e-4;
tf = 0.01;
stopcount = floor(1e-6+tf/dt);
impcoeff = 1;
t=0;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
D = diffmat(N, 1, [0 L], 'chebkind1');
% Fiber initialization 
q=7; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
XMPor0 = [0;0;0]; XMP=XMPor0;
saveEvery=max(1e-4/dt,1);
InitFiberVars;
updateFrame=1;
Xpts=[];
Thetass=[];
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            clf;
            plot3(Xt(1:3:end),Xt(2:3:end),Xt(3:3:end))
            movieframes(length(movieframes)+1)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,N+1)'];
        Thetass = [Thetass; theta_s];
    end
    U0 = zeros(3*Nx,1);
    n_ext = zeros(Npsi,1);
    f_ext = zeros(3*Nx,1);
    TemporalIntegrator_wTwist1Fib;
    Xt = Xp1;
    Xst = Xsp1;
    XMPor0 = XMPor0_p1;
    theta_s = theta_sp1;
end
Xpts=[Xpts;reshape(Xt,3,N+1)'];
Thetass = [Thetass; theta_s];
if (makeMovie)
    movieframes(1)=[];
end