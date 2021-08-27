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
N=40;
Lf=[10 2];   % microns
nonLocal = 0; % whether to do the nonlocal terms 
maxiters = 1;
Ld = 11; % periodic domain size (microns)
xi = 1; % Ewald parameter
mu=1;%0.01;  % 10 x water
eps=[1e-2 0.1];  % slenderness
Eb=[0 8e-3]/3;%0.0411; % 10*kT
omega=0;%2*pi;
T = 2*pi/omega;
gam0=0*omega;
dt = 2.5e-3;
t=0;
tf=5;
grav=0; % for falling fibers, value of gravity
Kspring = 10; % spring constant for cross linkers
rl = 0; % rest length for cross linkers
attachfrac = 0;
FarAway = true;
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
fibpts = [pinv(D{1})*X_s(1:N,:); pinv(D{2})*X_s(N+1:end,:)+[0 Lf(2)/2-attachfrac*Lf(2) 0]];
if (makeMovie)
    f=figure;
    movieframes=getframe(f);
end
Xpts=[];
forces=[];
links = [1 Lf(1)/2 2 attachfrac*Lf(2) 0 0 0];
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
U1 = (cos((0:N-1).*pi/2));
X2rs = U1*Lmat{2}^(-1);
XColor='m';

% Fully 3D slender body theory
% First half step update for the links
%links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt/2);
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
%         Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
%         forces=[forces; reshape(lambdas,3,N*nFib)'];
        X1 = reshape(Xt(1:3*N),3,N)';
        X2 = reshape(Xt(3*N+1:end),3,N)';
        X2mp = X2rs*X2;
        X1all = X1(1,:)+[0:0.001:1]'*(X1(end,:)-X1(1,:));
        d = X1all-X2mp;
        md = min(sqrt(sum(d.*d,2)));
        if (FarAway & md < 0.3)
            FarAway=false;
            timeClose = t;
            COMClose = X2mp;
            XColor='g';
        end
        if (makeMovie) 
            clf;
            makePlot; 
            movieframes(length(movieframes)+1)=getframe(f);
        end
    end
    % Evolve system
    % Background flow, strain, external force
    [U0,g] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1,flowtype);
    fCL = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',N,s0,w0,Lf, Kspring,rl,g,Ld)',3*N*nFib,1);
    gravF = reshape(([0 0 grav/Lf(1)].*ones(N*nFib,3))',3*N*nFib,1);
    maxIters = 100;
    lamguess = lambdas;
    if (t > 1.6*dt)
        maxIters=maxiters;
        lamguess = 2*lambdas-lambdalast;
    end
    lambdalast=lambdas;
    [Xnp1,lambdas,fE,Xsp1]=BESolve(nFib,N,deltaLocal,Lmat,I,wIt,FE,Xt,Xtm1,...
        Xst,Xstm1,U0,dt,s0,w0,Lf,eps,Ds,mu,xi,Ld,nonLocal,lamguess,maxIters,fCL+gravF,g);
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xnp1;
    Xst = Xsp1;
    % Compute total force (per length) at time n+1/2
%     totforce = lambdas+fEstar;
%     totStress = 1/Ld^3*getStress(Xstar,totforce,w0,N,nFib,links,nCL,rl,Kspring,gn,Ld,s0,Lf);
%     fibstresses(count+1)=totStress(2,1);
    % Update myosin
%     if (count==stopcount-1)
%         links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt/2);
%     else
%         links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt);
%     end
    links(2)=min(Lf(1)/2+t,Lf(1));
end
Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
if (makeMovie)
    movieframes(1)=[];
end
% tstr = gam0/omega;
% % Only take modulus over last 2 cycles
% [Gp, Gdp] = getModuli(fibstresses(stopcount/2+1:end)/tstr,stopcount/2,dt,omega);
% % Add fluid
% Gdp = Gdp + mu*omega;
% close;
%save(strcat('Loc_nFib',num2str(nFib),'_nCL',num2str(nCL),'_K',num2str(Kspring),'.mat'))