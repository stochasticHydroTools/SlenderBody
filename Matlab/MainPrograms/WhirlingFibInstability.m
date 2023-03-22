% Main program for simulation of a twirling fiber that goes unstable
% (whirling instability)
% This Section 5.3 in the paper Maxian et al. "The hydrodynamics of a
% twisting, bending, inextensible filament in Stokes flow"
% The variables below are documented in SBTRelaxingFib.m. Here I just
% clarify the ones that are new/different. 
%close all;
addpath('../functions-solvers')
RectangularCollocation = 1;
clamp0 = 1;
TorqBC = 0;
exactRPY = 1;
epshat=1e-2;
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
omegafac = 1;
TurnFreq = omegafac*wcrit;
smperiod = 2*pi/TurnFreq;
deltaLocal = 0; % part of the fiber to make ellipsoidal
makeMovie = 0;
if (makeMovie)
    f=figure;
    movieframes(1)=getframe(f);
end
updateFrame = 1;
nFib = 1;
N = 40;
noRotTransAtAll = 0;
nonLocalRotTrans=1;
Temporal_order=1;
dtfactor = 1e-3;
dt=dtfactor*period;
t=0;
tf=5*period;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
initZeroTheta=0;
deflect = 0.01;
X_s = [deflect*cos(s.* (s-L).^3) ones(N,1) deflect*sin(s.*(s - L).^3) ]/sqrt(1+deflect^2);
p = atan(deflect);
% Apply rotation matrix to get the right BC in X_s
R = [cos(p) -sin(p) 0 ;sin(p) cos(p) 0;0 0 1 ];
X_s = (R*X_s')';
X0BC=[0;0;0];
Tau0BC=[0;1;0];
XMPor0=[0;0;0]; 
XMP = XMPor0;% for initialization
saveEvery=max(1e-3/dtfactor,1);
InitFiberVars;
% Parameters for the bottom wall, where we take upsampled integrals 
Npl=1000;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
RplN = barymat(spl,s,b);
Rpl=RplNp1;
gn = 0;
links=[];
Xtm1 = Xt;
endPoints=[];
D1s=[];
BCers=[];
stopcount=floor(tf/dt+1e-5);
Xpts=[];
Thetass=[];
Nusteric = 101;
suSteric = (0:Nusteric-1)'*L/(Nusteric-1);
RuniSteric = barymat(suSteric,sNp1,bNp1);
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            clf;
            plot3(Xt(1:3:end),Xt(2:3:end),Xt(3:3:end))
            movieframes(length(movieframes)+1)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,(N+1)*nFib)'];
        Thetass=[Thetass; theta_s];
        endPoints = [endPoints; barymat(L,sNp1,bNp1)*reshape(Xt,3,N+1)'];
        D1s=[D1s; D1];
        BCdiff=XBCMat_low*reshape(Xt,3,N+1)';
        BCdiff(2,:)=BCdiff(2,:)-[0 1 0];
        BCers=[BCers;BCdiff];
    end
    % Evolve system
    U0 = zeros(3*Nx,1);
    n_ext = zeros(Npsi,1);
    f_ext = zeros(3*Nx,1);
    StericForce = getStericForce(reshape(Xt,3,[])',RuniSteric,a,4.1e-3,suSteric,1,0,100);
    StericForceDen= WTilde_Np1\ reshape(StericForce',[],1);
    f_ext = StericForceDen;
    %periodnumber = floor(t/(2*period));
    %TurnFreq=(0.96+periodnumber*0.08)*wcrit;
    TemporalIntegrator_wTwist1Fib;
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xp1;
    Xst = Xsp1;
    theta_s = theta_sp1;
    D1=D1next;
    Xrs = RplNp1*reshape(Xt,3,N+1)';
    for ip=1:1000
        displac = Xrs-Xrs(ip,:);
        displac = sqrt(sum(displac.*displac,2));
        displac(abs(spl(ip)-spl) < 2.2*a)=inf;
        [val,ind]=min(displac);
        if (val < 2*a)
            warning('Possible encroachment!')
        end
    end
end
Xpts=[Xpts;reshape(Xt,3,(N+1)*nFib)'];
D1s=[D1s; D1];
if (makeMovie)
    movieframes(1)=[];
end