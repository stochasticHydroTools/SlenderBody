% Twisting cross-linked bundle
close all;
addpath('../../functions-solvers')

%% Physical parameters
poly = 1;
L = 2;   % microns
rtrue = 4e-3; % 4 nm radius
eps=rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
twmod = 1e-15;%kbT; % pN*um^2 (LpTwist = 1 um)
nFib = 1;
clamp0 = 0;
helicalPitch = 0.07; % rad/s


%% Set up fiber positions
XMP=[0;0;L/2];
X0BC=[0;0;0];
p=0.01;
Tau0BC=[p;0;sqrt(1-p^2)];

%% Numerical parameters
RectangularCollocation = 1; % doing this for twist
TorqBC = 0;
exactRPY = 0;
mu=1;
a=rtrue*exp(3/2)/4;
deltaLocal = 1; % part of the fiber to make ellipsoidal
makeMovie = 1;
updateFrame = 1;
NupsampleHydro = 100;
upsamp=0;
N = 16;
tf= 4;
dt = 1e-3;
t=0;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
initZeroTheta=1;
saveEvery=1;
X_s=repmat(Tau0BC',N,1);
TurnFreq=0;
InitFiberVars;
Lfacs = ones(nFib,1);
Lprime=1;

Npl =1000;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
RplN = barymat(spl,s,b);
D1s=[];
stopcount=floor(tf/dt+1e-5);
Xpts=[];
Thetass=[];
AllXs = Xst;
AllX = Xt;
Allthetas = theta_s;
AllD1mid = D1mid;
AllBCShift = BCShift;
DotProducts=[]; d1=1; d2=1;
XMPSave=[];
if (makeMovie)
f=figure;
%tiledlayout(1,5, 'Padding', 'none', 'TileSpacing', 'compact');
end
D1mids=[];
frameNum=0;
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        frameNum=frameNum+1;
        PtsThisT = reshape(Xt,3,Nx*nFib)';
        %max(abs(sqrt(sum(fTw3.*fTw3,2))))
        if (makeMovie) 
            clf;
            plot3(PtsThisT(:,1),PtsThisT(:,2),PtsThisT(:,3));
            xlim([-0.2 0.2])
            ylim([-0.2 0.2])
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,[])'];
        XMPSave = [XMPSave; XMP'];
        Thetass=[Thetass; Allthetas];
        D1s=[D1s; D1];
    end
    X3 = reshape(Xt,3,[])';
    fwall3 = zeros(Nx,3);
    fwall3(:,3) = -100*(X3(:,3)-L).*(X3(:,3) > L);
    f_ext = reshape(fwall3',[],1);
    n_ext = zeros(Npsi,1);
    BCShift = AllBCShift(3*(N+5)*(iFib-1)+1:3*(N+5)*iFib);
    XMPor0 = XMP;
    if (clamp0)
        XMPor0 = X0BC(:,iFib);
    end
    U0 = zeros(3*Nx,1);
    TemporalIntegrator_Fil;
    Xt = Xp1;
    Xst = Xsp1;
    theta_s = theta_sp1;
    XMP=XMPor0_p1;
    D1((iFib-1)*N+1:iFib*N,:)=D1next;
end