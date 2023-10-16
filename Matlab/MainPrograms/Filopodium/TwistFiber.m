% Twisting cross-linked bundle
close all;
addpath('../../functions-solvers')

%% Physical parameters
poly = 0;
L = 2;   % microns
rtrue = 4e-3; % 4 nm radius
eps=rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
twmod = 10*kbT; % pN*um^2 (LpTwist = 1 um)
nFib = 1;
clamp0 = 0; % For the X BCs
TwistTorq = 1; % Torque at s=L or rotation
helicalPitch = 0.07; % rad/s


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
XMP=[0;0;L/2];
tf= 10;
dt = 1e-3;
t=0;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
q=1; 
p = 0.1;
X_s = [p*cos(q*s.^3 .* (s-L).^3) p*sin(q*s.^3.*(s - L).^3) sqrt(1-p^2)*ones(N,1)];
%X_s = [ones(N,1) zeros(N,2)];
D = diffmat(N, 1, [0 L], 'chebkind1');
saveEvery=10;
TurnFreq=0;
Lfacs = ones(nFib,1);
Lprime=zeros(nFib,1);
initZeroTheta=1;
InitFiberVars;
RecomputeDiscMats;
if (TwistTorq)
    nTurns = 2;
    PsiBCL= 2*pi/L*nTurns;
    theta_s=PsiBCL*ones(Npsi,1);
end

Npl =1000;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
[sFr,~,~]=chebpts(100,[0 L]);
RplN = barymat(sFr,s,b);
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
velmax=zeros(stopcount,1);
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        frameNum=frameNum+1;
        PtsThisT = RplNp1*reshape(Xt,3,Nx*nFib)';
        %max(abs(sqrt(sum(fTw3.*fTw3,2))))
        if (makeMovie) 
            clf;
            plot3(PtsThisT(:,1),PtsThisT(:,2),PtsThisT(:,3));
            hold on
            FramePts = RplN*RNp1ToN*reshape(Xt,3,Nx*nFib)';
            FrameToPlot = RplN*D1;
            quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
                FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),'LineWidth',1.0);
            PlotAspect
            %xlim([-0.2 0.2])
            %ylim([-0.2 0.2])
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
    velmax(count+1)=max(abs(Xt-Xp1)/dt);
    Xt = Xp1;
    Xst = Xsp1;
    %theta_s = theta_sp1; DO NOT UPDATE theta (assume it's at steady state)
    XMP=XMPor0_p1;
    D1((iFib-1)*N+1:iFib*N,:)=D1next;
end