% Single fluctuating clamped filament
close all;
nFib = 1;
L = 2;   % microns
N = 20;
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
mu = 1;
a = rtrue*exp(3/2)/4;
upsamp=0; % For hydro
makeMovie =1;
dt = 0.25;
tf = 200;
Tau0BC = [0;1;0];
XMP=[0;L/2;0];
X_s=repmat(Tau0BC(:,iFib)',N,1);
[s,w,b] = chebpts(N, [0 L], 2); 
InitializationNoTwist;

%% Initialization 
stopcount=floor(tf/dt+1e-5);
Xpts=[];
AllXs = Xst;
AllX = Xt;
if (makeMovie)
    f=figure;
    frameNum=0;
end
% Temporary: M = I 
M = eye(3*Nx)/(8*pi*mu);
MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
MWsymHalf = real(MWsym^(1/2));

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        PtsThisT = reshape(AllX,3,Nx*nFib)';
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            plot3(RplNp1*PtsThisT(:,1),RplNp1*PtsThisT(:,2),...
                RplNp1*PtsThisT(:,3));
            title(strcat('$t=$',num2str((frameNum-1)*saveEvery*dt)))
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
    end
    % Evolve system
    % Obtain Brownian velocity
    g = randn(3*(N+1),1);
    %g = load(strcat('RandVec1_',num2str(count+1),'.txt'));
    %g = g(3*(N+1)*(iFib-1)+1:3*(N+1)*iFib);
    RandomVel = sqrt(2*kbT/dt)*MWsymHalf*g;
    RandomVelSpl = RandomVel;
    OmegaTilde = cross(reshape(Xst,3,N)',RNp1ToN*DNp1*reshape(RandomVelSpl,3,[])');
    XMPTilde = XMP+dt/2*BMNp1*RandomVelSpl;
    Xstilde = rotateTau(Xs3,OmegaTilde,dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    
end
% norm(Xt(1:3)'-Xpts(Nx+1,:))
% Xs0=barymat(0,s,b)*Xs3;
% Xs0=Xs0/norm(Xs0);
% norm(Xs0-[0 1 0])
