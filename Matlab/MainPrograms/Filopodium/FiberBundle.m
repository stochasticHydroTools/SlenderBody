% Twisting cross-linked bundle
%close all;
addpath(genpath('../../../Matlab'))
addpath(genpath('/home/om759/Documents/MATLAB/'))

%% Physical parameters
KCL = 100;
updateCL=1;
startRandomCL=0;
konCL = 0.2;
koffCL = 10;
Ktorq = 0;
MembraneForce = 0;
ellCL = 0.1; % um = 100 nm (alpha actinin)
MotorLengthFrac = 1/2; % fraction for motors 
MotorCircleFrac = 1/4; % motors on the outer 1/4 of the circle
fmot0 = 50; % pN/um (assuming 5 pN per motor x 10 motors/micron)
fmotDwn = 0; 
AdjustForceTangential = 0;
poly = 1;
KPoly = 2; % deterministic coeff in units of 1/s
SDPoly = 0.05; % standard deviation of length rate
Poly0 = 0.5; % Base Rate
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
MotorTorq = 0*fmot0*rtrue;
eps = rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
twmod = 10*kbT; % pN*um^2 (LpTwist = 10 um)
NFromFormin = 0.1;
nTurnsPerMicron = (NFromFormin/twmod)*L/(2*pi);
doSterics = 1;
clamp0 = 1;
TwistTorq=1;
nTrial=1;
BigBundle=1;
for iTrial=1:nTrial
rng(iTrial);

%% Set up fiber positions
if (BigBundle)
    nCircles = 4;
    nPerCircle = [1 3 6 12];
    rEachCircle = [0 ellCL 2*ellCL 3*ellCL];
    fibsToLink =[1 2; 1 3; 1 4; 2 3; 3 4; 4 2; ...
        5 2; 6 2; 7 3; 8 3; 9 4; 10 4; 5 6; 6 7; 7 8; 8 9; 9 10; 10 5; ...
        11 5; 12 5; 13 6; 14 6; 15 7; 16 7; 17 8; 18 8; 19 9; 20 9; 21 10; 22 10; ...
        11 12; 12 13; 13 14; 14 15; 15 16; 16 17; 17 18; 18 19; 19 20; 20 21; 21 22; 22 11];
else
    nCircles = 2;
    nPerCircle = [4 1];
    fibsToLink = [1 2; 2 3; 3 4; 1 4; 1 5; 2 5; 3 5; 4 5];
    rEachCircle = [ellCL 0];
end
nFib = sum(nPerCircle);
RFilo = max(rEachCircle);
XMP=[];
X0BC=[];
Tau0BC=[];
for iCircle=1:nCircles
    theta = (0:nPerCircle(iCircle)-1)/nPerCircle(iCircle)*2*pi;
    MPsToAdd=[rEachCircle(iCircle)*cos(theta); rEachCircle(iCircle)*sin(theta); 0*theta];
    XMP =[XMP MPsToAdd];
    ZerosToAdd=[rEachCircle(iCircle)*cos(theta); rEachCircle(iCircle)*sin(theta); 0*theta-1/2];
    X0BC = [X0BC ZerosToAdd];
    if (rEachCircle(iCircle)==0)
        perturb=0.01;
        Tau0BC = [Tau0BC [0;perturb;1]/sqrt(perturb^2+1^2)];
    else
        Tau0BC = [Tau0BC repmat([0;0;1],1,nPerCircle(iCircle))];
    end
    TurnFreq = 0;
end
if (clamp0)
    XMP=X0BC;
end

%% Numerical parameters
RectangularCollocation = 1; % doing this for twist
sigPoly = SDPoly*sqrt(2*KPoly);
TorqBC = 0;
exactRPY = 0;
upsamp=0;
mu=1;
a=rtrue*exp(3/2)/4;
deltaLocal = 1; % part of the fiber to make ellipsoidal
makeMovie =0;
updateFrame = 1;
N = 20;
dt = 1e-4;
tf = 2;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
initZeroTheta=0;
saveEvery=max(floor(1e-2/dt),1);
X_s=[];
%Lfacs = 1/L*ones(nFib,1);
Lfacs = ones(nFib,1);
for iFib=1:nFib
    %X_s=[X_s; Lfacs(iFib)*repmat(Tau0BC(:,iFib)',N,1)];
    X_s=[X_s; repmat(Tau0BC(:,iFib)',N,1)];
end
Lprime = zeros(nFib,1);
if (poly)
    Lprime=Poly0*ones(nFib,1)+SDPoly*randn(nFib,1);
end
InitFiberVars;
Mrr = eye(Npsi)/(4*pi*mu*rtrue^2);
Mrr_Psip2 = eye(Npsi+2)/(4*pi*mu*rtrue^2);

%% Initialize CLs
Nu = 41;
su = (0:Nu-1)'*L/(Nu-1);
Runi = barymat(su,sNp1,bNp1);
Nusteric = 101;
suSteric = (0:Nusteric-1)'*L/(Nusteric-1);
RuniSteric = barymat(suSteric,sNp1,bNp1);
RUniTau = barymat(su,s,b);
%links = [CLspot*(Nu-1)+1 CLspot*(Nu-1)+1+Nu 0 0 0];
if (startRandomCL)
X3All = reshape(Xt,3,[])';
[links,rCLs] = updateDynamicCLs([],[],Runi,X3All,nFib,0,KCL,ellCL,kbT,konCL,koffCL,1);
else
links=[];
LinkPts = Nu;
[nPairs,~]=size(fibsToLink);
for iLp = LinkPts
for iPair = 1:nPairs
    pt1 = (fibsToLink(iPair,1)-1)*Nu+iLp;
    pt2 = (fibsToLink(iPair,2)-1)*Nu+iLp;
    links = [links; pt1 pt2 0 0 0];
end
end
end
[nLinks,~]=size(links);
rCLs = zeros(nLinks,1);
[~,X1stars,X2stars] = getCLforceEn(links,reshape(Xt,3,[])',Runi,KCL, rCLs,0,0);
diff = X1stars - X2stars;
rCLs = sqrt(sum(diff.*diff,2));
%[CLCoords_1,CLCoords_2] = addLinkMaterialVector(links,nLinks,X1stars,X2stars,X_s,D1,RUniTau);

%% Initialization 
Npl=1000;
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
    %tiledlayout(1,4, 'Padding', 'none', 'TileSpacing', 'compact');
end
D1mids=[];
AllLinks=[];
AllLfacs=[];
nLinksT=[];
frameNum=0;
Lprimes=[];
tmovies = [0.02 0.25 0.6 1];
%% Computations
for count=0:stopcount
    t=count*dt;
    if (poly)
        Lprime=Lprime+dt*KPoly*(Poly0-Lprime)+sigPoly*sqrt(dt)*randn(nFib,1);
        Lprime(Lfacs < 0.1 & Lprime < 0)=0; % stop them from getting too short
        Lprime(Lfacs > 2 & Lprime > 0)=0; % stop them from getting too long
    end
    if (mod(count,saveEvery)==0)
        Lprimes=[Lprimes Lprime];
        PtsThisT = reshape(AllX,3,Nx*nFib)';
        %max(abs(sqrt(sum(fTw3.*fTw3,2))))
        if (makeMovie)% && abs(t-tmovies(frameNum+1)) < dt/2) 
            clf;
            %nexttile
            frameNum=frameNum+1;
            %subplot(3,3,frameNum)
            for iFib=1:nFib
                fibInds = (iFib-1)*Nx+1:iFib*Nx;
                plot3(RplNp1*PtsThisT(fibInds,1),RplNp1*PtsThisT(fibInds,2),...
                    RplNp1*PtsThisT(fibInds,3));
                hold on
                FramePts = RNp1ToN*PtsThisT(fibInds,:);
                FrameToPlot = D1((iFib-1)*N+1:iFib*N,:);
                %set(gca,'ColorOrderIndex',iFib)
                %quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
                %    FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),0.25,'LineWidth',1.0);
            end
            [nLinks,~]=size(links);
            X3All = reshape(AllX,3,[])';
            [~,X1stars,X2stars] = getCLforceEn(links,X3All,Runi,KCL, rCLs,0,0);
            for iLink=1:nLinks
                linkPts = [X1stars(iLink,:); X2stars(iLink,:)];
                plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko');
            end
            %xlim([-0.15 0.15])
            %ylim([-0.15 0.15])
            %zlim([-0.5 0.5])
            %view([ -59.1403    5.5726])
            view([210.7240   34.8854])
            PlotAspect
            %title(strcat('$t=$',num2str(tmovies(frameNum))))
            title(strcat('$t=$',num2str((frameNum-1)*saveEvery*dt)))
            movieframes(frameNum)=getframe(f);
        end
        [nLinks,~]=size(links);
        nLinksT=[nLinksT;nLinks];
        AllLinks=[AllLinks;links];
        Xpts=[Xpts;PtsThisT];
        Thetass=[Thetass; Allthetas];
        D1s=[D1s; D1];
        AllLfacs=[AllLfacs; Lfacs'];
     end
    % Evolve system
    FiloExtFT;
    for iFib=1:nFib
        Xst = AllXs((iFib-1)*3*N+1:iFib*3*N);
        Xt = AllX((iFib-1)*3*Nx+1:iFib*3*Nx);
        n_ext = n_extAll((iFib-1)*Npsi+1:iFib*Npsi);
        f_ext = reshape(fextAll((iFib-1)*Nx+1:iFib*Nx,:)',[],1);
        theta_s = Allthetas((iFib-1)*Npsi+1:iFib*Npsi); 
        D1mid = AllD1mid(iFib,:);
        BCShift = AllBCShift(3*(N+5)*(iFib-1)+1:3*(N+5)*iFib);
        XMPor0 = XMP(:,iFib);
        U0 = U0All((iFib-1)*3*Nx+1:iFib*3*Nx);
        TemporalIntegrator_Fil;
        AllX((iFib-1)*3*Nx+1:iFib*3*Nx) = Xp1;
        AllXs((iFib-1)*3*N+1:iFib*3*N) = Xsp1;
        Allthetas((iFib-1)*Npsi+1:iFib*Npsi) = theta_sp1;
        AllD1mid(iFib,:) = D1mid;
        XMP(:,iFib)=XMPor0_p1;
        D1((iFib-1)*N+1:iFib*N,:)=D1next;
    end
end
AllnLinks{iTrial}=nLinksT;
AllLinksConfs{iTrial}=AllLinks;
AllPositions{iTrial}=Xpts;
AllAngles{iTrial}=Thetass;
AllMatFrames{iTrial}=D1s;
AllExtensions{iTrial}=AllLfacs;
%save(strcat('DownwardControl.mat'))
end
%exit;
