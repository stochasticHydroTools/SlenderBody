% Twisting cross-linked bundle
close all;
addpath('../../functions-solvers')

%% Physical parameters
KCL = 100;
Ktorq = 0;
ellCL = 0.1; % um = 50 nm (alpha actinin)
forceFrac = 1; 
fmot0 = 10; % pN/um
poly = 0;
doSterics = 0;
MotorCircleFrac = 1;
MotorLengthFrac=1;
RFilo = 1;
L = 1;   % microns
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
twmod = kbT; % pN*um^2 (LpTwist = 1 um)
nFib = 5;
clamp0 = 1;
updateCL=0;
helicalPitch = 0.07; % rad/s
TwistTorq=1;
nTrial=1;
for iTrial=1:nTrial
rng(iTrial);


%% Set up fiber positions
%nCircles = 3;
%nPerCircle = [8 4 1];
%fibsToLink = [1 2; 2 3; 3 4; 4 1; 1 5; 2 5; 3 5; 4 5]+8;
%fibsToLink = [fibsToLink; 1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 1;...
%    1 9; 2 9; 2 10; 3 10; 4 10; 4 11; 5 11; 6 11; 6 12; 7 12; 8 12; 8 9];
%rEachCircle = [2*ellCL ellCL 0];
nCircles = 2;
nPerCircle = [4 1];
fibsToLink = [1 2; 2 3; 3 4; 4 1; 1 5; 2 5; 3 5; 4 5];
rEachCircle = [ellCL 0];
RMotorBounds = [0.9*max(rEachCircle) 1.1*max(rEachCircle)];
BoundingCircle = 1.1*max(rEachCircle);
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
TorqBC = 0;
exactRPY = 0;
upsamp=0;
mu=1;
a=rtrue*exp(3/2)/4;
deltaLocal = 1; % part of the fiber to make ellipsoidal
makeMovie = 1;
updateFrame = 1;
NupsampleHydro = 100;
N = 24;
dt = 1e-3;
tf= 0.3;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');
initZeroTheta=1;
saveEvery=max(floor(1e-2/dt),1);
X_s=[];
Lfacs = 1/L*ones(nFib,1);
for iFib=1:nFib
    X_s=[X_s; Lfacs(iFib)*repmat(Tau0BC(:,iFib)',N,1)];
end
InitFiberVars;
Mrr = eye(Npsi)/(4*pi*mu*rtrue^2);
Mrr_Psip2 = eye(Npsi+2)/(4*pi*mu*rtrue^2);
Lprime = zeros(nFib,1);


%% Initialize CLs
Nu = 41;
su = (0:Nu-1)'*L/(Nu-1);
Runi = barymat(su,sNp1,bNp1);
Nusteric = 101;
suSteric = (0:Nusteric-1)'*L/(Nusteric-1);
RuniSteric = barymat(suSteric,sNp1,bNp1);
RUniTau = barymat(su,s,b);
%links = [CLspot*(Nu-1)+1 CLspot*(Nu-1)+1+Nu 0 0 0];
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
%tiledlayout(1,5, 'Padding', 'none', 'TileSpacing', 'compact');
end
D1mids=[];
AllAngles = [];
AllZVals = [];
AllMeanCurvs=[];
frameNum=0;
%% Computations
for count=0:stopcount
    t=count*dt;
    if (poly)
        %Lprime=Lprime+sqrt(2*kbT*dt)*randn(nFib,1);
        Lprime(Lfacs < 0.1 & Lprime < 0)=0; % stop them from getting too short
        Lprime(Lfacs > 2 & Lprime > 0)=0; % stop them from getting too long
    end
    tmovies = [0 0.1 0.2 0.4 0.7 1 1.1 1.5 2];
    if (mod(count,saveEvery)==0)
        PtsThisT = reshape(AllX,3,Nx*nFib)';
        %max(abs(sqrt(sum(fTw3.*fTw3,2))))
        if (makeMovie && abs(t-tmovies(frameNum+1)) < dt/2) 
            %clf;
            frameNum=frameNum+1;
            subplot(3,3,frameNum)
            for iFib=1:nFib
                fibInds = (iFib-1)*Nx+1:iFib*Nx;
                plot3(RplNp1*PtsThisT(fibInds,1),RplNp1*PtsThisT(fibInds,2),...
                    RplNp1*PtsThisT(fibInds,3));
                hold on
                FramePts = RNp1ToN*PtsThisT(fibInds,:);
                FrameToPlot = D1((iFib-1)*N+1:iFib*N,:);
                %quiver3(FramePts(:,1),FramePts(:,2),FramePts(:,3),...
                %    FrameToPlot(:,1),FrameToPlot(:,2),FrameToPlot(:,3),0.5,'LineWidth',1.0);
            end
            [nLinks,~]=size(links);
            X3All = reshape(AllX,3,[])';
            [~,X1stars,X2stars] = getCLforceEn(links,X3All,Runi,KCL, rCLs,0,0);
            for iLink=1:nLinks
                linkPts = [X1stars(iLink,:); X2stars(iLink,:)];
                plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko');
            end
            %xlim([-0.25 0.5])
            %ylim([-0.4 0.5])
            %zlim([-0.5 0.6])
            PlotAspect
            view([ -59.1403    5.5726])
            %movieframes(frameNum)=getframe(f);
            title(strcat('$t=$',num2str(tmovies(frameNum))))
        end
        Xpts=[Xpts;PtsThisT];
        Thetass=[Thetass; Allthetas];
        D1s=[D1s; D1];
        % Information about the bundle as a whole
        ThisCurv=zeros(1,nFib);
        ThisAngle = zeros(1,nFib+1);
        ThisZVal = zeros(1,nFib);
        meanEP = zeros(1,3);
        meanSP = zeros(1,3);
        for iFib=1:nFib
            fibInds = (iFib-1)*Nx+1:iFib*Nx;
            XFib = PtsThisT(fibInds,:);
            CurvMatrix = 1/Lfacs(iFib)^2*DNp1^2;
            WTilde_Np1_L = Lfacs(iFib)*WTilde_Np1;
            EPthis = barymat(L,sNp1,bNp1)*XFib;
            meanEP=meanEP+EPthis/nFib;
            Zerothis = barymat(0,sNp1,bNp1)*XFib;
            meanSP=meanSP+Zerothis/nFib;
            [thistheta,thisr] = cart2pol(EPthis(1),EPthis(2));
            FibCurv = reshape((CurvMatrix*XFib)',[],1);
            ThisCurv(iFib)=sqrt(FibCurv'*WTilde_Np1_L*FibCurv);
            % Compute angle with initial tangent axis
            ThisAngle(iFib)=acos((EPthis(3)-Zerothis(3))/norm(EPthis-Zerothis));
            ThisZVal(iFib)=EPthis(3);
        end
        ThisAngle(nFib+1)=acos((meanEP(3)-meanSP(3))/norm(meanEP-meanSP));
        AllAngles = [AllAngles; ThisAngle];
        AllZVals = [AllZVals; ThisZVal];
        AllMeanCurvs=[AllMeanCurvs; ThisCurv];
    end
    % Evolve system
    FiloExtFT;
    for iFib=1:nFib
        Xst = AllXs((iFib-1)*3*N+1:iFib*3*N);
        Xt = AllX((iFib-1)*3*Nx+1:iFib*3*Nx);
        n_ext = n_extAll((iFib-1)*Npsi+1:iFib*Npsi);
        PsiBC0 = -barymat(0,sPsi,bPsi)*Mrr*n_ext;
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
%MeanAngles(:,iTrial) = AllAngles(:,end);
%MeanZVals(:,iTrial) = mean(AllZVals')';
%MeanCurvsFibs(:,iTrial)= mean(AllMeanCurvs')';
end
% nGroups = 2;
% nPerGroup = nTrial/nGroups;
% for iGroup=1:nGroups
%     MeanMeanAngles(:,iGroup)=mean(MeanAngles(:,(iGroup-1)*nPerGroup+1:iGroup*nPerGroup)')';
%     MeanMeanZVals(:,iGroup) = mean(MeanZVals(:,(iGroup-1)*nPerGroup+1:iGroup*nPerGroup)')';
%     MeanMeanCurves(:,iGroup) = mean(MeanCurvsFibs(:,(iGroup-1)*nPerGroup+1:iGroup*nPerGroup)')';
% end
% AllMeanMeanAngles{iSet}=MeanMeanAngles;
% AllMeanMeanZVals{iSet}=MeanMeanZVals;
% AllMeanMeanCurvs{iSet}=MeanMeanCurves;
% iSet=iSet+1;