close all;
addpath(genpath('../'))
deltaP = 0.003; 
UpdateFil = 1;
nTrial = 1;
Kster = 0;
KCL = 10;
%for iF=1:length(Fmems)
%for iTrial=1:nTrial
% Polymerization
kPolyOn = 100; % 1/sec
rng(1);

% Temporal integration
tf = 100;
dt = 1e-3;
impcoeff = 1;

% Membrane discretization
Fmem = 0;%Fmems(iF);
mu = 1;
Lm = 1;
Kh = 1;
Kc = 0.2;
Mem = InitializeMembraneDisc(mu,Lm,Kc,Kh,dt);

% Fiber discretization
nFib = 3;
L = 1;   % microns
N = 12;
Nu = 41;
rtrue = 4e-3; % 4 nm radius
kbT = 4.1e-3;
lp = 17;
Eb = 10;%lp*kbT; % pN*um^2 (Lp=17 um)
makeMovie = 1;
Tau0BC = [0;0;1];
TrkLoc=0;
XTrk0=[Lm/2;Lm/2;-L-deltaP];
XTrk=zeros(3,nFib);
rCirc = 0.02;
clamp=1;
uPts = [];
% Initialize fibers
for j = 1:nFib
    t=2*pi/nFib*(j-1);
    XTrk(:,j) = XTrk0+rCirc*[cos(t); sin(t); 0];
    Discr(j) = InitializeDiscretization(repmat(Tau0BC',N,1),...
        XTrk(:,j),TrkLoc,L,Eb,rtrue,mu,clamp,Nu);
    uPts = [uPts; Discr(j).Runi*reshape(Discr(j).Xt,3,[])'];
end
% Initialize CLs
fibsToLink=[];%[1 2; 2 3; 3 1];
nLinks = size(fibsToLink,1);
links=[];
rCLs = [];
LinkPts = Nu;
for iLp = LinkPts
    for iPair = 1:nLinks
        pt1 = (fibsToLink(iPair,1)-1)*Nu+iLp;
        pt2 = (fibsToLink(iPair,2)-1)*Nu+iLp;
        links = [links; pt1 pt2 0 0 0];
        rCLs = [rCLs;norm(uPts(pt1,:)-uPts(pt2,:))];
    end
end

stopcount=floor(tf/dt+1e-5);
saveEvery=floor(1e-1/dt+1e-10);
Xpts=[];
FibLens=[];
AllMemPts=[];
MemEnergy=[];
FibEnergy=[];
TotalFibEnergy=[];
if (makeMovie)
    f=figure;
    frameNum=0;
end

% Equilibrate membrane
minh=[];
stcteq = floor(10/dt+1e-5);
for count=0:stcteq
    Fpush = Fmem*ones(Mem.M^2,1);
    Mem = UpdateMembrane(Mem,Fpush,kbT,dt);
    minh=[minh;min(Mem.hmem)];
end
mempts=zeros(stopcount,1);
filH = zeros(stopcount,1);


%% Computations
for count=0:stopcount
    t=count*dt;
    Xt = [];
    Lens = [];
    for iFib=1:nFib
        Xt = [Xt;Discr(iFib).Xt];
        Lens = [Lens;Discr(iFib).L];
    end
    X3All = reshape(Xt,3,[])';
    if (mod(count,saveEvery)==0)
        %t
        if (makeMovie)
            clf;
            %nexttile
            frameNum=frameNum+1;
            for iFib=1:nFib
                Rpl = Discr(iFib).RplNp1;
                Xv = Rpl*reshape(Discr(iFib).Xt,3,[])';
                plot3(Xv(:,1),Xv(:,2),Xv(:,3));
                hold on
            end
            [~,X1stars,X2stars] = getCLforceEn(links,X3All,Discr(1).Runi,...
                KCL, rCLs,0,0);
            for iLink=1:size(links,1)
                linkPts = [X1stars(iLink,:); X2stars(iLink,:)];
                plot3(linkPts(:,1),linkPts(:,2),linkPts(:,3),':ko');
            end
            title(sprintf('$t=$ %1.2f',(frameNum-1)*saveEvery*dt))
            zlim([-1 0.5])
            PlotAspect
            hold on
            xbd = xlim;
            ybd = ylim;
            xCt = floor(xbd(1)/Lm):ceil(xbd(2)/Lm);
            yCt = floor(ybd(1)/Lm):ceil(ybd(2)/Lm);
            PlotMem(Mem,xCt(1:end-1),yCt(1:end-1));
            view([-45 0])
            xbd = xlim;
            ybd = ylim;
            if (range(xbd)<Lm)
                xlim([0 Lm])
            end
            if (range(ybd)<Lm)
                ylim([0 Lm])
            end
            PlotAspect
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;reshape(Xt,3,[])'];
        AllMemPts =[AllMemPts Mem.hmem];
        FibLens=[FibLens Lens];
        % Compute energy stored in the membrane
        MemEnergy = [MemEnergy computeMembraneEnergy(Mem)];
        FibEn=0;
        for iFib=1:nFib
            Xthis = Discr(iFib).Xt;
            FibEn=FibEn+Xthis'*Discr(iFib).BendingEnergyMatrix_Np1*Xthis;
        end
        FibEnergy = [FibEnergy FibEn];

    end

    % Polymerize the filaments
    Fiberxypts=X3All(:,1:2);
    for iFib=1:nFib
        Discr(iFib) = PolymerizeClampedFib(Discr(iFib),Mem,kPolyOn*dt,deltaP);
    end

    % Compute steric forces 
    Xt = [];
    for iFib=1:nFib
        Xt = [Xt;Discr(iFib).Xt];
    end
    X3All = reshape(Xt,3,[])';
    Fiberxypts=X3All(:,1:2);
    MemPtsAtX = InterpolatehNUFFT(Fiberxypts,Mem);
    dh = (MemPtsAtX - X3All(:,end)); % Should be > 0
    stForce = Kster*(1-dh/deltaP).*(dh<deltaP); % This is key; need to think about it
    
    % Evolve the membrane
    Fpush = Fmem*ones(Mem.M^2,1);
    Fgmem = SpreadStericForceToMem(Fiberxypts,stForce,Mem);
    Mem = UpdateMembrane(Mem,Fpush+Fgmem,kbT,dt);
    minh=[minh;min(Mem.hmem)];

    % Evolve fibers
    if (UpdateFil)
        % Compute cross linking forces
        [CLf,X1stars,X2stars] = getCLforceEn(links,X3All,Discr(1).Runi, ...
            KCL,rCLs,0,0);
        F_Ext = reshape(CLf',[],1);
        F_Ext(3:3:end) = F_Ext(3:3:end) - stForce;
        Nx = Discr(1).Nx;
        for iFib=1:nFib
            RandomNums = randn(9*(N+1),1);
            Discr(iFib)=EvolveClampedFil(Discr(iFib),kbT,dt,...
                impcoeff,RandomNums,F_Ext((iFib-1)*3*Nx+1:3*Nx*iFib));
        end
    end
end