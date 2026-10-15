% Simulation of a particle orbitting a circular trajectory above a wall
% x = (x,y) 
% c(x) = x^2+(y-yc)^2-R^2
% M(x) = [1 0; 0 y]
Dts = [0.5 0.2 0.1 0.05];
MeansByDt = zeros(length(Dts),1);
StdsByDt = zeros(length(Dts),1);
for iDt=1:length(Dts)
rng(0);
doplot=0;
R = 1;
kbT = 1e-1;
yc = 1.1;
EMat = [0 0; 0 0];
nX = 2;
nC = 1;
delta=1e-5;
nW=1; % number to average over for RFD
dt=Dts(iDt)
tf = 2;
nSt = (tf/dt);
tol = 1e-10;
MaxIts = 20;
saveEvery = max(1e-1/dt,1);
nSave = nSt/saveEvery;
nRepeats=floor(1e-6+100*0.25/dt^2);
MeanfStats= zeros(nRepeats,1);
nFail=0;
for kRep=1:nRepeats

nRuns = 1000;
dxh = pi/20;
edges = -pi:dxh:pi;
mps = (edges(1:end-1)+edges(2:end))/2;
PosCounts = zeros(nRuns,length(mps));
fStats = zeros(nRuns,1);
opts=optimoptions(@fsolve,'OptimalityTolerance',tol,...
    'SpecifyObjectiveGradient',true,'Display','off');

for iRun=1:nRuns
% Start at a random place on the circle
tst = 0;%rand*2*pi;
x = [cos(tst); yc+sin(tst)];
NumIts = zeros(nSave,1);
Allx = zeros(nSave,2);


% Plot the circle
if (doplot)
th = (0:1000)*2*pi/1000;
plot(R*cos(th),R*sin(th)+yc,'-k','LineWidth',1.0)
xlim([-1.1*R 1.1*R])
ylim([yc-1.1*R yc+1.1*R])
hold on
PlotAspect
Colors=get(gca,'ColorOrder');
end

for iT=1:nSt
    [xg,newtits,newtfail,matfail] = ...
        AdvanceProjection(x,dt,kbT,EMat,@(x)Mobility(x),@(x) GradMat(x,yc),@(x) c(x,yc,R),...
        MaxIts,tol,nC,0,opts);

    if (doplot)
        PlotPtsDirection(x,xtilde,Colors(3,:),0)
        PlotPtsDirection(xtilde,xg,Colors(3,:),0)
        plot([x(1) xg(1)],[x(2) xg(2)],':o','Color',Colors(1,:),'MarkerFaceColor',Colors(1,:))
    end
    x = xg;

    if (mod(iT,saveEvery)==0)
        index = iT/saveEvery;
        Allx(index,:)=x;
        NumIts(index)=newtits;
    end
    if (newtfail)
        nFail=nFail+1;
    end
end
if (doplot)
    figure(2);
end
%t=atan2(Allx(:,2)-yc,Allx(:,1));
%t=t(end/2+1:end);
tend = atan2(x(2)-yc,x(1));
fStats(iRun)=cos(tend)^2;
%PosCounts(iRun,:)=histcounts(t,edges)/(nSave/2*dxh);
%ItCounts(iRun,:)=histcounts(NumIts,0:MaxIts)/(nSave/2);
end
MeanfStats(kRep)=mean(fStats);
end
MeansByDt(iDt) = mean(MeanfStats)
StdsByDt(iDt) = 2*std(MeanfStats)/sqrt(nRepeats)
end
return
MC = mean(PosCounts);
SC = 2*std(PosCounts)/sqrt(nRuns);
Colors=get(gca,'ColorOrder');
cIndex=1;
fill([mps, fliplr(mps)], [MC-SC, fliplr(MC+SC)],...
    Colors(cIndex,:), 'FaceAlpha', 0.2, 'linestyle', 'none');
hold on
plot(mps,MC,'-','Color',Colors(cIndex,:),'LineWidth',2)
hold on
dx=1e-3;
xv=-R:dx:R;
pdfEn = exp(-1/2*xv.^2/kbT);
pdfEn=pdfEn/sum(pdfEn*dx);
%plot(xv,pdfEn,'-k')
plot([-pi pi],1/(2*pi)*[1 1],'-k')

function cd = c(x,yc,R)
    cd = x(1).^2+(x(2)-yc).^2-R.^2;
end

function C = GradMat(x,yc)
    C = [2*x(1) 2*(x(2)-yc)];
end

function H = HessMat(x)
    H = [2 0; 0 2];
end

function M = Mobility(x)
    M = [1 0; 0 x(2)];
end

function PlotPtsDirection(x1,x2,color,fillface)
    if (~fillface)
        plot([x1(1) x2(1)],[x1(2) x2(2)],':o','Color',color)
    else
        plot([x1(1) x2(1)],[x1(2) x2(2)],':o','Color',color,'MarkerFaceColor',color)
    end
    mp = (x1+x2)/2;
    dir = (x2-x1);
    quiver(x1(1),x1(2),dir(1),dir(2),'Color',color,'LineWidth',1)
end
