% Simulation of a particle orbitting a circular trajectory above a wall
% x = (x,y) 
% c(x) = x^2+(y-yc)^2-R^2
% M(x) = [1 0; 0 y]
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
dt=1e-2;
tf = 400;
nSt = (tf/dt);
saveEvery = max(1e-1/dt,1);
nSave = nSt/saveEvery;

nRuns = 100;
dxh = pi/20;
edges = -pi:dxh:pi;
mps = (edges(1:end-1)+edges(2:end))/2;
PosCounts = zeros(nRuns,length(mps));

for iRun=1:nRuns
% Start at a random place on the circle
tst = rand*2*pi;
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
M = Mobility(x);
Mhalf = chol(M)';
C = GradMat(x,yc);
prefac = M*C'*(C*M*C')^(-1);
GradU = EMat*x;

if (nW>0)
divM = zeros(nX,1);
divMC = zeros(nX,1);
for iP=1:nW
    w1 = randn(nX,1);
    xr = x + delta*w1;
    Mu = Mobility(xr);
    divM = divM + 1/(nW*delta)*(Mu-M)*w1;

    w2 = randn(nC,1);
    xc = x + delta*prefac*w2;
    Mc = Mobility(xc);
    Cc = GradMat(xc,yc);
    divMC = divMC + 1/(nW*delta)*(Mc*Cc'-M*C')*w2;
end
divMtru = divM;
divMctru = divMC;
else
divMtru = [0;1];
divMctru = 1/(x(1)^2+x(2)*(x(2)-yc)^2)*[x(1);x(2)*(x(2)-yc)*(2*x(2)-yc)];
end

% Take unconstrained step
W = randn(nX,1);
xtilde = x - dt*M*GradU + dt*kbT*divMtru+sqrt(2*dt*kbT)*Mhalf*W;

% Half step and evaluate projection matrices
xHalf = x + sqrt(kbT*dt/2)*Mhalf*W;
Mhalf = Mobility(xHalf);
Chalf = GradMat(xHalf,yc);

% Nonlinear system for the projection
% x - xtilde + Mhalf*Chalf'*lambda = 0 
% c(x) = 0
% Newton solve
xg = x;
lam = 0;
er=1;
tol = 1e-10;
MaxIts = 20;
Allresids = zeros(MaxIts,1);
for it=1:MaxIts
    % Compute the gradient and Hessian at x
    C = GradMat(xg,yc);
    J = [eye(nX) -Mhalf*Chalf'; C zeros(nC)];
    resid = [(xg-xtilde) - Mhalf *Chalf'*lam; c(xg,yc,R)];
    er=norm(resid);
    Allresids(it)=er;
    if (er > tol)
        newsol = [xg;lam] - J \ resid;
        xg = newsol(1:nX);
        lam = newsol(nX+1:end);
    else
        break
    end
end
if (it>=MaxIts)
    keyboard
end
if (doplot)
PlotPtsDirection(x,xtilde,Colors(3,:),0)
PlotPtsDirection(xtilde,xg,Colors(3,:),0)
plot([x(1) xg(1)],[x(2) xg(2)],':o','Color',Colors(1,:),'MarkerFaceColor',Colors(1,:))
end
x = xg;

if (mod(iT,saveEvery)==0)
    index = iT/saveEvery;
    Allx(index,:)=x;
    NumIts(index)=it;
end
end
if (doplot)
    figure(2);
end
t=atan2(Allx(:,2)-yc,Allx(:,1));
t=t(end/2+1:end);
PosCounts(iRun,:)=histcounts(t,edges)/(nSave/2*dxh);
ItCounts(iRun,:)=histcounts(NumIts,0:MaxIts)/(nSave/2);
end
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
