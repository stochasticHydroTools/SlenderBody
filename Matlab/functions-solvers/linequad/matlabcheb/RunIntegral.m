addpath('configs')
addpath('recursions')
Nfib=100;
Ntarg = 100;
dist = 0.12;
sperss=zeros(Nfib,Ntarg);
dir16ers=zeros(Nfib,Ntarg);
dir32ers=zeros(Nfib,Ntarg);
unid16s=zeros(Nfib,Ntarg);
unid32s=zeros(Nfib,Ntarg);
groots = zeros(Nfib,Ntarg);
specneeded = zeros(Nfib,Ntarg);
ctures=zeros(Nfib,16);
load('configs/targsOnFibs.mat');
tonFib=t;
load('configs/randomvecs.mat');
randv = v;
for iFib=1:100
   load(strcat('configs/Fib_',num2str(iFib),'.mat'));
   cFib = c;
   [spers,d16ers,d32ers,unid16, unid32,gs, specneed, C]=demo_long_fiber_Cheb(cFib,tonFib,randv,dist);
    sperss(iFib,:)=spers; dir16ers(iFib,:)=d16ers; dir32ers(iFib,:)=d32ers; 
    unid16s(iFib,:) = unid16; unid32s(iFib,:) = unid32; groots(iFib,:)=gs; ctures(iFib,:)=C;
    specneeded(iFib,:) = specneed;
    % Test fibers
    K = 15;   % max Chebyshev mode in derivatives
    k = 0:K-1;
    xpf = @(t) c(1,:)*(cos(k'.*acos(2*t(:)/Lf'-1)));
    ypf = @(t) c(2,:)*(cos(k'.*acos(2*t(:)/Lf'-1)));
    zpf = @(t) c(3,:)*(cos(k'.*acos(2*t(:)/Lf'-1)));
    s = @(t) sqrt(xpf(t(:)').^2+ypf(t(:)').^2+zpf(t(:)').^2);
    % Normalize
    Lf=2;
    xp = @(t) xpf(t)./(s(t));
    yp = @(t) ypf(t)./(s(t));
    zp = @(t) zpf(t)./(s(t));
    N=16;
    tj = chebpts(N,[0 Lf],1);
    x = pos(tj,xp);
    y = pos(tj,yp);
    z = pos(tj,zp);
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    tj1=cos(th);
    L = (cos((0:N-1).*th));
    hats = L \ [x y z];
    if (sum(sum(abs(hats(3:end,:)) > exp(-0.55*[k(3:end) K])')) > 0)
        keyboard
    end
%     hatsmax = [hatsmax max(abs(hats(end-1,end,:)))];
end

function x = pos(t,xp)
    % Integrate to get position
    opts = odeset('RelTol',5e-14,'AbsTol',5e-14);
    [~,x] = ode45(@(t,x) xp(t),[0 t(:)'],0,opts);
    if (length(t) > 1)
        x=x(2:end);
    else 
        x=x(end);
    end
end
