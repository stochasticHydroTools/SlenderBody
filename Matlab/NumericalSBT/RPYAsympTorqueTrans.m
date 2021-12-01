% RPY asymptotics on translational velocity due to torque 
% This is asymptotics on 8*pi*mu*U
L = 2;
as= [0.08 0.04 0.02 0.01];

for ia=4
a = as(ia);
mu = 1;
N = 6400;
Nfp = 40;
sFP = chebpts(Nfp,[0 L],1);
[n,nprime,~] = ParalleltorqDen(sFP);
[X,Xs,Xss,Xsss] = fibGeo(sFP);
Oonevel = UFromNFPIntegral(X,Xs,Xss,Xsss,sFP,Nfp,L,n,nprime);
allfiniteparts = zeros(Nfp,3);
ts = sFP;
for iPt=1:length(ts)
t = ts(iPt,:);

[Xpt,Xspt,Xsspt] = fibGeo(t);
[npt,nprimept]=torqDen(t);
v = cross(Xspt,nprimept)+1/2*cross(Xsspt,npt);

% Integral for R < 2a 
[smalls,wsm] = chebpts(N,[max(0,t-2*a) min(L,t+2*a)]);
[smX,smXs,smXss] = fibGeo(smalls);
[nsm,nprimesm]=torqDen(smalls);
Rsm = Xpt-smX;
nRsm = sqrt(sum(Rsm.*Rsm,2));
smallig = 1/(2*a^2)*(1/a-(3*nRsm)/(8*a^2)).*cross(nsm,Rsm);
numersmall = wsm*smallig;
taucrossn = cross(Xspt,npt);
if (t > 2*a && t < L-2*a)
    asymptsmall = 7/6*v;
elseif (t <=2*a)
    sbar = t/a;
    asymptsmall = taucrossn/(2*a)*(1-1/2*sbar^2+1/8*sbar^3)+...
        1/192*(112+32*sbar^3-9*sbar^4)*v;
else
    sbar = (L-t)/a;
    asymptsmall = taucrossn/(2*a)*(-1+1/2*sbar^2-1/8*sbar^3)+...
        1/192*(112+32*sbar^3-9*sbar^4)*v;
end
er_small = numersmall-asymptsmall

% Integral for R > 2a
lesspart = [0 0 0];
if (t > 2*a)
[sless,wless] = chebpts(N,[0 t-2*a]);
[Xl,Xsl,Xssl] = fibGeo(sless);
[nl,nsl] = torqDen(sless);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
kernel = cross(nl,Rl)./nR.^3;
lesspart = wless*kernel;
end

greaterpart = [0 0 0];
if (t < L-2*a)
[sgreater,wgreater] = chebpts(N,[min(L,t+2*a) L]);
[Xl,Xsl,Xssl] = fibGeo(sgreater);
[nl,nsl] = torqDen(sgreater);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
kernel = cross(nl,Rl)./nR.^3;
greaterpart = wgreater*kernel;
end
total_RPY(iPt,:) = lesspart + greaterpart + numersmall;

[sall,wall] = chebpts(N,[0 L]);
[X,Xs,Xss]=fibGeo(sall);
[n,nprime] = torqDen(sall);
CommIG = taucrossn.*(sall-t)./abs(sall-t).^3+v./abs(sall-t);
R = Xpt-X;
nR = sqrt(sum(R.*R,2));
totIG = cross(n,R)./nR.^3;
FPIG = totIG-CommIG;
[~,minindex] = min(abs(sall-t));
FPIG(minindex,:)=0;
if (t > 2*a && t < L-2*a)
    Local = (log(((L-t)*t)/(4*a^2))+7/6)*v+taucrossn*(L-2*t)/((L-t)*t);
elseif (t <=2*a)
    sbar = t/a;
    Local = taucrossn/(2*a)*(1-1/2*sbar^2+1/8*sbar^3+(L-t-2*a)/(L-t))+...
        (7/12+1/6*sbar^3-3/64*sbar^4+log((L-t)/(2*a)))*v;
else
    sbar = (L-t)/a;
    Local = taucrossn/(2*a)*(-1+1/2*sbar^2-1/8*sbar^3+(2*a-t)/t)+...
        (7/12+1/6*sbar^3-3/64*sbar^4+log(t/(2*a)))*v;
end
finitepart=wall*FPIG;
asymptotic(iPt,:) = finitepart + Local;
totalerror = norm(asymptotic-total_RPY)
allfiniteparts(iPt,:)=finitepart;
ers(1,ia)=totalerror;
end
end

function [X,Xs,Xss,Xsss] = fibGeo(s)
    N = length(s);
    X = 1/sqrt(2)*[cos(s) sin(s) s];
    Xs = 1/sqrt(2)*[-sin(s) cos(s) ones(N,1)];
    Xss = 1/sqrt(2)*[-cos(s) -sin(s) zeros(N,1)];
    Xsss = 1/sqrt(2)*[sin(s) -cos(s) zeros(N,1)];
end

function [n,nprime,ndoubleprime] = torqDen(s)
    [~,Xs,Xss, Xsss] = fibGeo(s);
    n = cos(s).*Xs;
    nprime = -sin(s).*Xs+cos(s).*Xss;
    ndoubleprime = -cos(s).*Xs - 2*sin(s).*Xss + cos(s).*Xsss;
    %n = [2*cos(s).^2 sin(s) s.^2];
    %nprime = [-4*cos(s).*sin(s) cos(s) 2*s];
    %ndoubleprime = [4*sin(s).^2-4*cos(s).^2 -sin(s) 2*ones(length(s),1)];
end

function [n,nprime,ndoubleprime] = ParalleltorqDen(s)
    n = cos(s);
    nprime = -sin(s);
    ndoubleprime = -cos(s);
end

