% RPY asymptotics on translational velocity due to torque 
% This is asymptotics on 8*pi*mu*Omega
L = 2;
a = 0.01;
mu = 1;
N = 1000;
Nfp = 40;

sFP = chebpts(Nfp,[0 L],1);
[f,fprime,fdoubleprime] = forceDen(sFP);
[X,Xs,Xss,Xsss] = fibGeo(sFP);
Oonevel = OmFromFFPIntegral(X,Xs,Xss,Xsss,sFP,Nfp,L,f,fprime);
allfiniteparts = zeros(Nfp,3);

ts = sFP;
for iPt=1:length(ts)
t = ts(iPt,:);
[Xpt,Xspt,Xsspt] = fibGeo(t);
[fpt,fprimept]=forceDen(t);
omega = (cross(Xspt,fprimept)+1/2*cross(Xsspt,fpt));
TauCrossf = cross(Xspt,fpt);


% Integral for R < 2a 
[smalls,wsm] = chebpts(N,[max(0,t-2*a) min(L,t+2*a)]);
[smX,smXs,smXss] = fibGeo(smalls);
[fsm,fprimesm]=forceDen(smalls);
Rsm = Xpt-smX;
nRsm = sqrt(sum(Rsm.*Rsm,2));
smallig = 1/(2*a^2)*(1/a-(3*nRsm)/(8*a^2)).*cross(fsm,Rsm);
numersmall = wsm*smallig;
if (t > 2*a && t < L-2*a)
    asymptsmall = 7/6*omega;
elseif (t <=2*a)
    sbar = t/a;
    asymptsmall = TauCrossf/(2*a)*(1-1/2*sbar^2+1/8*sbar^3)+...
        (7/12+1/6*sbar^3-3/64*sbar^4)*omega;
else
    sbar = (L-t)/a;
    asymptsmall = TauCrossf/(2*a)*(-1+1/2*sbar^2-1/8*sbar^3)+...
        (7/12+1/6*sbar^3-3/64*sbar^4)*omega;
end
er_small = numersmall-asymptsmall

% Integral for R > 2a
lesspart = [0 0 0];
if (t > 2*a)
[sless,wless] = chebpts(N,[0 t-2*a]);
[Xl,Xsl,Xssl] = fibGeo(sless);
[fl,fsl] = forceDen(sless);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
kernel = cross(fl,Rl)./nR.^3;
lesspart = wless*kernel;
end

greaterpart = [0 0 0];
if (t < L-2*a)
[sgreater,wgreater] = chebpts(N,[min(L,t+2*a) L]);
[Xl,Xsl,Xssl] = fibGeo(sgreater);
[fl,fsl] = forceDen(sgreater);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
kernel = cross(fl,Rl)./nR.^3;
greaterpart = wgreater*kernel;
end
total_RPY(iPt,:) = lesspart + greaterpart + numersmall;

[sall,wall] = chebpts(N,[0 L]);
[Xall,Xsall,Xssall]=fibGeo(sall);
[f,fprime] = forceDen(sall);
Ds = stackMatrix(diffmat(N,[0 L]));
%MFP = NLMatrixRotTrans(reshape(X',3*N,1),reshape(Xs',3*N,1),Ds,sall,wall,N);
CommIG = TauCrossf.*(sall-t)./abs(sall-t).^3+omega./abs(sall-t);
R = Xpt-Xall;
nR = sqrt(sum(R.*R,2));
totIG = cross(f,R)./nR.^3;
FPIG = totIG-CommIG;
[~,minindex] = min(abs(sall-t));
FPIG(minindex,:)=0;
if (t > 2*a && t < L-2*a)
    Local = (log(((L-t)*t)/(4*a^2))+7/6)*omega+TauCrossf*(L-2*t)/((L-t)*t);
elseif (t <=2*a)
    sbar = t/a;
    Local = TauCrossf/(2*a)*(1-1/2*sbar^2+1/8*sbar^3+(L-t-2*a)/(L-t))+...
        (7/12+1/6*sbar^3-3/64*sbar^4+log((L-t)/(2*a)))*omega;
else
    sbar = (L-t)/a;
    Local = TauCrossf/(2*a)*(-1+1/2*sbar^2-1/8*sbar^3+(2*a-t)/t)+...
        (7/12+1/6*sbar^3-3/64*sbar^4+log(t/(2*a)))*omega;
end
finitepart=wall*FPIG;
finitePartDotTau = dot(finitepart,Xs(iPt,:));
asymptotic(iPt,:) = finitepart + Local;
totalerror = norm(asymptotic-total_RPY)
allfiniteparts(iPt,:)=finitepart;
allFPPars(iPt,:)=finitePartDotTau;
end

function [X,Xs,Xss,Xsss] = fibGeo(s)
    N = length(s);
    X = 1/sqrt(2)*[cos(s) sin(s) s];
    Xs = 1/sqrt(2)*[-sin(s) cos(s) ones(N,1)];
    Xss = 1/sqrt(2)*[-cos(s) -sin(s) zeros(N,1)];
    Xsss = 1/sqrt(2)*[sin(s) -cos(s) zeros(N,1)];
end

function [f,fprime,fdoubleprime] = forceDen(s)
    f = [cos(s).^2 sin(s) s.^2];
    fprime = [-2*cos(s).*sin(s) cos(s) 2*s];
    fdoubleprime = [2*sin(s).^2-2*cos(s).^2 -sin(s) 2*ones(length(s),1)];
end

