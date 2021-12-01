% RPY asymptotics on rotational velocity due to torque 
% This is asymptotics on 8*pi*mu*Omega
L = 2;
a = 0.02;
mu = 1;
N = 200;
t = 1.965;  % location on fiber
[Xpt,Xspt,Xsspt,Xssspt] = fibGeo(t);
[npt,nprimept]=torqDen(t);
kap = dot(Xsspt,Xsspt);
gam = dot(Xspt,Xssspt);
g = dot(Xspt,npt)/2*Xsspt+Xspt*dot(Xspt,nprimept);

% Integral for R < 2a 
[smalls,wsm] = chebpts(N,[max(0,t-2*a) min(L,t+2*a)]);
[smX,smXs,smXss] = fibGeo(smalls);
[nsm,nprimesm]=torqDen(smalls);
Rsm = Xpt-smX;
nRsm = sqrt(sum(Rsm.*Rsm,2));
Rsm = Rsm./nRsm; % Rsm -> Rhat
CloseInt1 = 1/(a^3)*(1-27.*nRsm/(32*a)+5/64*nRsm.^3/a^3).*nsm;
CloseInt2 = 1/(a^3)*(9/32*nRsm/a-3/64*(nRsm/a).^3).*Rsm.*sum(Rsm.*nsm,2);
numersmall = wsm*(CloseInt1+CloseInt2);
asympsmall=1/(a^2)*(5/4*npt+3/4*Xspt*dot(npt,Xspt));
if (t < 2*a)
    %asympsmall = npt/a^2*(2+t/a-27/32*(2+0.5*(t/a)^2)+3/256*(32+12*(t/a)^2-(t/a)^4)+...
    %    5/64*(4+0.25*(t/a)^4));
    %asympsmall = asympsmall+nprimept/a*(2-0.5*(t/a)^2-27/32*(8/3-1/3*(t/a)^3)+5/64*(32/5-0.2*(t/a)^5));
    %asympsmall= asympsmall+1/a*(g*(9/20 - (3*t^3)/(32*a^3) + (3*t^5)/(320*a^5)));
    asympsmall = npt/a^2*(5/8+t/a-27*t^2/(64*a^2)+5*t^4/(256*a^4));
    asympsmall = asympsmall+Xspt*dot(npt,Xspt)/a^2*(3/8+9/64*t^2/a^2 - 3/256*t^4/a^4);
elseif (t > L-2*a)
    d = L-t;
%     asympsmall = npt/a^2*(2+d/a-27/32*(2+0.5*(d/a)^2)+3/256*(32+12*(d/a)^2-(d/a)^4)+...
%         5/64*(4+0.25*(d/a)^4));
%     asympsmall =asympsmall-nprimept/a*(2-0.5*(d/a)^2-27/32*(8/3-1/3*(d/a)^3)+5/64*(32/5-0.2*(d/a)^5));
%     asympsmall= asympsmall-1/a*(g*(9/20 - (3*d^3)/(32*a^3) + (3*d^5)/(320*a^5)));
    asympsmall = npt/a^2*(5/8+d/a-27*d^2/(64*a^2)+5*d^4/(256*a^4));
    asympsmall = asympsmall+Xspt*dot(npt,Xspt)/a^2*(3/8+9/64*d^2/a^2 - 3/256*d^4/a^4);
end
er_small = (numersmall-asympsmall)/norm(numersmall)

% Integral for R > 2a
lesspart = [0 0 0];
if (t > 2*a)
[sless,wless] = chebpts(N,[0 t-2*a]);
[Xl,Xsl,Xssl] = fibGeo(sless);
[nl,nsl] = torqDen(sless);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
Rhat = Rl./nR;
kernel = -1/2*(nl./nR.^3-3*Rhat.*sum(Rhat.*nl,2)./nR.^3);
lesspart = wless*kernel;
end

greaterpart = [0 0 0];
if (t < L-2*a)
[sgreater,wgreater] = chebpts(N,[min(L,t+2*a) L]);
[Xl,Xsl,Xssl] = fibGeo(sgreater);
[nl,nsl] = torqDen(sgreater);
Rl = Xpt-Xl;
nR = sqrt(sum(Rl.*Rl,2));
Rhat = Rl./nR;
kernel = -1/2*(nl./nR.^3-3*Rhat.*sum(Rhat.*nl,2)./nR.^3);
greaterpart = wgreater*kernel;
end
total_RPY = lesspart + greaterpart + numersmall;

Local = -1/(2*a^2)*(npt-3*Xspt*dot(npt,Xspt))*(1/4-a^2/(2*t^2)-a^2/(2*(L-t)^2));
if (t < 2*a)
    %Local = npt/(8*a^2)+(3*g-nprimept)/(4*a);
    Local = -1/(2*a^2)*(npt-3*Xspt*dot(npt,Xspt))*(1/8-a^2/(2*(L-t)^2));
elseif (t > L-2*a)
    %Local = npt/(8*a^2)-(3*g-nprimept)/(4*a);
    Local = -1/(2*a^2)*(npt-3*Xspt*dot(npt,Xspt))*(1/8-a^2/(2*t^2));
end
s = chebpts(N,[0 L]);
[X,Xs,Xss,~] = fibGeo(s);
[n,~,~]=torqDen(s);
Ds=0;
[MTT, MTR, MRR,sNew] = getGrandMloc(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),Ds,a,L,1/(8*pi),s,0);
fst=reshape(n',3*N,1);
LocalMat=MRR*fst;


TotalAsymptotic = asympsmall+Local;
er = (TotalAsymptotic-total_RPY)/norm(total_RPY)

function [X,Xs,Xss,Xsss] = fibGeo(s)
    N = length(s);
    X = 1/sqrt(2)*[cos(s) sin(s) s];
    Xs = 1/sqrt(2)*[-sin(s) cos(s) ones(N,1)];
    Xss = 1/sqrt(2)*[-cos(s) -sin(s) zeros(N,1)];
    Xsss = 1/sqrt(2)*[sin(s) -cos(s) zeros(N,1)];
end

function [n,nprime,ndoubleprime] = torqDen(s)
%     [~,Xs,Xss, Xsss] = fibGeo(s);
%     n = cos(s).*Xs;
%     nprime = -sin(s).*Xs+cos(s).*Xss;
%     ndoubleprime = -cos(s).*Xs - 2*sin(s).*Xss + cos(s).*Xsss;
    n = [2*cos(s).^2 sin(s) s.^2];
    nprime = [-4*cos(s).*sin(s) cos(s) 2*s];
    ndoubleprime = [4*sin(s).^2-4*cos(s).^2 -sin(s) 2*ones(length(s),1)];
end

