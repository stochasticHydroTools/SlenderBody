% RPY asymptotics on translational velocity due to torque 
% This is asymptotics on 8*pi*mu*U
L = 2;
as= [0.08 0.04 0.02 0.01];

for ia=2
a = as(ia);
mu = 1;
Nq = 200;
N = 50;
[s,w,b] = chebpts(N,[0 L],1);
[n,nprime,~] = ParalleltorqDen(s);
[X,Xs,Xss,Xsss] = fibGeo(s);
Allb_trueFP = precomputeStokesletInts(s,L,0,N,1);
Oonevel = UFromNFPIntegral(X,Xs,Xss,Xsss,s,N,L,n,nprime,Allb_trueFP,mu);
[MTT, MTR, MRT, MRR,sNew] = getGrandMloc(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,0);
ULocal = MTR*n;
UAsymp = reshape(ULocal,3,N)'+Oonevel;
U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,n.*Xs,s,b,Nq,L,a);
max(abs(UAsymp-U_ref))
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

