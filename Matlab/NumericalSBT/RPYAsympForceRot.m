% RPY asymptotics on translational velocity due to force 
L = 2;
as= [0.08 0.04 0.02 0.01];

for ia=2
a = as(ia);
mu = 1;
Nq = 400;
N = 50;
[s,w,b] = chebpts(N,[0 L],1);
D = diffmat(N,[0 L],'chebkind1');
[f,fprime,~] = forceDen(s);
[X,Xs,Xss,Xsss] = fibGeo(s);
Allb_trueFP = precomputeStokesletInts(s,L,0,N,1);
FPVel=OmFromFFPIntegral(X,Xs,Xss,Xsss,s,N,L,f,fprime,Allb_trueFP,mu);
[MTT, MTR, MRT, MRR,sNew] = getGrandMloc(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),a,L,mu,s,0);
OmLocal = MRT*reshape(f',3*N,1);
UAsymp = OmLocal+FPVel;
U_ref=1/(8*pi*mu)*upsampleRPYRotTrans(X,s,X,f,s,b,Nq,L,a);
U_ref = sum(U_ref.*Xs,2);
max(abs(UAsymp-U_ref))
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

