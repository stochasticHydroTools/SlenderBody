% Define a fiber
% This is asymptotics on 8*pi*mu*U
L = 1;
a = 10e-3;
N = 100;
%ds = L/(N-1);
%w=ds*ones(1,N);
%s = (0:ds:L)';
%NCheb=40;
%[sCheb,wCheb,bCh]=chebpts(NCheb,[0 L],1);
[s,w,b]=chebpts(N,[0 L],1);
Ds = stackMatrix(diffmat(N,[0 L],'chebkind1'));
X = 1/sqrt(2)*[cos(s) sin(s) s];
Xs = 1/sqrt(2)*[-sin(s) cos(s) ones(N,1)];
Xss = 1/sqrt(2)*[-cos(s) -sin(s) zeros(N,1)];
f = [cos(s).^2 sin(s) s.^2]; % force densities
% X = [s zeros(N,2)];
% Xs = [ones(N,1) zeros(N,2)];
% XCheb = [sCheb zeros(NCheb,2)];
% fCheb = [cos(sCheb) sin(sCheb) sCheb]; % force densities
asymptotics=0*f;
RPYSum=0*f;
finiteparts=0*f;
%finiteparts2=finiteparts;
% inds = 1:10:ceil(10*a/ds);
inds=1:N;%100:N;
% if (N > 10000)
%     inds=[1:100:N N];
% elseif (N> 1000)
%     inds=[1:10:N N];
% end
Uactual = upsampleRPY(X(inds,:),s(inds),X,f,s,b,256,L,a);
for iT=inds
t=s(iT);
R = X(iT,:)-X;
nR = sqrt(sum(R.*R,2));
Rdotf = sum(R.*f,2);

% Stokeslet integral
Xsdotf = dot(Xs(iT,:),f(iT,:));
StokIG = f./nR + Rdotf.*R./nR.^3;
CommIG = (f(iT,:)+Xs(iT,:)*Xsdotf)./abs(s-t);
FPIG = StokIG-CommIG;
FPIG(iT,:)=0;

% Smooth finite part integrand so it is 0 on [-2a, 2a]
if (t > 2*a && t < L-2*a)
    aI = log((4*(L-t).*t)/(a^2))-log(16)+(1/6-0*2/3*(a^2/(2*t^2)+a^2/(2*(L-t)^2)))+23/6;
    atau = log((4*(L-t).*t)/(a^2))-log(16)-3*(1/6-0*2/3*(a^2/(2*t^2)+a^2/(2*(L-t)^2)))+1/2;
elseif (t <=2*a)
    aI = log((L-t)/(2*a))+2+4*t/(3*a)-3*t^2/(16*a^2);
    atau = log((L-t)/(2*a))+t^2/(16*a^2);
else
    aI = log(t/(2*a))+2+4*(L-t)/(3*a)-3*(L-t)^2/(16*a^2);
    atau = log(t/(2*a))+(L-t)^2/(16*a^2);
end
asymptotics(iT,:) = aI*f(iT,:)+atau*Xs(iT,:)*dot(Xs(iT,:),f(iT,:));
finiteparts(iT,:) = w*FPIG;
% FPfirst = StokesletFPSpectral(NCheb,a,L,XCheb,fCheb,sCheb,t,X(iT,:));
% FPsecond = log((t*(L-t))/(4*a^2))*(f(iT,:)+Xs(iT,:)*dot(Xs(iT,:),f(iT,:)));
% if (t < 2*a)
%     FPsecond = log((L-t)/(2*a))*(f(iT,:)+Xs(iT,:)*dot(Xs(iT,:),f(iT,:)));
% elseif (t > L-2*a)
%     FPsecond = log(t/(2*a))*(f(iT,:)+Xs(iT,:)*dot(Xs(iT,:),f(iT,:)));
% end
% finiteparts2(iT,:)=FPfirst - FPsecond;
% RPYSum(iT,:) =numer+actual; 
end
asymp = asymptotics+finiteparts;
[MTT, MTR, MRR,sNew] = getGrandMloc(N,reshape(Xs',3*N,1),reshape(Xss',3*N,1),Ds,0,0,a,L,1/(8*pi),s,0);
fst=reshape(f',3*N,1);
Local=MTT*fst;



% Nuni=1000;
% CoeffsToValsCheb = cos(acos(2*(s/L)-1).*(0:N-1));
% ds = L/(Nuni-1);
% sUni = (0:ds:L)';
% CoeffstoValsUniform = cos(acos(2*sUni/L-1).* (0:N-1));
% ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);
% FUniform = ChebtoUniform*finiteparts;

% Do the finite part to spectral accuracy with 20 pts
% Nc = 24;
% [s0,w0,b0] = chebpts(Nc, [0 L], 1); % 1st-kind grid for ODE.
% Xc = 1/sqrt(2)*[cos(s0) sin(s0) s0];
% Xsc = 1/sqrt(2)*[-sin(s0) cos(s0) ones(Nc,1)];
% fc = [cos(1-s0.^20) sin(s0) 1-s0.^10]; % force densities
% D = diffmat(Nc, 1, [0 L], 'chebkind1');
% Ds = zeros(3*Nc);
% for id=1:3
%     Ds(id:3:3*Nc,id:3:3*Nc)=D;
% end
% MFP = FinitePartMatrix(Xc,Xsc,D,Ds,s0,L,Nc,1/(8*pi));
% FPCheb = reshape(MFP*reshape(fc',3*Nc,1),3,Nc)';
% % Resample at s(inds)
% CoeffsToValsCheb = cos(acos(2*(s0/L)-1).*(0:Nc-1));
% CoeffstoValsUniform = cos(acos(2*s(inds)/L-1).* (0:Nc-1));
% ChebtoUniform =  CoeffstoValsUniform*(CoeffsToValsCheb)^(-1);
% FPUniform = ChebtoUniform*FPCheb;

