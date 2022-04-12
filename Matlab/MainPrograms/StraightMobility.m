% Compute  6 x 6 mobility matrix forfiber
mu=1;
L = 1;
eps=1e-2*4/exp(3/2);
a=exp(3/2)/4*eps*L;
N = 32;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
D = diffmat(N,1,[0 L],'chebkind1');
Ds=zeros(3*N);
Ds(1:3:3*N,1:3:3*N)=D;
Ds(2:3:3*N,2:3:3*N)=D;
Ds(3:3:3*N,3:3:3*N)=D;
X0 = [0 0 0];
XsOne = [1;0;0];
%XsOne = 1/sqrt(14)*[2;1;3];
Xs = repmat(XsOne,N,1);
X = XsOne'.*s+X0;
Xcenter = XsOne'*L/2;
delta = 0.1;
Mloc = getGrandMloc(N,Xs,zeros(3*N,1),a,L,mu,s,delta);
Allb_trueFP = precomputeStokesletInts(s,L,0,N,1);
AllbS = precomputeStokesletInts(s,L,a,N,1);
AllbD = precomputeDoubletInts(s,L,a,N,1);
MFP = StokesletFinitePartMatrix(X,reshape(Xs,3,N)',zeros(N,3),D,s,L,N,mu,Allb_trueFP);
NForSmall = 8;
MRPY = ExactRPYSpectralMobility(N,X,reshape(Xs,3,N)',zeros(N,3),zeros(N,3),a,L,mu,s,b,D,AllbS,AllbD,NForSmall);
reg=0.1;
if (reg > 0)
    Sm = MobilitySmoother(s,w,reg*L);
    MRPY = stackMatrix(Sm)*MRPY*stackMatrix(Sm)+0.03*stackMatrix(eye(N)-Sm*Sm);
end
for doFp=[0,1,2]
M = Mloc+doFp*MFP;
if (doFp==2)
    M = MRPY;
end
K = zeros(3*N,6);
Kt=zeros(6,3*N);
for iR=1:N
    dX = X(iR,:)-Xcenter;
    cpmat = [0 dX(3) -dX(2); -dX(3) 0 dX(1); dX(2) -dX(1) 0];
    K(3*(iR-1)+1:3*iR,1:3)=eye(3); 
    K(3*(iR-1)+1:3*iR,4:6)=cpmat; % Omega x (X-Xcenter)
    Kt(1:3,3*(iR-1)+1:3*iR)=w(iR)*eye(3);
    Kt(4:6,3*(iR-1)+1:3*iR)=-w(iR)*cpmat; % (X-Xcenter) x f
end
GMob = pinv(Kt*(M\K)); % mobility matrix
tableline(3*doFp+1:3*(doFp+1))= [GMob(2,2) GMob(1,1)-GMob(2,2) GMob(5,5)];
end
tableline