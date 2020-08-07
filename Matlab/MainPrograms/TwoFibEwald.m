% 2 Fiber Ewald Test
N=32;
Lf=2;
Lx = 2.5; % periodic domain size
Ly = 3.7;
Lz = 2.2;
xi = 4; % Ewald parameter
g = 0.1; % shift in periodic domain
mu=1;    
epsilon=1e-3;
[s, ~, b] = chebpts(N+4, [0 Lf], 2);
[s0,w0,~] = chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.
D=diffmat(N,1,[0 Lf],'chebkind1');
nFib=2;
X1 = 1/sqrt(2)*[cos(s0) sin(s0) s0];
f1 = [cos(2*pi*s0/Lf) sin(2*pi*s0/Lf) s0-Lf/2];
sh=0.39+Ly;
X2 = [0.25+zeros(N,2) s0+0.1]+sh*[g 1 0];
f2 = [s0-Lf/2 sin(2*pi*s0/Lf) cos(2*pi*s0/Lf)];
X=[X1; X2];
f=[f1; f2]';
uloc=zeros(2*N,3);
Xs=zeros(2*N,3);
for iFib=1:nFib
    inds=(iFib-1)*N+1:iFib*N;
    Xs(inds,:) = D*X(inds,:);
%     Mloc = getMloc(N,Xs(inds,:),eps,Lf,mu,s0);
%     uloc(inds,:)=reshape(Mloc*reshape(X(inds,:)',3*N,1),3,N)';
end
% MnL = MNonLocalSlow(nFib,N,s0,w0,Lf,epsilon,f,X,Xs,D,mu);
MnLPer = MNonLocal(nFib,N,s0,w0,Lf,epsilon,f,X,Xs,D,mu,xi,Lx,Ly,Lz,g);
% targs=load('../Python/tagrs.txt');
% for iT=1:1000
%     tvels(iT,:)=nearField(targs(iT,:),X1(:,1),X1(:,2),X1(:,3),f1(:,1),...
%         f1(:,2),f1(:,3),Lf,epsilon,mu,0*X1,1);
% end