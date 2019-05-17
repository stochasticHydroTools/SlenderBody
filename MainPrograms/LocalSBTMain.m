% Fully 3D slender body theory
addpath('chebfun-master')
addpath('functions-solvers')
N=8;
Lf=2;
Oone=1; % right now only local term of the O(1) part. 
solver = 2; % -3 = RK3; -1 = FE; 1 = BE; 2 = CN; 3 = BDF2; 4 = My 2nd order
% Nodes for solution, plus quadrature and barycentric weights:
[s, w, b] = chebpts(N+4, [0 Lf], 2);
[s0, w0,b2]= chebpts(N, [0 Lf], 1); % 1st-kind grid for ODE.
% Solve for the positions initially
X_s = [cos(s0.^3 .* (s0-Lf).^3) sin(s0.^3.*(s0 - Lf).^3) ones(N,1)]/sqrt(2);
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[~,xp] = ode45(@(s,X) cos(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
[~,yp]= ode45(@(s,X) sin(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
zp = s0/sqrt(2);
fibpts=[xp(2:end) yp(2:end) zp];
D = diffmat(N, 1, [0 Lf], 'chebkind1');
max(abs(D*[xp(2:end) yp(2:end) zp]-X_s)) % error in X_s
[Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf);
EvalMat=(vander(s0-Lf/2));
mu=1;    
eps=1e-3;
Eb=1;
FE = -Eb*Rs*D4s;
dt=1e-4;
t=0;
tf=0.01;
% f=figure;
% plot(fibpts(:,1),fibpts(:,2),'-bo')
% movieframes=getframe(f);
Xpts=[];
Xt= reshape(fibpts',3*N,1);
Xtm1 = Xt;
stopcount=floor(tf/dt+1e-5);
saveEvery=max(1,floor(1e-4/dt+1e-5));
if (solver==-1) s='Forward Euler';
elseif (solver==-3) s='RK3';
elseif (solver==1) s='Backward Euler'; 
elseif (solver==2) s='Crank-Nicolson';
else error('Invalid solver!'); end
disp(strcat(sprintf("Running with N=%d, dt=%f, and solver ",N,dt), s))
tic
for count=0:stopcount-1 
    if (mod(count,saveEvery)==0)
%         plot(X(1:3:end),X(2:3:end),'-bo')
%         movieframes(length(movieframes)+1)=getframe(f);
        Xpts=[Xpts;reshape(Xt,3,N)'];
    end
    [Xt,Xtm1] = getXp1(Xt,Xtm1,solver,dt,FE,LRLs,URLs,Ds,N,eps,mu,chebyshevmat,...
                    Dinv,s0,w0,Oone,I,wIt);
    % Resample the arclength - not doing this yet
end
toc
Xpts=[Xpts;reshape(Xt,3,N)'];
close all;