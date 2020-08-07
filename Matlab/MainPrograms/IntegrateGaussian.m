% Compare the actual velocity from the slender body PDE 
% to the slender body approximation (local drag and nonlocal integral) 
% for a curved filament with smooth forcing
addpath('../chebfun-master')
addpath('../functions-solvers')
N=256;    % Number of points
Lf=2;     % length of fiber
epsilon=1e-2/Lf;
aeps=2*epsilon;
[s0,w,b]=chebpts(N,[0 Lf],1); % Chebyshev grid
Xs = [cos(s0.^3 .* (s0-Lf).^3) sin(s0.^3.*(s0 - Lf).^3) ones(N,1)]/sqrt(2);
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[~,xp] = ode45(@(s,X) cos(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
[~,yp]= ode45(@(s,X) sin(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
zp = s0/sqrt(2);
X=[xp(2:end) yp(2:end) zp];
% Smooth function f
f=[s0-Lf/2 (s0-Lf/2).^2 (s0-Lf/2).^3]';
% Split the non local integral into 2 pieces to see if we can get away with less
% points
nsingf=zeros(N,3);
singf=zeros(N,3);
h=0.06;
eg=double(vpa(eulergamma));
z = (Lf - s0).^2/(2*h^2);
c=0.5*(-2*eg + ei(-(s0.^2./(2*h^2))) - ...
   -ei(-z)+0.5*(log(-z)-log(-1./z))-log(z) + ...
   log(2) + 2*log(h./(Lf - s0)) + log((2*h^2)./s0.^2));
% Non local part
for iPt=1:N
    singf(iPt,:) = (eye(3)+Xs(iPt,:)'*Xs(iPt,:))*c(iPt)*f(:,iPt);
    fsing = g(s0-s0(iPt),h)'.*f(:,iPt);
    freg = f-fsing;
    vals=zeros(N,3);
    for jPt=1:N
        if (jPt~=iPt)
            R = X(iPt,:)-X(jPt,:);
            nR = norm(R);
            Rhat = R/nR;
            vals(jPt,:)=(eye(3)+Rhat'*Rhat)/norm(R)*freg(:,jPt) - ...
                (eye(3)+Xs(iPt,:)'*Xs(iPt,:))/abs(s0(jPt)-s0(iPt))*freg(:,iPt);
        end
    end
    nsingf(iPt,:) = w*vals;
end
MSkip=NLMatrix(reshape(X',3*N,1),reshape(Xs',3*N,1),s0,w,N,0);
fstack=reshape(f,3*N,1);
NLvel = MSkip*fstack;
NLvel = reshape(NLvel,3,N)';
[s1000, w1000,b1000] = chebpts(1000, [0 Lf], 1); % common grid
NLvelUP = barymat(s1000,s0,b)*NLvel;
gUP = barymat(s1000,s0,b)*(nsingf+singf);
% Compute the L2 error of each
erNL = sqrt(w1000*sum((NLvelUP-exsol).*(NLvelUP-exsol),2))
erG = sqrt(w1000*sum((gUP-exsol).*(gUP-exsol),2))

function [v,vals] = integrate(X,p,f,w,epsilon)
    [N,~]=size(X);
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        vals(iPt,:)=((eye(3)+R'*R)/nR+epsilon^2/2*(eye(3)-3*(R'*R))/nR^3)*f(:,iPt);
    end
    v=w*vals;
end

function v = g(x,h)
    v = exp(-x.^2/(2*h^2));
end

