% Compare the actual velocity from the slender body PDE 
% to the slender body approximation (local drag and nonlocal integral) 
% for a curved filament with smooth forcing
addpath('../chebfun-master')
addpath('../functions-solvers')
N=1000;    % Number of points
Lf=4;     % length of fiber
epsilon=1e-2;
[s0,w,b]=chebpts(N,[0 Lf],1); % Chebyshev grid
%Xs = [cos(s0.^3 .* (s0-Lf).^3) sin(s0.^3.*(s0 - Lf).^3) ...
%    ones(N,1)]/sqrt(2);
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
%[~,xp] = ode45(@(s,X) cos(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
%[~,yp]= ode45(@(s,X) sin(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
%zp = s0/sqrt(2);
%X=[xp(2:end) yp(2:end) zp];
Xs = [cos(s0) sin(s0) zeros(N,1)];
X = [sin(s0) -cos(s0) zeros(N,1)];
aeps=2*epsilon;
% Smooth function f
f=[cos(2*pi*s0) -cos(pi*s0) s0-Lf/2]';
% n1s=[-Xs(:,2) Xs(:,1) zeros(N,1)];
% n1s=n1s./(sqrt(n1s(:,1).^2+n1s(:,2).^2+n1s(:,3).^2));
% n2s=cross(Xs,n1s);
% n2s=n2s./(sqrt(n2s(:,1).^2+n2s(:,2).^2+n2s(:,3).^2));
% N2=6;
% t=0:2*pi/N2:2*pi-2*pi/N2;
% avgvalues=zeros(N,3);
% rpyvals=zeros(N,3);
% for iPt=1:N
% %     iPt=ceil(N/2);
%     pts=X(iPt,:)+epsilon*Lf*(cos(t').*n1s(iPt,:)+sin(t').*n2s(iPt,:));
% %     plot3(pts(:,1),pts(:,2),pts(:,3));
%     % Trapezoidal integration for those points
%     for iTrap=1:N2
%         [val,iG]=integrate(X,pts(iTrap,:),f,w,epsilon*Lf);
%         avgvalues(iPt,:)=avgvalues(iPt,:)+val;
%     end
% end
% avgvalues=avgvalues/N2;
% Local = zeros(N,3);
% s=s0*2/Lf-1;
% Ls = log((2*(1-s.^2)+2*sqrt((1-s.^2).^2+4*aeps.^2))./(aeps.^2));
% for iPt=1:N
%     XsXs=Xs(iPt,:)'*Xs(iPt,:);
%     Local(iPt,:) = ((eye(3)-3*XsXs)+Ls(iPt)*(eye(3)+XsXs))*f(:,iPt);
% end
% Non local part
MSkip=NLMatrix(reshape(X',3*N,1),reshape(Xs',3*N,1),s0,w,N,0);
fstack=reshape(f,3*N,1);
NLvel = MSkip*fstack;
NLvel = reshape(NLvel,3,N)';

[s1000, w1000,b1000] = chebpts(2000, [0 Lf], 1); % common grid
NLvelUP1 = barymat(s1000,s0,b)*NLvel;
% erLoc=barymat(s1000,s0,b)*(Local-avgvalues);
% normerLoc=sqrt(sum(erLoc.*erLoc,2));

% Tornberg way
N=28;    % Number of points
Lf=4;     % length of fiber
epsilon=1e-1/2;
mu=1/(8*pi);
[s0,w,b]=chebpts(N,[0 Lf],1); % Chebyshev grid
% Xs = [cos(s0.^3 .* (s0-Lf).^3) sin(s0.^3.*(s0 - Lf).^3) ...
%     ones(N,1)]/sqrt(2);
% opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
% [~,xp] = ode45(@(s,X) cos(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
% [~,yp]= ode45(@(s,X) sin(s^3 * (s-Lf)^3)/sqrt(2),[0; s0],0,opts);
% zp = s0/sqrt(2);
% X=[xp(2:end) yp(2:end) zp];
Xs = [cos(s0) sin(s0) zeros(N,1)];
X = [sin(s0) -cos(s0) zeros(N,1)];
D = diffmat(N, 1, [0 Lf], 'chebkind1');
% Smooth function f
f=[cos(2*pi*s0) -cos(pi*s0) s0-Lf/2]';
fprime=(D*f')';
Xss = D*Xs;
NLvel = NLIntegrate(X,Xs,Xss,s0,N,Lf,f,fprime);

NLvelUP = barymat(s1000,s0,b)*NLvel;
%Compute the L^2 error
% erNL = erLoc+NLvelUP;
% normNL = sqrt(sum(erNL.*erNL,2));

function [v,vals] = integrate(X,p,f,w,radius)
    [N,~]=size(X);
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        vals(iPt,:)=((eye(3)+R'*R)/nR+...
            radius^2/2*(eye(3)-3*(R'*R))/nR^3)*f(:,iPt);
    end
    v=w*vals;
end

function [v,vals] = integrateRPY(X,p,f,w,radius)
    [N,~]=size(X);
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        if (nR==0)
            R=[0 0 0];
        end
        a=radius*sqrt(3/2);%1.1204222;
        if (nR <=2*a)
            M=4/(3*a)*((1-9*nR/(32*a))*eye(3)+3*nR/(32*a)*(R'*R));
        else
            M=4/(3*a)*((3*a/(4*nR)+a^3/(2*nR^3))*eye(3)+...
                (3*a/(4*nR)-3*a^3/(2*nR^3))*(R'*R));
        end
        vals(iPt,:)=M*f(:,iPt);
    end
    v=w*vals;
end    


