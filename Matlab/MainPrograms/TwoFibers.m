% Compare the actual velocity from the slender body PDE 
% to the slender body approximation (local drag) for a straight filament
% with constant forcing
addpath('../chebfun-master')
N=5001;    % Number of points
Lf=pi;     % length of filament
[s,w,b]=chebpts(N,[0 Lf],1);
epsilon=1e-2;
d=32*epsilon*Lf;
%X1=[cos(s) sin(s) zeros(N,1)];
X1=[zeros(N,1) s-Lf/2 zeros(N,1)];
X2=[zeros(N,1)-d zeros(N,1) s-Lf/2];
f = [ones(N,1) zeros(N,2)];
% We are trying to compute the velocity on the second fiber due to the
% second fiber. 
% First the reference solution based on integrating over theta (trapezoid rule)
% Real integral
n1s=[-sin(s) cos(s) zeros(N,1)];
n2s=[cos(s) sin(s) zeros(N,1)];
Ntrap=24;
t=0:2*pi/Ntrap:2*pi-2*pi/Ntrap;
uavg=zeros(1,3);
iPt = ceil(N/2);
pts=X2(iPt,:)+epsilon*Lf*(cos(t').*n1s(iPt,:)+sin(t').*n2s(iPt,:));
% Trapezoidal integration for those points
for iTrap=1:Ntrap
    [val2,~]=integrate(X1,pts(iTrap,:),f',w,epsilon*Lf); 
    % second fib
    uavg=uavg+val2;
end
uavg=uavg/Ntrap;
% Next the solution just on the centerline
velTS = integrate(X1,X2(iPt,:),f',w,epsilon*Lf);
% RPY solution just on the centerline
velRPY = integrate(X1,X2(iPt,:),f',w,sqrt(2)*epsilon*Lf);

function [v,vals] = integrate(X,p,f,w,r)
    [N,~]=size(X);
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        vals(iPt,:)=((eye(3)+R'*R)/nR+r^2/2*...
            (eye(3)-3*(R'*R))/nR^3)*f(:,iPt);
    end
    v=w*vals;
end 

function [v,vals] = integratebyP(X,p,F,w,epsilon)
    [N,~]=size(X);
    Xs =diffmat(N)*X;
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        term1 = -(R*Xs(iPt,:)'*(eye(3)+3*R'*R)-...
            (Xs(iPt,:)'*R+R'*Xs(iPt,:)))*F(:,iPt)/nR^2;
        term2 = -(3*R*Xs(iPt,:)'*(eye(3)-5*R'*R)+...
            3*(Xs(iPt,:)'*R+R'*Xs(iPt,:)))*F(:,iPt)/nR^4;
        vals(iPt,:)=term1+epsilon^2/2*term2;
    end
    v=w*vals;
end
