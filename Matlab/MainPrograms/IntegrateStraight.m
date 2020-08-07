% Compare the actual velocity from the slender body PDE 
% to the slender body approximation (local drag) for a straight filament
% with constant forcing
addpath('../chebfun-master')
N=4001;    % Number of points
Lf=2;     % length of filament
[s0,w]=chebpts(N,[0 Lf],2);
epsilon=1e-2/Lf;
aeps=2*epsilon;
Xs = [ones(N,1) zeros(N,2)];
X=[s0 zeros(N,2)];
r = 2*epsilon*sqrt(s0.*(Lf-s0));
% r = epsilon*Lf*ones(N,1);
% Smooth function f
f=ones(N,3)';
fe=reshape(f,3*N,1);
% Real integral
n1s=[-Xs(:,2) Xs(:,1) zeros(N,1)];
n1s=n1s./(sqrt(n1s(:,1).^2+n1s(:,2).^2+n1s(:,3).^2));
n2s=cross(Xs,n1s);
n2s=n2s./(sqrt(n2s(:,1).^2+n2s(:,2).^2+n2s(:,3).^2));
N2=30;
t=0:2*pi/N2:2*pi-2*pi/N2;
avgvalues=zeros(N,3);
for iPt=[floor(N/4) ceil(N/2) ceil(3*N/4)]
%     iPt=ceil(N/2);
    pts=X(iPt,:)+r(iPt)*(cos(t').*n1s(iPt,:)+sin(t').*n2s(iPt,:));
%     plot3(pts(:,1),pts(:,2),pts(:,3));
    % Trapezoidal integration for those points
    for iTrap=1:N2
        [val,iG]=integrate(X,pts(iTrap,:),f,w,epsilon*Lf);
        avgvalues(iPt,:)=avgvalues(iPt,:)+val;
    end
end
avgvalues=avgvalues/N2;
% Local drag model (Mori/Gotz)
XsXs=[1;0;0]*[1 0 0];
Local = zeros(N,3);
s=s0*2/Lf;
Ls = log((2*(1-s.^2)+2*sqrt((1-s.^2).^2+4*aeps.^2))./(aeps.^2));
for iPt=1:N
    Local(iPt,:) = ((eye(3)-3*XsXs)+Ls(iPt)*(eye(3)+XsXs))*[1;1;1];
end
% Exact solutions
exmiddle = [-2*log(-1 + sqrt(1 + 4*epsilon^2)) + (-2 - 4*epsilon^2 + ...
  2*(1 + 4*epsilon^2)^(3/2)*log(1 + sqrt(1 + 4*epsilon^2)))/(1 + ...
   4*epsilon^2)^(3/2);...
   (1 + 2*epsilon^2)/(1 + 4*epsilon^2)^(3/2)-log(-1 + sqrt(1 + 4*epsilon^2)) + ...
   log(1 + sqrt(1 + 4*epsilon^2));
   (1 + 2*epsilon^2)/(1 + 4*epsilon^2)^(3/2)-log(-1 + sqrt(1 + 4*epsilon^2)) + ...
   log(1 + sqrt(1 + 4*epsilon^2))];
exquarter=[(-1 - 8*epsilon^2)/(1 + 16*epsilon^2)^(3/2) - ...
    (3*(9 + 8*epsilon^2))/(9 + 16*epsilon^2)^(3/2) + 2*log(1 + sqrt(1 + 16*epsilon^2)) - ...
   2*log(-3 + sqrt(9 + 16*epsilon^2)); ...
   (1/(2*pi))*((3*(9 + 8*epsilon^2)*pi)/(9 + 16*epsilon^2)^(3/2) + ...
   (pi + 8*epsilon^2*pi)/(1 + 16*epsilon^2)^(3/2) + ...
    2*pi*log(1 + sqrt(1 + 16*epsilon^2)) - 2*pi*log(-3 + sqrt(9 + 16*epsilon^2)));...
   (1/(2*pi))*((3*(9 + 8*epsilon^2)*pi)/(9 + 16*epsilon^2)^(3/2) + ...
   (pi + 8*epsilon^2*pi)/(1 + 16*epsilon^2)^(3/2) + ...
    2*pi*log(1 + sqrt(1 + 16*epsilon^2)) - 2*pi*log(-3 + sqrt(9 + 16*epsilon^2)))];
exend=[1/2*(-((2 + epsilon^2)/(1 + epsilon^2)^(3/2)) - 4*log(epsilon) + ...
     4*log(1 + sqrt(1 + epsilon^2))); (2 + epsilon^2 + 4*(1 + epsilon^2)^(3/2)*...
     log((1 + sqrt(1 + epsilon^2))/epsilon))/(4*(1 + epsilon^2)^(3/2));...
     (2 + epsilon^2 + 4*(1 + epsilon^2)^(3/2)*log((1 + sqrt(1 + epsilon^2))/epsilon))...
     /(4*(1 + epsilon^2)^(3/2))];
sprintf("Local error (Middle, end): (%d, %d)",...
    max(abs(exmiddle'-Local(ceil(N/2),:))),max(abs(exend'-Local(1,:))))
sprintf("Averaging error (Middle, end): (%d, %d)",...
    max(abs(exmiddle'-avgvalues(ceil(N/2),:))),max(abs(exend'-avgvalues(1,:))))

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

