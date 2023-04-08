%% RUN CHAIN_LENGTH_SAMP.M PRIOR TO RUNNING THIS FILE
% QuadProgram=0;
% AdHoc=0;
% BVP=1;
[s,w,b]=chebpts(N,[0 L],1);
Nx = N+1;
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
stDNp1 = stackMatrix(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*(N+1),3);
for iR=1:N+1
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*stackMatrix(IntDNp1*RToNp1) I];
ESToB = stackMatrix(barymat(sBlob,sNp1,bNp1));
% Matrix for computing L^2 error on the blob grid
WBlob = ds*diag(ones(3*Nblob,1)); WBlob(1:3,1:3)=1/2*WBlob(1:3,1:3); 
WBlob(end-2:end,end-2:end)=1/2*WBlob(end-2:end,end-2:end);
Xblob = reshape(XBL',[],1);
if (QuadProgram)
% Set up the programming problem
% (E*XonNp1*taubar-Xblob)^T W (E*XonNp1*taubar-Xblob)
Q = XonNp1Mat'*ESToB'*WBlob*ESToB*XonNp1Mat;
f = -XonNp1Mat'*ESToB'*WBlob*Xblob;
c = 1/2*Xblob'*WBlob*Xblob;
% Constraint matrices
% The constraint is taubar^T H{j} taubar + d{j}=0
for j=1:N
    H{j}=zeros(3*Nx,3*Nx);
    H{j}(3*j-2,3*j-2)=1;
    H{j}(3*j-1,3*j-1)=1;
    H{j}(3*j,3*j)=1;
    k{j}=zeros(3*Nx,1);
    d{j}=-1;
end

fun = @(x)quadobj(x,Q,f,c);
nonlconstr = @(x)quadconstr(x,H,k,d);
% Initial guess - just take the closest tangent vector from the blob link
% and the midpoint
x0=zeros(3*Nx,1);
x0(end-2:end)=XBL(Nblob/2+0.5,:);
for iPt=1:N
    [~,ind]=min(abs(sLink-s(iPt)));
    x0(3*iPt-2:3*iPt)=taus(:,ind);
end
options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,...
    'HessianFcn',@(x,lambda)quadhess(x,lambda,Q,H),'MaxFunctionEvaluations',1e5,...
    'MaxIterations',1e5);
[x,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],[],[],nonlconstr,options);
% Construct fiber
X = XonNp1Mat*x;
Xe = ESToB*X;
plot3(Xe(1:3:end),Xe(2:3:end),Xe(3:3:end))
% hold on
%plot3(X(1:3:end),X(2:3:end),X(3:3:end),'o')
opter=1/L*sqrt(1/L*(Xe-Xblob)'*WBlob*(Xe-Xblob))
end
if (AdHoc)
% Obtain positions at Chebyshev nodes
XInterp = zeros(Nx,3);
TauInterp = zeros(N,3);
for iPt=1:Nx
    % Find two closest points
    [dists,inds]=sort(abs(sBlob-sNp1(iPt)),'ascend');
    dtot = dists(1)+dists(2);
    XInterp(iPt,:) =  dists(2)/dtot*XBL(inds(1),:)+dists(1)/dtot*XBL(inds(2),:);
end
% Get tangent vectors
taubar = XonNp1Mat \ reshape(XInterp',[],1);
% Normalize
tauSpec = reshape(taubar(1:3*N),3,[])';
tauSpec = tauSpec./sqrt(sum(tauSpec.*tauSpec,2));
X = XonNp1Mat*[reshape(tauSpec',[],1); taubar(end-2:end)];
Xe = ESToB*X;
plot3(Xe(1:3:end),Xe(2:3:end),Xe(3:3:end))
hold on
%plot3(X(1:3:end),X(2:3:end),X(3:3:end),'o')
aper=1/L*sqrt(1/L*(Xe-Xblob)'*WBlob*(Xe-Xblob))
end
if (BVP)
% Compute a natural spline and sample at Cheb X points
Xspl = zeros(Nx,3);
X = zeros(Nx,3);
%b = 0.025;
BMat = eye(Nx)-bReg^2*DNp1^2;
BMat(1,:)=[1 zeros(1,Nx-1)]; BMat(end,:)=[zeros(1,Nx-1) 1];
for iC=1:3
    cspl=csape(sBlob,XBL(:,iC),'variational');
    Xspl(:,iC)=ppval(cspl,sNp1);
    % Solve the BVP
    X(:,iC) = BMat \ Xspl(:,iC);
end
if (bReg==0)
   X = (ESToB'*WBlob*ESToB) \ ESToB'*WBlob*Xblob;
end
X=reshape(X',[],1);
Xe = ESToB*X;
if (N==8)
set(gca,'ColorOrderIndex',1)
end
tau = reshape(XonNp1Mat \ X,3,[])';
tau = tau(1:N,:);
if (plotTaus)
    plot(s,sqrt(sum(tau.*tau,2))-1,'-o')
    hold on
else
    plot3(Xe(1:3:end),Xe(2:3:end),Xe(3:3:end))
    hold on
end
bvpper=1/L*sqrt(1/L*(Xe-Xblob)'*WBlob*(Xe-Xblob));
end



function hess = quadhess(x,lambda,Q,H)
hess = Q;
jj = length(H); % jj is the number of inequality constraints
for i = 1:jj
    hess = hess + lambda.eqnonlin(i)*H{i};
end
end

function [y,grady] = quadobj(x,Q,f,c)
y = 1/2*x'*Q*x + f'*x + c;
if nargout > 1
    grady = Q*x + f;
end
end

function [y,yeq,grady,gradyeq] = quadconstr(x,H,k,d)
jj = length(H); % jj is the number of inequality constraints
yeq = zeros(1,jj);
for i = 1:jj
    yeq(i) = x'*H{i}*x + k{i}'*x + d{i};
end
y = [];
    
if nargout > 2
    gradyeq = zeros(length(x),jj);
    for i = 1:jj
        gradyeq(:,i) = H{i}*x + k{i};
    end
end
grady = [];
end
