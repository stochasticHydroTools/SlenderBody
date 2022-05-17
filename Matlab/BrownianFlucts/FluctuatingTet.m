% Brownian dynamics simulation of tetrahedron. We are simulating the
% equation dX = (M*F + (div M)kBT) dt + sqrt(2 kBT)*M^(1/2) dW
% Generate tetrahedron
mu = 1;
a = 0.01;
nP = 4;
kbT = 1e-5;
delta = 1e-4; %RFD
MeanLength = 0.1;
% Initially a parallelogram lattice with positions (-h/2,-h/2, h), (-h/2,-h/2,h),
% (h/2,h/2,h), (h/2,h/2,h)
SquareLat = [-MeanLength/2 -MeanLength/2 MeanLength; ...
    MeanLength/2 MeanLength/2 MeanLength;...
   -MeanLength/2 -MeanLength/2 2*MeanLength; ...
    MeanLength/2 MeanLength/2 2*MeanLength];
% Add some noise to get the resting positions
X = SquareLat+0.02*randn(nP,3);

% If it were rigid, what is the diffusion matrix and diffusion coeffient
% (this sets the final time)
M = WallTransTransMatrix(X, mu, a);
Xc = mean(X);
Kin = zeros(3*nP,6);
for iP=1:nP
    Kin(3*(iP-1)+1:3*iP,1:3)=eye(3);
    Kin(3*(iP-1)+1:3*iP,4:6)=-CPMatrix(X(iP,:)-Xc);
end
N = (Kin'*M^(-1)*Kin)^(-1);
rsqCoeff = 2*kbT*trace(N(1:3,1:3));
tf = (0.1*MeanLength)^2/rsqCoeff;
K = 1000*ones(nP*(nP-1)/2,1);
% Set time step from expected timescale
tscale = (6*pi*mu*a)/max(K); % units s
dt = tscale/10;

% Calculate initial rest lengths
Connections = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];
RestLength = zeros(nP*(nP-1)/2,1);
dX = X(Connections(:,1),:)-X(Connections(:,2),:);
RestLen = sqrt(sum(dX.*dX,2));
% Sample initial configuration from GB distribution
dXi = sampleInitial(nP,Connections,K,X,dX,RestLen);
X=X+dXi;

% Run Euler-Maryuama and compute first and second moments of stress
stopcount = ceil(tf/dt);
Stress = zeros(stopcount,9);
COM = zeros(stopcount,3);
for iT=1:stopcount
    COM(iT,:)=mean(X);
    F = calculateForce(X,K,RestLen,Connections);
    str = zeros(3);
    for iP=1:nP
        str=str+X(iP,:)'*F(iP,:);
    end
    Stress(iT,:)=reshape(str,1,9);
    M = WallTransTransMatrix(X, mu, a);
    Udet = reshape(M*reshape(F',[],1),3,[])';
    WRFD = randn(nP,3);
    MPlus = WallTransTransMatrix(X+delta*a/2*WRFD, mu, a);
    MMinus = WallTransTransMatrix(X-delta*a/2*WRFD, mu, a);
    U_RFD = reshape(1/(delta*a)*(MPlus-MMinus)*reshape(WRFD',[],1),3,[])';
    UBrown = reshape(sqrt(2*kbT/dt)*M^(1/2)*randn(3*nP,1),3,[])';
    dXdt = Udet + kbT * U_RFD + UBrown;
    X = X + dt*dXdt;
end
% Compute mean and covariance of stress
StrMean = mean(Stress);
% Covariance matrix. There are 9 elements of the stress
CovMatrix = zeros(9);
for iE=1:9
    for jE=iE+1:9
        covs = cov(Stress(:,iE),Stress(:,jE));
        CovMatrix(iE,iE)=covs(1,1);
        CovMatrix(jE,jE)=covs(2,2);
        CovMatrix(iE,jE)=covs(1,2);
        CovMatrix(jE,iE)=covs(2,1);
    end
end

function energy= calculateEnergy(X,K,RestLen,Connections)
    [nC,~]=size(Connections);
    energy=0;
    dX = X(Connections(:,1),:)-X(Connections(:,2),:);
    r = sqrt(sum(dX.*dX,2));
    for iCon=1:nC
        energy=energy+K(iCon)/2*(r(iCon)-RestLen(iCon))^2;
    end
end


function F = calculateForce(X,K,RestLen,Connections)
    [nP,~]=size(X); [nC,~]=size(Connections);
    F = zeros(nP,3);
    dX = X(Connections(:,1),:)-X(Connections(:,2),:);
    r = sqrt(sum(dX.*dX,2));
    for iCon=1:nC
        iPt = Connections(iCon,1);
        jPt = Connections(iCon,2);
        force = -K(iCon)*dX(iCon,:)/r(iCon)*(r(iCon)-RestLen(iCon));
        F(iPt,:) = F(iPt,:)+force;
        F(jPt,:) = F(jPt,:)-force;
    end
end

function dXi = sampleInitial(nP,Connections,K,X0,dX0,RestLen)
    % Compute the Hessian matrix 
    H = zeros(3*nP);
    for iCon=1:length(Connections)
        i = Connections(iCon,1);
        iinds = 3*i-2:3*i; 
        j = Connections(iCon,2);
        jinds = 3*j-2:3*j;
        rr = K(iCon)*(dX0(iCon,:)'*dX0(iCon,:))/RestLen(iCon)^2;
        % Cross terms
        H(iinds,iinds)=H(iinds,iinds)+rr;
        H(jinds,jinds)=H(jinds,jinds)+rr;
        H(iinds,jinds)=H(iinds,jinds)-rr;
        H(jinds,iinds)=H(jinds,iinds)-rr;
    end
    % Finite difference check of Hessian
    deltaX = rand(nP,3);
    dXr = reshape(deltaX',3*nP,1);
    for pow=1:10
        eps = 10^(-pow);
        energy(pow)= calculateEnergy(X0+eps*deltaX,K,RestLen,Connections);
        energy2(pow) = calculateEnergy(X0,K,RestLen,Connections)+1/2*eps^2*dXr'*H*dXr;
    end
    Diff=abs(energy-energy2); % check that it goes as epsilon^3
    % Decompose Hessian
    [U,Lambda] = eig(H);
    Hhalf = U*abs(Lambda)^(1/2)*U';
    % Initial sample
    b = randn(3*nP,1);
    eigHalf = sort(eig(Hhalf),'descend');
    dXi = pinv(Hhalf,2*eigHalf(7))*b;
    dXi = reshape(dXi,3,[])';
end