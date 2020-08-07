N=1025;    % Number of points
Lf=2;     % length of fiber
epsilon=1e-3;
mu=1/(8*pi);
[s0,w,b]=chebpts(N,[0 Lf],1); % Chebyshev grid
X = [s0 zeros(N,2)];
Xs = [ones(N,1) zeros(N,2)];
% Smooth function f
f=[ones(N,1) zeros(N,2)]';
% Centerline velocity
global deltaLocal;
deltaLocal=0.2;
Mloc = getMloc(N,reshape(Xs',3*N,1),epsilon,Lf,mu,s0);
% MSkip=NLMatrix(reshape(X',3*N,1),reshape(Xs',3*N,1),s0,w,N,0);
fstack=reshape(f,3*N,1);
CLvel = reshape(Mloc*fstack,3,N)';
ds=(0.1:0.1:200)'*epsilon;
CLwts = CLwt(ds,2.2*epsilon*Lf,4.4*epsilon*Lf);
vtarg=zeros(length(ds),3);
iPt = ceil(N/2);
sc = s0(iPt);
n = [0 1 0];
targs = X(iPt,:)+n.*ds;
for iTarg=1:length(targs)
    vtarg(iTarg,:) = integrate(X,targs(iTarg,:),f,w,epsilon*Lf*sqrt(2));
end
allvels = CLwts.*CLvel(iPt,:)+vtarg.*(1-CLwts);

% Alternative scheme - fattening the fiber
dfat = 8*epsilon*Lf;
epsFat = dfat/Lf;
Mfatloc = getMloc(1,Xs(iPt,:),epsFat,Lf,mu,sc);
ufat = Mfatloc*f(:,iPt);
newwts = CLwt2(ds,dfat,2*dfat);
allvels2 = newwts.*ufat'+vtarg.*(1-newwts);


function wts = CLwt(d,CLdist,blenddist)
    wts=zeros(length(d),1);
    for iD=1:length(d)
        if (d(iD) < CLdist)
            wts(iD)=1;
        elseif (d(iD) < blenddist)
            wts(iD)=(blenddist-d(iD))/(blenddist-CLdist);
        end
    end
end

function wts = CLwt2(d,CLdist,blenddist)
    wts=zeros(length(d),1);
    for iD=1:length(d)
        if (d(iD) < CLdist)
            wts(iD)=1;
        elseif (d(iD) < blenddist)
            dstar=(d(iD)-(blenddist+CLdist)/2)/(blenddist-CLdist);
            wts(iD)=1/(1+exp(15*dstar));
        end
    end
end


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