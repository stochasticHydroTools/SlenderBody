% Single layer boundary integral for ellipsoidal filament and comparison
% with Johnson
% Discretization on centerline
addpath(genpath('/home/om759/Documents/SLENDER_FIBERS'));
%clear;
BI=1;
SBT=1;
%% Solve with bundary integral method
if (BI)
for Nthet = [16]
trans=0;
rot=1;
index=1;
L = 2;
a = 80e-3;
Ns = 200;
for N= Ns
mu = 1;
dtheta = 2*pi/Nthet;
[s,w,b]=chebpts(N,[-1 1],1);
Nc = 500;
[sc,wc,bc]=chebpts(Nc,[-1 1],1);
Rcompare = barymat(sc,s,b);
D=diffmat(N,[0 L],'chebkind1');
[X,Xs,Xss,n1,n2,dn1,dn2] = fibGeoHelix(s,b,L,D);
radius = a*sqrt(1-s.^2);
rho = radius/a;
Drho = -s./sqrt(1-s.^2);
% Compute mobility matrix U = M*F
Nt=Nthet*N;
M = zeros(3*Nt);
U3 = zeros(Nt,3);
X3 = zeros(Nt,3);
TanVecs = zeros(Nt,3);
RhoHats = zeros(Nt,3);
for iN=1:N
    for iT=1:Nthet
        iindex = (iN-1)*Nthet+iT;
        Xic = X(iN,:);
        itheta = iT*dtheta;
        iRhoHat = n1(iN,:)*cos(itheta)+n2(iN,:)*sin(itheta);
        if (trans)
            U3(iindex,:) = Uhelix(s(iN),L);
        end
        if (rot)
            U3(iindex,:) = U3(iindex,:)+1/a^2*cross(Xs(iN,:),radius(iN)*iRhoHat); % Omega^\parallel = 1
        end
        TanVecs(iindex,:)=Xs(iN,:);
        RhoHats(iindex,:)=iRhoHat;
        rhohat_s = dn1(iN,:)*cos(itheta)+dn2(iN,:)*sin(itheta);
        Xi = Xic + radius(iN)*iRhoHat;
        X3(iindex,:)=Xi;
    end
end
    % Compute effective spheriod
    efflength = 1+a*rho(iN)*dot(Xs(iN,:),rhohat_s);
    %findase = @(cse)([cse(1)*sqrt(1-cse(2)^2)-a*rho(iN) cse(1)*cse(2)/sqrt(1-cse(2)^2)+a*Drho(iN)]);
    %cseroots = fsolve(findase,[a s(iN)]);
    %c = cseroots(1); se = cseroots(2);
    if (abs(Drho(iN)) > 1e-10)
        se = (rho(iN) - sqrt(rho(iN)^2 + 4*Drho(iN)^2))/(2*Drho(iN));
        c = a*rho(iN)/sqrt(1-se^2);
    else
        se = 0;
        c = a*rho(iN);
    end
    ecc = sqrt(1-c^2/efflength^2);
    Rat=log((1+ecc)/(1-ecc));
    ResistancePar = 16*pi*efflength*ecc^3*mu/((1+ecc^2)*Rat-2*ecc);
    ResistancePerp = 32*pi*efflength*ecc^3*mu/((3*ecc^2-1)*Rat+2*ecc);
    ParCoeff = 4*pi/ResistancePar;
    PerpCoeff = 4*pi/ResistancePerp;
    Ans2 = ParCoeff*Xs(iN,:)'*Xs(iN,:)+PerpCoeff*(eye(3)-Xs(iN,:)'*Xs(iN,:));
    sprime = se;
    Xisph = efflength*(sprime-se)*Xs(iN,:)+c*sqrt(1-sprime^2)*iRhoHat+Xic;
    for jN=1:N
        for jT=1:Nthet
            jindex = (jN-1)*Nthet+jT;
            Xjc = X(jN,:);
            jtheta = jT*dtheta;
            jRhoHat = n1(jN,:)*cos(jtheta)+n2(jN,:)*sin(jtheta);
            Xj = Xjc + radius(jN)*jRhoHat;
            sprime = s(jN);
            Xjsph = efflength*(sprime-se)*Xs(iN,:)+c*sqrt(1-sprime^2)*jRhoHat+Xic;
            R = Xi-Xj;
            Re = Xisph-Xjsph;
            nR = norm(R);
            nRe = norm(Re);
            nRs(jN)=nR;
            nRes(jN)=nRe;
            if (nR > 1e-12)
                M(3*iindex-2:3*iindex,3*jindex-2:3*jindex) = 1/(8*pi*mu)*(eye(3)/nR+R'*R/nR^3)...
                    *w(jN)*dtheta;
            end
            if (nRe > 1e-12)
                M(3*iindex-2:3*iindex,3*iindex-2:3*iindex) = M(3*iindex-2:3*iindex,3*iindex-2:3*iindex)...
                    -1/(8*pi*mu)*(eye(3)/nRe+Re'*Re/nRe^3)*w(jN)*dtheta;
            end
        end
    end
    M(3*iindex-2:3*iindex,3*iindex-2:3*iindex) =M(3*iindex-2:3*iindex,3*iindex-2:3*iindex)+Ans2;
end
Allf = reshape(M \ reshape(U3',3*Nt,1),3,Nt)';
% Integrate over theta to get centerline force and parallel torque
favg = zeros(N,3);
navg = zeros(N,3);
for iN=1:N
    inds = (iN-1)*Nthet+1:iN*Nthet;
    favg(iN,:)=sum(Allf(inds,:))*dtheta;
    navg(iN,:)=radius(iN)*sum(cross(RhoHats(inds,:),Allf(inds,:)))*dtheta;
end
navg_par = sum(navg.*Xs,2);
nperp=navg-navg_par.*Xs;
sBI = s;
wBI = w;
XBI=X;
FToCompare{index}=Rcompare*favg;
NToCompare{index}=Rcompare*navg_par;
index=index+1;
end
FErrors=zeros(index-2,1);
NErrors=zeros(index-2,1);
for iEr=1:index-2
    FErrorVec = FToCompare{iEr}-FToCompare{iEr+1};
    FErrors(iEr) = sqrt(wc*sum(FErrorVec.*FErrorVec,2));
    NErrorVec = NToCompare{iEr}-NToCompare{iEr+1};
    NErrors(iEr) = sqrt(wc*(NErrorVec.*NErrorVec));
end
end
%% Estimate error using forward SBT operator with various k
% (assumes an ellipsoidal fiber)
if (SBT)
ks = [1 2 2.5 2.85 3];
UEr = zeros(index-1,length(ks));
OmEr = zeros(index-1,length(ks));
normForceEr = zeros(index-1,length(ks));
normTorqEr = zeros(index-1,length(ks));
% % Johnson SBT
for getIndex=1:index-1
[sc,wc,bc]=chebpts(Nc,[-1 1],1);
for ik=1:length(ks)
k = ks(ik);
NSBT=30;
[s,w,b]=chebpts(NSBT,[0 L],1);
[sD,wD,bD]=chebpts(2*NSBT,[0 L],1);
RToDouble = barymat(sD,s,b);
Rdowncompare = barymat(s-1,sc,bc);
D=diffmat(NSBT,[0 L],'chebkind1');
[X,Xs,Xss,n1,n2,~,~] = fibGeoHelix(s-1,b,L,D);
XSB=X;
warning('True finite part!')
Allb = precomputeStokesletInts(s,L,0,NSBT,1);
[MTT, MTR, MRT, MRR,sNew] = GrandJohnsonMob(NSBT,Xs,Xss,a,L,mu,s,k);
SletFP = StokesletFinitePartMatrix(X,Xs,Xss,D,s,L,NSBT,mu,Allb);
fBI = Rdowncompare*FToCompare{getIndex};
nBI = Rdowncompare*NToCompare{getIndex};
if (trans && ~rot)
    UBI = reshape((MTT+SletFP)*reshape(fBI',3*NSBT,1),3,[])';
    OmBI = zeros(NSBT,1);
else
    UBI = reshape((MTT+SletFP)*reshape(fBI',3*NSBT,1) + MTR*nBI,3,NSBT)';
    UBI= UBI + UFromNFPIntegral(X,D,s,NSBT,L,nBI,Allb,mu);
    OmBI = MRT*reshape(fBI',3*NSBT,1)+OmFromFFPIntegral(X,D,s,NSBT,L,fBI,Allb,mu)+...
        MRR*nBI;
end
if (trans)
    UBI = UBI - Uhelix(s-1,L);
end
if (rot)
    OmBI = OmBI-1/a^2;
end
UEr(getIndex,ik) = sqrt(wD*sum((RToDouble*UBI).*(RToDouble*UBI),2));
OmEr(getIndex,ik) = sqrt(wD*((RToDouble*OmBI).*(RToDouble*OmBI)));
end
end
if (rot)
    OmEr=OmEr*a^2;
end
end
clear M
%save(strcat('FloreGeo_a',num2str(a*1000),'Nt',num2str(Nthet),'.mat'))
%exit
% % 
% % 

%% Velocities and fiber geometries
function U3 = Uhelix(s,L)
    N=length(s);
    nTurns=0.5;
    L = L/nTurns;
    U3 = L/(2*pi)*[-sin(2*pi*s/L) cos(2*pi*s/L) zeros(N,1)];
end

%             
function [X,Xs,Xss,n1,n2,dn1,dn2] = fibGeoHelix(s,b,L,D)
    warning('s matters for a helix!')
    N = length(s);
    nTurns=0.5;
    L = L/nTurns;
    X = [cos(2*pi*s/L)*L/(2*pi) sin(2*pi*s/L)*L/(2*pi) s]/sqrt(2);
    Xs = [-sin(2*pi*s/L) cos(2*pi*s/L) ones(N,1)]/sqrt(2);
    Xss = [-cos(2*pi*s/L) -sin(2*pi*s/L) zeros(N,1)]/sqrt(2)*2*pi/L;
    n1 = [-cos(2*pi*s/L) -sin(2*pi*s/L) zeros(N,1)];
    dn12 = [sin(2*pi*s/L)*2*pi/L -cos(2*pi*s/L)*2*pi/L zeros(N,1)];
    dn1 = D*n1;
    n2 = cross(Xs,n1);
    dn2 = D*n2;
end

function [X,Xs,Xss,n1,n2,dn1,dn2] = fibGeoFloren(s,b,L,D)
    warning('s matters for Florens fiber!')
    s=s+1;
    N = length(s);
    q=1;
    syms t
    XsSym = [cos(q*t.^3 .* (t-L).^3) sin(q*t.^3.*(t - L).^3) 1]/sqrt(2);
    Xss = double(subs(diff(XsSym,t),s));
    Xs = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
    n1 = [-sin(q*s.^3.*(s- L).^3) cos(q*s.^3 .* (s-L).^3) zeros(N,1)];
    dn1 = D*n1;
    n2 = cross(Xs,n1);
    dn2 = D*n2;
    X = pinv(D)*Xs;
    X = X-barymat(0,s,b)*X;
end
      
function [X,Xs,Xss,n1,n2,dn1,dn2] = fibGeo(s,b,L,D)
    N = length(s);
    X = [zeros(N,2) s];
    Xs = [zeros(N,2) ones(N,1)];
    Xss = zeros(N,3);
    n1 = [ones(N,1) zeros(N,2)];
    dn1 = zeros(N,3);
    n2 = cross(Xs,n1);
    dn2 = zeros(N,3);
end

%% SBT Evaluations
% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. This is the velocity from scalar torque.
function Oonevel = UFromNFPIntegral(X,D,s0,N,L,n,Allbs,mu)
    Oonevel = zeros(N,3);
    Xs = D*X;
    Xss = D*Xs;
    Xsss = D*Xss;
    nprime = D*n;
    for iPt=1:N
        nXs = norm(Xs(iPt,:));
        b = Allbs(:,iPt);
        gloc = zeros(N,3);
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt,:) = (cross(n(jPt)*Xs(jPt,:),R)/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/(2*nXs^3)*cross(Xs(iPt,:),n(iPt)*Xss(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt,:)=-nprime(iPt)/(2*nXs^3)*cross(Xss(iPt,:),Xs(iPt,:))...
            -n(iPt)/(3*nXs^3)*cross(Xsss(iPt,:),Xs(iPt,:))...
            -3/4*cross(Xs(iPt,:),Xss(iPt,:))*n(iPt)/nXs^5*dot(Xs(iPt,:),Xss(iPt,:));
        Oonevel(iPt,:)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end

% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. 
% This is the integral that gives Omega from the force
function Oonevel = OmFromFFPIntegral(X,D,s0,N,L,f,Allbs,mu)
    Oonevel = zeros(N,1);
    Xs = D*X;
    Xss = D*Xs;
    Xsss = D*Xss;
    fprime=D*f;
    for iPt=1:N
        b = Allbs(:,iPt);
        gloc = zeros(N,1);
        nXs = norm(Xs(iPt,:));
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt) = (dot(cross(R,Xs(iPt,:)),f(jPt,:))/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/(2*nXs^3)*dot(cross(Xs(iPt,:),Xss(iPt,:)),f(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt)= -1/(2*nXs^3)*(dot(fprime(iPt,:),cross(Xss(iPt,:),Xs(iPt,:)))+ ...
            1/3*dot(f(iPt,:),cross(Xsss(iPt,:),Xs(iPt,:))))...
            -3/4*dot(cross(Xs(iPt,:),Xss(iPt,:)),f(iPt,:))/nXs^5*dot(Xs(iPt,:),Xss(iPt,:));
        Oonevel(iPt)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end