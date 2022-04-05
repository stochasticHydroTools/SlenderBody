% % % Single layer boundary integral for ellipsoidal filament and comparison
% % % with Johnson
% % % Discretization on centerline
BI=1;
SBT=0;
trans=1;
rot=0;
nTurns=0.5;
if (BI)
index=1;
L = 2;
a = 40e-3;
Ns = [10 20 40 80];
for N=Ns
mu = 1;
ds = L/N;
s = (0.5:N)'*ds-1;
Nc = 500;
[X,Xs,Xss,n1,n2,dn1,dn2] = fibGeoHelix(s,L,nTurns);
radius = a*sqrt(1-s.^2);
% Choose number of theta pts so that spacing on cross section = centerline
CLdist = 2*pi*radius;
Ntheta = max(ceil(CLdist/ds),2);
ThetaStart = [1;1+cumsum(Ntheta(1:end-1))];
dtheta = 2*pi./Ntheta;
rho = radius/a;
Drho = -s./sqrt(1-s.^2);
% Compute mobility matrix U = M*F
Nt=sum(Ntheta);
M = zeros(3*Nt);
U3 = zeros(Nt,3);
TanVecs = zeros(Nt,3);
RhoHats = zeros(Nt,3);
chkindex=0;
for iN=1:N
    for iT=1:Ntheta(iN)
        chkindex=chkindex+1;
        iindex = ThetaStart(iN)+(iT-1);
        if (iindex-chkindex ~=0)
            keyboard
        end
        Xic = X(iN,:);
        itheta = iT*dtheta(iN);
        iRhoHat = n1(iN,:)*cos(itheta)+n2(iN,:)*sin(itheta);
        if (trans)
            U3(iindex,:) = Uhelix(s(iN),L,nTurns);
        end
        if (rot)
            U3(iindex,:) =U3(iindex,:)+1/(a^2)*cross(Xs(iN,:),radius(iN)*iRhoHat); % Omega^\parallel = 1
        end
        TanVecs(iindex,:)=Xs(iN,:);
        RhoHats(iindex,:)=iRhoHat;
        rhohat_s = dn1(iN,:)*cos(itheta)+dn2(iN,:)*sin(itheta);
        Xi = Xic + radius(iN)*iRhoHat;
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
            for jT=1:Ntheta(jN)
                jindex = ThetaStart(jN)+(jT-1);
                Xjc = X(jN,:);
                jtheta = jT*dtheta(jN);
                jRhoHat = n1(jN,:)*cos(jtheta)+n2(jN,:)*sin(jtheta);
                Xj = Xjc + radius(jN)*jRhoHat;
                sprime = s(jN);
                Xjsph = efflength*(sprime-se)*Xs(iN,:)+c*sqrt(1-sprime^2)*jRhoHat+Xic;
                R = Xi-Xj;
                Re = Xisph-Xjsph;
                nR = norm(R);
                nRe = norm(Re);
                if (nR > 1e-12)
                    M(3*iindex-2:3*iindex,3*jindex-2:3*jindex) = 1/(8*pi*mu)*(eye(3)/nR+R'*R/nR^3)...
                        *ds*dtheta(jN);
                end
                if (nRe > 1e-12)
                    M(3*iindex-2:3*iindex,3*iindex-2:3*iindex) = M(3*iindex-2:3*iindex,3*iindex-2:3*iindex)...
                        -1/(8*pi*mu)*(eye(3)/nRe+Re'*Re/nRe^3)*ds*dtheta(jN);
                end
            end
        end
        M(3*iindex-2:3*iindex,3*iindex-2:3*iindex) =M(3*iindex-2:3*iindex,3*iindex-2:3*iindex)+Ans2;
    end
end
Allf = reshape(M \ reshape(U3',3*Nt,1),3,Nt)';
% Integrate over theta to get centerline force and parallel torque
favg = zeros(N,3);
navg = zeros(N,3);
for iN=1:N
    try
    inds = ThetaStart(iN):ThetaStart(iN+1)-1;
    catch
    inds = ThetaStart(iN):Nt;
    end
    favg(iN,:)=sum(Allf(inds,:))*dtheta(iN);
    navg(iN,:)=radius(iN)*sum(cross(RhoHats(inds,:),Allf(inds,:)))*dtheta(iN);
end
navg_par = sum(navg.*Xs,2);
nperp=navg-navg_par.*Xs;
FToCompare{index}=favg;
NToCompare{index}=navg_par;
sv{index}=s;
index=index+1;
end
FErrors=zeros(index-2,1);
NErrors=zeros(index-2,1);
for iEr=1:index-2
    Refined = 1/2*(FToCompare{iEr+1}(1:2:end,:)+FToCompare{iEr+1}(2:2:end,:));
    FErrorVec = FToCompare{iEr}-Refined;
    FErrors(iEr) = sqrt(ds*sum(sum(FErrorVec.*FErrorVec,2)));
    Refined = 1/2*(NToCompare{iEr+1}(1:2:end)+NToCompare{iEr+1}(2:2:end));
    NErrorVec = NToCompare{iEr}-Refined;
    NErrors(iEr) = sqrt(ds*sum(NErrorVec.*NErrorVec));
end
end
if (SBT)
%ks = [2 2.25 2.5 2.75 3];
ks = 0;
UEr = zeros(index-1,length(ks));
OmEr = zeros(index-1,length(ks));
normForceEr = zeros(index-1,length(ks));
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
%radius = a*sqrt(1-(2*s/L-1).^2);
aRPY = a;%*exp(3/2)/4;
D=diffmat(NSBT,[0 L],'chebkind1');
[X,Xs,Xss,n1,n2,~,~] = fibGeoHelix(s-1,L,nTurns);
XSB=X;
warning('True finite part!')
Allb = precomputeStokesletInts(s,L,0,NSBT,1);
[MTT, MTR, MRT, MRR,sNew] = GrandJohnsonMob(NSBT,Xs,Xss,a,L,mu,s,k);
SletFP = StokesletFinitePartMatrix(X,Xs,Xss,D,s,L,NSBT,mu,Allb);
Xsss = D*Xss;
fBI = Rdowncompare*FToCompare{getIndex};
nBI = Rdowncompare*NToCompare{getIndex};
if (trans && ~rot)
    UBI = reshape((MTT+SletFP)*reshape(fBI',3*NSBT,1),3,[])';
    OmBI = zeros(NSBT,1);
else
    UBI = reshape((MTT+SletFP)*reshape(fBI',3*NSBT,1) + MTR*nBI,3,NSBT)';
    UBI= UBI + UFromNFPIntegral(X,Xs,Xss,Xsss,s,NSBT,L,nBI,D*nBI,Allb,mu);
    OmBI = MRT*reshape(fBI',3*NSBT,1)+OmFromFFPIntegral(X,Xs,Xss,Xsss,s,NSBT,L,fBI,D*fBI,Allb,mu)+...
        MRR*nBI;
end
if (trans)
    UBI = UBI - Uhelix(s-1,L,nTurns);
end
if (rot)
    OmBI = OmBI-1/a^2;
end
UEr(getIndex,ik) = sqrt(wD*sum((RToDouble*UBI).*(RToDouble*UBI),2));
OmEr(getIndex,ik) = sqrt(wD*((RToDouble*OmBI).*(RToDouble*OmBI)));
% Rotlet FP matrices column by column
% MTR_FP = zeros(3*NSBT,NSBT);
% for iCol=1:NSBT
%     n = zeros(NSBT,1);
%     n(iCol)=1;
%     MTR_FP(:,iCol) = reshape(UFromNFPIntegral(X,Xs,Xss,Xsss,s,NSBT,L,n,D*n,Allb,mu)',3*NSBT,1);
% end
% MRT_FP = zeros(NSBT,3*NSBT);
% for iCol=1:3*NSBT
%     f = zeros(3*NSBT,1);
%     f(iCol)=1;
%     f = reshape(f,3,NSBT)';
%     MRT_FP(:,iCol) = OmFromFFPIntegral(X,Xs,Xss,Xsss,s,NSBT,L,f,D*f,Allb,mu);
% end
MSBT = MTT+SletFP;
U3 = Uhelix(s-1,L,nTurns);
USB = reshape(U3',3*NSBT,1);
OmSB = ones(NSBT,1)./a^2;
fSBT = reshape(MSBT\ USB,3,NSBT)';
% Upsample fsbt to meaasure error
% GrandMobility = [MTT+SletFP MTR+MTR_FP; MRT+MRT_FP MRR];
% UOm = [USB; OmSB];
% fn = GrandMobility \ UOm;
% fSBT = reshape(fn(1:3*NSBT),3,[])';
% npar = fn(3*NSBT+1:end);
Rup = barymat(sBI+1,s,b);
fSBT_c = Rup*fSBT;
% nSBT_c = Rup*npar;
% 2 norm difference in force and torque
forceEr = fSBT_c-favg;
normForceEr(getIndex,ik) = sqrt(wBI*sum(forceEr.*forceEr,2));
% TorqEr = nSBT_c-navg_par;
% normTorqEr(ik) = sqrt(wBI*(TorqEr.^2))
end
end
if (rot)
    OmEr=OmEr*a^2;
end
end
% % 
% % 
function U3 = Uhelix(s,L,nTurns)
    N=length(s);
    L = L/nTurns;
    U3 = L/(2*pi)*[-sin(2*pi*s/L) cos(2*pi*s/L) zeros(N,1)];
end

%             
function [X,Xs,Xss,n1,n2,dn1,dn2] = fibGeoHelix(s,L,nTurns)
    warning('s matters for a helix!')
    N = length(s);
    nTurns=0.5;
    L = L/nTurns;
    X = [cos(2*pi*s/L)*L/(2*pi) sin(2*pi*s/L)*L/(2*pi) s]/sqrt(2);
    Xs = [-sin(2*pi*s/L) cos(2*pi*s/L) ones(N,1)]/sqrt(2);
    Xss = [-cos(2*pi*s/L) -sin(2*pi*s/L) zeros(N,1)]/sqrt(2)*2*pi/L;
    n1 = [-cos(2*pi*s/L) -sin(2*pi*s/L) zeros(N,1)];
    dn1 = [sin(2*pi*s/L)*2*pi/L -cos(2*pi*s/L)*2*pi/L zeros(N,1)];
    n2 = cross(Xs,n1);
    dn2 = cross(Xss,n1)+cross(Xs,dn1);
end
      
function [X,Xs,Xss,n1,n2,dn1,dn2] = fibGeo(s,L)
    N = length(s);
    X = [zeros(N,2) s];
    Xs = [zeros(N,2) ones(N,1)];
    Xss = zeros(N,3);
    n1 = [ones(N,1) zeros(N,2)];
    dn1 = zeros(N,3);
    n2 = cross(Xs,n1);
    dn2 = zeros(N,3);
end

function TotInt = IntegrateEllipsoid(efflength,c,se,tau,theta,n1,n2)
    syms spr tpr
    prRhoHat = n1*cos(tpr)+n2*sin(tpr);
    iRhoHat = n1*cos(theta)+n2*sin(theta);
    ReVec = efflength*(spr-se)*tau+c*sqrt(1-spr.^2)*prRhoHat-c*sqrt(1-se.^2)*iRhoHat;
    OneOverR = 1/sqrt(dot(ReVec,ReVec));
    RRtOverR3 = ReVec'*ReVec./dot(ReVec,ReVec).^(3/2);
    OneOverRIntegral = integral2(matlabFunction(OneOverR),-1,1,0,2*pi,'AbsTol',1e-12,'RelTol',1e-12);
    RRtOverR3Int = zeros(3);
    for iDim=1:3
        for jDim=1:3
            RRtOverR3Int(iDim,jDim) = ...
                integral2(matlabFunction(RRtOverR3(iDim,jDim)),-1,1,0,2*pi,'AbsTol',1e-12,'RelTol',1e-12);
        end
    end
    TotInt = OneOverRIntegral*eye(3)+RRtOverR3Int;
end

    