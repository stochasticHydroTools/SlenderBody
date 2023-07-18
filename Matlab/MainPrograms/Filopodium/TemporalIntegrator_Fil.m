% Temporal integrator for fibers that resist bend and twist
% Only ONE fiber for this 
RecomputeDiscMats;
Xt = XonNp1Mat*[Xst;XMPor0];
% Calculate twisting force according to BC
thetaarg=theta_s;
theta_s_Psip2= UpsampleThetaBCMat_L \ [thetaarg; PsiBC0; PsiBCL];
Theta_ss = DPsip2_L*theta_s_Psip2;
XBC = UpsampleXBCMat_L*Xt + BCShift_L;
XBC3 = reshape(XBC,3,[])';
% Twist force computed on common N+5 grid, then downsampled to Nx
fTw3 = twmod*((RPsiToNp5*Theta_ss).*cross(DNp5_L*XBC3,DNp5_L^2*XBC3)+...
    (RPsiToNp5*theta_s_Psip2).*cross(DNp5_L*XBC3,DNp5_L^3*XBC3));
fTw = reshape((RNp5ToNp1*fTw3)',3*Nx,1);
fE =  BendForceDensityMat_L*Xt+BCForce_L;
% Arguments for the solver, assuming first order
impcoeff=1;
% Saddle point solve
Xs3 = reshape(Xst,3,N)';
%Xs3_Np1 = reshape((RNp5ToNp1*DNp5_L*XBC3)',[],1);
epsNow = a/(L*Lfacs(iFib));
LDcoeff = log(1/epsNow^2);
M = zeros(3*Nx);
XsFromX = DNp1_L*reshape(Xt,3,Nx)';
XsHat = XsFromX./sqrt(sum(XsFromX.*XsFromX,2));
for iX=1:Nx
    M(3*iX-2:3*iX,3*iX-2:3*iX)=log(1/epsNow^2)/(8*pi*mu)*(eye(3)+XsHat(iX,:)'*XsHat(iX,:));
end
% Solve for fiber evolution
K = KonNp1(Xs3,XonNp1Mat,I);
if (clamp0)
    K = K(:,1:3*N);
end
Kt = K'*WTilde_Np1_L;
B = K-impcoeff*dt*M*BendForceDensityMat_L*K;
RHS = Kt*(fE+f_ext+fTw+M \ U0);
alphaU = lsqminnorm(Kt*M^(-1)*B,RHS);
Omega = reshape(alphaU(1:3*N),3,N)';

% Solve the theta ODE
OmegaDotXss = 1/Lfac*RNToPsi*sum((D_L*Xs3).*Omega,2);
ExtTorqOmega = Mrr*n_ext;
RHS = theta_s + dt*(-OmegaDotXss+DPsi_L*ExtTorqOmega);
theta_sp1 = ThetaImplicitMatrix_L \ (RHS+ThetaImpPart_L*[zeros(Npsi,1);PsiBC0;PsiBCL]);

% Update with an inextensible motion
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
if (length(alphaU) > 3*N)
    XMPor0_p1 = XMPor0+dt*alphaU(end-2:end);
else
    XMPor0_p1 = XMPor0;
end
Xp1 = XonNp1Mat*[Xsp1;XMPor0_p1];

% Add the polymerization velocity and recompute center
if (Lprime(iFib) > 0)
Lextra = dt*Lprime(iFib);
% Extend the tangent vector at s = L outwards
Xp13 = reshape(Xp1,3,[])';
TauLast = barymat(L,sNp1,bNp1)*DNp1*Xp13;
Xadded = TauLast/norm(TauLast)*Lextra+barymat(L,sNp1,bNp1)*Xp13;
% Solve for a new parameterization that ends at Xadded and goes through the
% other points
XWithAdd = [Xp13; Xadded];
sToEval = sNp1*Lfacs(iFib);
Lfacs(iFib)=Lfacs(iFib)+Lextra/L; % add the extra length
sToEval=[sToEval;L*Lfacs(iFib)]; % pts where we evaluate new interp (max is L*Lfacs)
Rnew = barymat(sToEval/Lfacs(iFib),sNp1,bNp1); % For the new parameterization [0,L]
X3new = pinv(Rnew)*XWithAdd;
% Find the "best" interpolant that satisfies BCs (if clamped)
%if (clamp0)
%    BCrows = [XBCMat_low(1,:); XBCMat_low(2,:)*Lfacs(iFib)];
%    X3New = pinv([BCrows; Rnew])*[X0BC(:,iFib)'; Tau0BC(:,iFib)';XWithAdd];
%end
NewConf = XonNp1Mat \ reshape(X3new',[],1);
Xsp1 = NewConf(1:3*N);
XMPor0_p1 = NewConf(3*N+1:end);
end

% Update first material frame vector (at 0)
if (updateFrame)
    nparTw = twmod*RTheta*DPsip2_L*theta_s_Psip2;
    OmegaPar = RPsiToN*(Mrr*nparTw+ExtTorqOmega);
    XsNormed = reshape(Xsp1,3,N)';
    XsNormed = XsNormed./sqrt(sum(XsNormed.*XsNormed,2));
    OmegaTot = Omega+OmegaPar.*XsNormed;
    OmegaMid_re = barymat(L/2,s,b)*OmegaTot;
    D1mid = rotate(D1mid,dt*OmegaMid_re);
    XBCNext =  UpsampleXBCMat_L*Xt + BCShift_L;
    XssNext = RNp5ToN*DNp5_L^2*reshape(XBCNext,3,N+5)';
    theta = (eye(N)-barymat(L/2,s,b))*pinv(D_L)*RPsiToN*theta_sp1;
    [bishA,bishB,D1next,D2next] = ...
        computeBishopFrame(N,XsNormed,XssNext,pinv(D_L),barymat(L/2,s,b),theta,D1mid');
else
    D1next = D1;
end