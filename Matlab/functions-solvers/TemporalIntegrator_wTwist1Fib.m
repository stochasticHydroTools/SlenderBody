% Temporal integrator for fibers that resist bend and twist
% Only ONE fiber for this 
% Precompute the relevant forces using the correct BCs
thetaarg=theta_s;
XBC = UpsampleXBCMat*Xt + BCShift;
fE = FE*XBC;
% Calculate twisting force according to BC
theta_s_Psip2= UpsampleThetaBCMat \ [thetaarg; PsiBC0; 0];
Theta_ss = DPsip2*theta_s_Psip2;
XBC3 = reshape(XBC,3,[])';
% Twist force computed on common N+5 grid, then downsampled to Nx
fTw3 = twmod*((RPsiToNp5*Theta_ss).*cross(DNp5*XBC3,DNp5^2*XBC3)+...
    (RPsiToNp5*theta_s_Psip2).*cross(DNp5*XBC3,DNp5^3*XBC3));
fTw = reshape((RNp5ToNp1*fTw3)',3*Nx,1);
nparTw = twmod*RPsiToNp1*Theta_ss+RToNp1*RPsiToN*n_ext;
% Arguments for the solver, assuming first order
impcoeff=1;
% Saddle point solve
X3 = reshape(Xt,3,Nx)';
Xs3 = reshape(Xst,3,N)';
M = TransTransMobilityMatrix(X3,a,L,mu,sNp1,bNp1,DNp1,AllbS_Np1,...
    AllbD_Np1,NForSmall,~exactRPY,deltaLocal);
% Calculate velocity from torque for RHS
UFromTorq = zeros(3*Nx,1);
if (~noRotTransAtAll)
    UFromTorq = UFromN(X3,nparTw,DNp1,AllbS_Np1,a,L,mu,sNp1,bNp1,nonLocalRotTrans,NForSmall);
    UFromTorq = reshape(UFromTorq',[],1);
end
% Solve for fiber evolution
K = KonNp1(Xs3,XonNp1Mat,I);
if (clamp0)
    K = K(:,1:3*N);
end
Kt = K'*WTilde_Np1;
B = K-impcoeff*dt*M*FE*(UpsampleXBCMat*K);
RHS = Kt*(fE+fTw+f_ext+M \ (UFromTorq+ U0));
alphaU = lsqminnorm(Kt*M^(-1)*B,RHS);
Omega = reshape(alphaU(1:3*N),3,N)';
ut = K*alphaU;
Xp1 = Xt + dt*ut;
l_m = M \ (K*alphaU-U0-UFromTorq)...
    -FE*(UpsampleXBCMat*(impcoeff*Xp1+(1-impcoeff)*Xt)+BCShift)-fTw;
U2 = M*(FE*(UpsampleXBCMat*Xp1+BCShift)+fTw+f_ext+l_m)+UFromTorq+U0;

% Solve theta ODE 
force = l_m + fTw + f_ext + FE*(UpsampleXBCMat*Xp1+BCShift);
f3 = reshape(force,3,[])';
RotFromTrans = zeros(Npsi,1);
if (~noRotTransAtAll)
    RotFromTrans = OmegaFromF(X3,f3,DNp1,AllbS_Np1,a,L,mu,sNp1,bNp1,nonLocalRotTrans,NForSmall);
    RotFromTrans = RNp1ToPsi*RotFromTrans;
end
OmegaDotXss = RNToPsi*sum((D*Xs3).*Omega,2);
ExtTorqOmega = Mrr*n_ext;
RHS = theta_s + dt*(-OmegaDotXss+DPsi*RotFromTrans+DPsi*ExtTorqOmega);
ThetaBC0 = 0;
if (clamp0)
    ThetaBC0= TurnFreq-barymat(0,sPsi,bPsi)*RotFromTrans; % omega^parallel(0)
    if (TorqBC)
        ThetaBC0 = -wPsi*diag(Mrr^(-1))*TurnFreq/twmod;
    end
end
theta_sp1 = ThetaImplicitMatrix \ (RHS+ThetaImpPart*[zeros(Npsi,1);ThetaBC0;0]);
% Update with an inextensible motion
newXs = rotateTau(Xs3,Omega,dt);
Xsp1 = reshape(newXs',[],1);
if (length(alphaU) > 3*N)
    XMPor0_p1 = XMPor0+dt*alphaU(end-2:end);
else
    XMPor0_p1 = XMPor0;
end
Xp1 = XonNp1Mat*[Xsp1;XMPor0_p1];
lambdas=l_m;
% Compute material frame (start of next time step)
% Update first material frame vector (at 0)
if (updateFrame)
    nparTw = twmod*RTheta*DPsip2*theta_s_Psip2;
    OmegaPar = RPsiToN*(RotFromTrans+Mrr*nparTw+ExtTorqOmega);
    OmegaTot = Omega+OmegaPar.*reshape(Xsp1,3,N)';
    OmegaMid_re = barymat(L/2,s,b)*OmegaTot;
    D1mid = rotate(D1mid,dt*OmegaMid_re);
    XssNext = D*reshape(Xsp1,3,N)';
    theta = (eye(N)-barymat(L/2,s,b))*pinv(D)*RPsiToN*theta_sp1;
    [bishA,bishB,D1next,D2next] = ...
        computeBishopFrame(N,reshape(Xsp1,3,N)',XssNext,pinv(D),barymat(L/2,s,b),theta,D1mid');
else
    D1next = D1;
end