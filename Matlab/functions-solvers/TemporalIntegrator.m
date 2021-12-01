% Temporal integrator for fibers
Xp1 = zeros(3*N*nFib,1);
Xsp1 = zeros(3*N*nFib,1);
lambdas = zeros(3*N*nFib,1);
fE = zeros(3*N*nFib,1);
fEprev = zeros(3*N*nFib,1);
fTw = zeros(3*N*nFib,1);
nparTw = zeros(N*nFib,1);
nLvel = zeros(3*N*nFib,1);
dU = zeros(3*N*nFib,1);
UFromTorq = zeros(3*N*nFib,1);
OmegaPar_Euler = zeros(N*nFib,1);
l_m1 = 0*lambdas;
l_m = lamguess;  % put in the previous lambda as the initial guess
iters=0;
reler = 10;
nIts=1;
thetaarg=theta_s;
%while reler > 1e-6
% Precompute the relevant fE's 
for iFib=1:nFib
    inds = (iFib-1)*3*N+1:3*N*iFib;
    XBCprev = UpsampleXBCMat2*Xtm1(inds) + BCShift;
    XBC = UpsampleXBCMat2*Xt(inds) + BCShift;
    fE(inds) = FE*XBC;
    fEprev(inds) = FE*XBCprev;
    % Calculate twisting force according to BC
    theta_s_sp2 = UpsampleThetaBCMat \ [thetaarg; ThetaBC0; 0];
    %theta_s_sp2 = barymat(sp2,s,b)*thetaarg;
    Theta_ss = D_sp2*theta_s_sp2;
    XBC3 = reshape(XBC,3,N+4)';
    fTwb = twmod*((R4ToDouble*(R2To4*Theta_ss)).*(R4ToDouble*cross(D_sp4*XBC3,D_sp4^2*XBC3))+...
        (R4ToDouble*(R2To4*theta_s_sp2)).*(R4ToDouble*cross(D_sp4*XBC3,D_sp4^3*XBC3)));
    fTw3=RDoubleToN*fTwb;
    fTw(inds) = reshape(fTw3',3*N,1);
    nparTw((iFib-1)*N+1:N*iFib) = twmod*R2ToN*Theta_ss;
end
fEarg = fE;
if (Temporal_order==2)
    fEarg = 1.5*fE-0.5*fEprev;
end
Xsarg = Xst;
Xarg = Xt;
if (Temporal_order==2)
    Xsarg = 1.5*Xst-0.5*Xstm1;
    Xarg = 1.5*Xt-0.5*Xtm1;
end
% while ((nonLocal || iters==0) && reler > 1e-6 && iters < maxiters)
%     if (iters > 0) % use CN for nonlocal also if doing fixed point
%         for iFib = 1:nFib
%             inds = (iFib-1)*3*N+1:3*N*iFib;
%             fEarg(inds)=FE*0.5*(Xt(inds)+Xp1(inds));
%         end
%     end
%     if (nonLocal)
%         if (~Periodic)
%             nLvel = MNonLocalUnbounded(nFib,N,s,w,L,eps,reshape(fEarg+l_m+fext,3,N*nFib),...
%                  reshape(Xarg,3,N*nFib)',reshape(Xsarg,3,N*nFib)',Ds(1:3:3*N,1:3:3*N),mu,deltaLocal);
%         else
%             nLvel = MNonLocalPeriodic(nFib,N,s,w,L,eps,reshape(fEarg+l_m+fext,3,N*nFib),...
%                  reshape(Xarg,3,N*nFib)',reshape(Xsarg,3,N*nFib)',Ds(1:3:3*N,1:3:3*N),mu,...
%                  xi,Ld,Ld,Ld,strain,deltaLocal);
%         end
%         nLvel = reshape(nLvel',3*N*nFib,1);
%     end
impcoeff=1;
if (Temporal_order==2)
    impcoeff=0.5;
end
for iFib=1:nFib % block diagonal solve
    inds = (iFib-1)*3*N+1:3*N*iFib;
    scinds = (iFib-1)*N+1:N*iFib;
    X3 = reshape(Xarg(inds),3,N)';
    Xs3 = reshape(Xsarg(inds),3,N)';
    % Xss, Xsss with BCs!!
    XBC = UpsampleXBCMat2*Xt(inds) + BCShift;
    Xss = stackMatrix(R4ToN*D_sp4^2)*XBC;
    Xsss = stackMatrix(R4ToN*D_sp4^3)*XBC;
    Xss3 = reshape(Xss,3,N)';
    Xsss3 = reshape(Xsss,3,N)';
    if (eps > 1e-3)
        NForSmall = 8;
    else
        NForSmall = 4;
    end
    if (exactRPY)
        M = ExactRPYSpectralMobility(N,X3,Xs3,Xss3,Xsss3,a,L,mu,s,b,D,AllbS,AllbD,NForSmall);
        if (min(real(eig(M))) < 0 && N < 50)
            keyboard
        end
        [~, MtrLoc, MrtLoc, Mrr,~] = getGrandMloc(N,Xsarg(inds),Xss,...
            a,L,mu,s,deltaLocal);
    else
        [M, MtrLoc, MrtLoc, Mrr,~] = getGrandMloc(N,Xsarg(inds),Xss,...
            a,L,mu,s,deltaLocal);
        if (includeFP)
            M = M+StokesletFinitePartMatrix(X3,Xs3,D,s,L,N,mu,Allb_trueFP);
        end
    end
    if (~noRotTransAtAll)
        if (exactRPY)
            UFromTorq(inds) = reshape(upsampleRPYTransRotSmall(X3,Xs3,nparTw(scinds),s,b,...
                NForSmall,L,a,mu)',3*N,1);
            UFromTorq(inds)  = UFromTorq(inds)  + ...
                reshape(UFromNFPIntegral(X3,Xs3,Xss3,Xsss3,s,N,L,nparTw(scinds),D*nparTw(scinds),...
                AllbS,mu)',3*N,1);
            UFromTorq(inds) = UFromTorq(inds) +...
                getMlocRotlet(N,Xsarg(inds),Xss,a,L,mu,s,0)'*nparTw(scinds);
        else
            UFromTorq(inds) = MtrLoc*nparTw(scinds);
            if (nonLocalRotTrans)
                UFromTorq=UFromTorq+reshape(UFromNFPIntegral(X3,Xs3,Xss3,Xsss3,s,N,L,nparTw(scinds),...
                D*nparTw(scinds),Allb_trueFP,mu)',3*N,1);
            end
        end
    end
    [K,Kt,nPolys]=getKMats3DClampedNumer(Xsarg(inds),Lmat,w,N,I,wIt,'U',[clamp0 clampL]);
    [~,dimK]=size(K);
    B = K-impcoeff*dt*M*FE*(UpsampleXBCMat2*K);
    RHS = Kt*(fE(inds)+fext(inds)+fTw(inds)+M \ (UFromTorq(inds) + U0(inds) + nLvel(inds)));
    alphaU = lsqminnorm(Kt*M^(-1)*B,RHS);
    ut = K*alphaU;
    dU(inds) = Ds*ut;
    Xp1(inds) = Xt(inds)+dt*ut;
    Xsp1(inds) = Ds*Xp1(inds);
    l_m1(inds) = l_m(inds);
    l_m(inds) = M \ (ut-nLvel(inds)-U0(inds)-UFromTorq(inds))...
        -FE*(UpsampleXBCMat2*(impcoeff*Xp1(inds)+(1-impcoeff)*Xt(inds))+BCShift)-fext(inds)-fTw(inds);
    U2 = M*(FE*(UpsampleXBCMat2*Xp1+BCShift)+fTw+l_m)+UFromTorq;
    % Solve theta ODE (implicit)
    force = l_m(inds)+fTw(inds)+FE*(UpsampleXBCMat2*Xp1+BCShift);
    f3 = reshape(force,3,N)';
    RotFromTrans = zeros(N,1);
    if (~noRotTransAtAll)
        if (exactRPY)
            RotFromTrans = upsampleRPYRotTransSmall(X3,Xs3,f3,s,b,NForSmall,L,a,mu);
            RotFromTrans = RotFromTrans + OmFromFFPIntegral(reshape(Xt(inds),3,N)',...
                Xs3,Xss3,Xsss3,s,N,L,f3,D*f3,AllbS,mu); 
            RotFromTrans = RotFromTrans + getMlocRotlet(N,Xsarg(inds),Xss,a,L,mu,s,0)*force;
        else
            RotFromTrans =MrtLoc*force;
            if (nonLocalRotTrans)
                RotFromTrans=RotFromTrans+... % Omega dot tau
                OmFromFFPIntegral(reshape(Xt(inds),3,N)',Xs3,Xss3,Xsss3,s,N,L,f3,D*f3,Allb_trueFP,mu); 
            end
        end
    end
    Omega = cross(Xs3,reshape(dU,3,N)');
    OmegaDotXss = sum(Xss3.*Omega,2);
    % Solve for theta on N+2 grid
    RHS = theta_s + dt*(-OmegaDotXss+D*RotFromTrans);
    ThetaBC0 = 0;
    if (clamp0)
        ThetaBC0 = TurnFreq-barymat(0,s,b)*RotFromTrans; % omega^parallel(0)
    end
    theta_sp1 = ThetaImplicitMatrix \ (RHS+dt*D*Mrr*R2ToN*twmod*D_sp2*...
        (UpsampleThetaBCMat \ [zeros(N,1);ThetaBC0;0]));
%     theta_sp1 = [ThetaBCMat_low(1,:); ThetaImplicitMatrix(2:N-1,:); ThetaBCMat_low(2,:)] \ ...
%        [ThetaBC0; RHS(2:end-1); 0];
%     if (abs(theta_sp1(1)+5.34) > 0.1)
%        keyboard
%     end
end   
% reler = norm(theta_sp1-thetaarg);
% thetaarg = theta_sp1;
% nIts=nIts+1;
% end
%     reler = norm(l_m-l_m1)/(max([1 norm(l_m)]));
%     iters=iters+1;
%     if (iters ==25 && reler > 1e-6)
%         disp('Fixed point iteration not converging - change tolerance')
%     end
% end
% Update with an inextensible motion
for iFib=1:nFib
    inds = (iFib-1)*3*N+1:3*N*iFib;
    scinds = (iFib-1)*N+1:N*iFib;
    [newX,newXs,OmegaPerp] = updateX(Xt(inds),(Xp1(inds)-Xt(inds))/dt,N,dt,...
        L(iFib),Xst(inds),Xstm1(inds),dU(inds),Temporal_order);
    if (max(abs(Xp1(inds)-newX)) > 1e-3)
        max(abs(Xp1(inds)-newX))
    end
    Xp1(inds) = newX;
    Xsp1(inds) = newXs;
end
lambdas=l_m;
% Compute material frame (start of next time step) in two ways
% Update first material frame vector (at 0)
% nparTw((iFib-1)*N+1:N*iFib) = twmod*R2ToN*D_sp2*theta_s_sp2;
% OmegaPar = RotFromTrans+Mrr*nparTw;
% OmegaTot = OmegaPerp+OmegaPar.*reshape(Xsp1,3,N)';
% OmegaPar0_re = barymat(0,s,b)*OmegaPar;
% OmegaMid_re = barymat(L/2,s,b)*OmegaTot;
% Xsmid = barymat(L/2,s,b)*reshape(Xsp1,3,N)';
% D1mid = rotate(D1mid,dt*OmegaMid_re);
% % 1) Rotate Bishop frame
% XBCNext =  UpsampleXBCMat2*Xt(inds) + BCShift;
% XssNext = reshape(stackMatrix(R4ToN*D_sp4^2)*XBC,3,N)';
% theta = (eye(N)-barymat(L/2,s,b))*pinv(D)*theta_sp1;
% [bishA,bishB,D1next,D2next] = computeBishopFrame(N,reshape(Xsp1,3,N)',XssNext,s,b,L,theta,D1mid');
% % Rotate material frame by (Omega+OmegaParallel)*dt
% D1next2 = 0*D1next;
% D2next2 = 0*D2next;
% for iPt=1:N
%     D1next2(iPt,:) = rotate(D1(iPt,:),dt*OmegaTot(iPt,:));
%     D2next2(iPt,:) = rotate(D2(iPt,:),dt*OmegaTot(iPt,:));
% end