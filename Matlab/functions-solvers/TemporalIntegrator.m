% Temporal integrator for fibers that resist bend and twist
l_m1 = 0*lambdas;
l_m = lamguess;  % put in the previous lambda as the initial guess
iters=0;
reler = 10;
nIts=1;
thetaarg=theta_s;
Xp1old = Xt;
%while reler > 1e-6
% Precompute the relevant forces using the correct BCs
for iFib=1:nFib
    inds = (iFib-1)*3*N+1:3*N*iFib;
    np4inds = (iFib-1)*3*(N+4)+1:3*(N+4)*iFib;
    scinds = (iFib-1)*N+1:N*iFib;
    XBCprev(np4inds) = UpsampleXBCMat*Xtm1(inds) + BCShift;
    XBC(np4inds) = UpsampleXBCMat*Xt(inds) + BCShift;
    fE(inds) = FE*XBC(np4inds);
    fEprev(inds) = FE*XBCprev(np4inds);
    % Calculate twisting force according to BC
    if (strongthetaBC)
        theta_s_sp2 = barymat(sp2,s,b)*thetaarg(scinds);
    else
        theta_s_sp2 = UpsampleThetaBCMat \ [thetaarg(scinds); ThetaBC0; 0];
    end
    Theta_ss = D_sp2*theta_s_sp2;
    XBC3 = reshape(XBC(np4inds),3,[])';
    fTwb = twmod*((R4ToDouble*(R2To4*Theta_ss)).*(R4ToDouble*cross(D_sp4*XBC3,D_sp4^2*XBC3))+...
        (R4ToDouble*(R2To4*theta_s_sp2)).*(R4ToDouble*cross(D_sp4*XBC3,D_sp4^3*XBC3)));
    fTw3=RDoubleToN*fTwb;
    fTw(inds) = reshape(fTw3',3*N,1);
    nparTw(scinds) = twmod*R2ToN*Theta_ss;
end
% Arguments for the solver, assuming first order
fEarg = fE;
Xsarg = Xst;
Xarg = Xt;
XBCarg = XBC;
impcoeff=1;
% Extrapolations that give second order accuracy
if (Temporal_order==2)
    fEarg = 1.5*fE-0.5*fEprev;
    Xsarg = 1.5*Xst-0.5*Xstm1;
    Xarg = 1.5*Xt-0.5*Xtm1;
    XBCarg = 1.5*XBC-0.5*XBCprev;
    impcoeff = 0.5;
end
nonLocal = interFiberHydro || includeFPonRHS;
while ((nonLocal || iters==0) && reler > 1e-6 && iters < (1000*(count < Temporal_order)+maxiters))
if (iters > 0) % use implicit for nonlocal also if doing fixed point
    for iFib = 1:nFib
        inds = (iFib-1)*3*N+1:3*N*iFib;
        np4inds = (iFib-1)*3*(N+4)+1:3*(N+4)*iFib;
        XBCp1 = UpsampleXBCMat*Xp1(inds) + BCShift;
        if (Temporal_order==2)
            fEarg(inds)=FE*0.5*(XBC(np4inds)+XBCp1);
        else
            fEarg(inds)=FE*(XBCp1);
        end
    end
end
if (nonLocal) % Nonlocal flows which will go on the RHS of the block diagonal solve
    Xssarg3 = zeros(N*nFib,3);
    for iFib=1:nFib
        scinds = (iFib-1)*N+1:iFib*N;
        np4inds = (iFib-1)*3*(N+4)+1:3*(N+4)*iFib;
        Xssarg3(scinds,:) = reshape(stackMatrix(R4ToN*D_sp4^2)*XBCarg(np4inds),3,N)';
    end
    fNLarg =reshape(fEarg+l_m+fext,3,N*nFib)';
    if (Periodic)
        nLvel = MNonLocalPeriodic(nFib,N,s,b,L,fNLarg,reshape(Xarg,3,N*nFib)',reshape(Xsarg,3,N*nFib)',Xssarg3,...
            a,D,mu,includeFPonRHS,Allb_trueFP,NupsampleforNL, splittingparam,Ld,strain);
    else
        nLvel = MNonLocalUnbounded(nFib,N,s,b,L,fNLarg,reshape(Xarg,3,N*nFib)',...
            reshape(Xsarg,3,N*nFib)',Xssarg3,a,D,mu,includeFPonRHS,Allb_trueFP,NupsampleforNL);
    end
    nLvel = reshape(nLvel',3*N*nFib,1);
end
for iFib=1:nFib % block diagonal solve
    inds = (iFib-1)*3*N+1:3*N*iFib;
    scinds = (iFib-1)*N+1:N*iFib;
    np4inds = (iFib-1)*3*(N+4)+1:3*(N+4)*iFib;
    X3 = reshape(Xarg(inds),3,N)';
    Xs3 = reshape(Xsarg(inds),3,N)';
    % Xss, Xsss with BCs!!
    Xss = stackMatrix(R4ToN*D_sp4^2)*XBCarg(np4inds);
    Xsss = stackMatrix(R4ToN*D_sp4^3)*XBCarg(np4inds);
    Xss3 = reshape(Xss,3,N)';
    Xsss3 = reshape(Xsss,3,N)';
    if (exactRPY)
        %disp('Matlab: using D*Xs in RPY mobility')
        M = ExactRPYSpectralMobility(N,X3,Xs3,Xss3,Xsss3,a,L,mu,s,b,D,AllbS,AllbD,NForSmall);
        if (min(real(eig(M))) < 0)
            warning('Negative eigenvalues in M; refine discretization or check fiber shape!')
        end
        [~, ~, ~, Mrr,~] = getGrandMloc(N,Xsarg(inds),Xss,a,L,mu,s,0); % no regularization w exact RPY
        if (bottomwall)
            % Subtract singular part for each s
            Xup = Rglobalupsample*reshape(Xt,3,N)';
            McorUp = single_wall_mobilityCor_trans_times_force(Xup,mu,a);
            Mtt_cor=stackMatrix(Rglobaldown)*McorUp*Wmat*stackMatrix(Rglobalupsample);
            M = M + Mtt_cor;
        end
    else % local drag + finite part
        [M, MtrLoc, MrtLoc, Mrr,~] = getGrandMloc(N,Xsarg(inds),Xss,a,L,mu,s,deltaLocal);
        if (includeFPonLHS)
            %disp('Matlab: using D*Xs in finite part integral')
            M = M+StokesletFinitePartMatrix(X3,Xs3,Xss3,D,s,L,N,mu,Allb_trueFP);
        end
    end
    % Calculate velocity from torque for RHS
    if (~noRotTransAtAll)
        if (exactRPY)
            U3FromTorq = upsampleRPYTransRotSmall(X3,Xs3,nparTw(scinds),s,b,NForSmall,L,a,mu)+...
                UFromNFPIntegral(X3,Xs3,Xss3,Xsss3,s,N,L,nparTw(scinds),D*nparTw(scinds),AllbS,mu);
            UFromTorq(inds) = reshape(U3FromTorq',3*N,1);
            UFromTorq(inds) = UFromTorq(inds) + getMlocRotlet(N,Xsarg(inds),Xss,a,L,mu,s,0)'*nparTw(scinds);
            if (bottomwall)
                npar_globalup = (Rglobalupsample*reshape(Xsarg(inds),3,N)').*(Rglobalupsample*nparTw(scinds));
                npar_globalup = npar_globalup.*wup';
                WallVelTorq = Rglobaldown*single_wall_mobilityCor_trans_times_torque(Xup,npar_globalup,mu,a);
                UFromTorq(inds) = UFromTorq(inds) + reshape(WallVelTorq',3*N,1);
            end
        else
            UFromTorq(inds) = MtrLoc*nparTw(scinds);
            if (nonLocalRotTrans)
                UFromTorq(inds)=UFromTorq(inds)+reshape(UFromNFPIntegral(X3,Xs3,Xss3,Xsss3,s,N,L,nparTw(scinds),...
                D*nparTw(scinds),Allb_trueFP,mu)',3*N,1);
            end
        end
    end
    % Solve for fiber evolution
    [K,Kt,nPolys]=getKMats3DClampedNumer(Xsarg(inds),Lmat,w,N,I,wIt,'U',[clamp0 clampL]);
    [~,dimK]=size(K);
    B = K-impcoeff*dt*M*FE*(UpsampleXBCMat*K);
    RHS = Kt*(fE(inds)+fext(inds)+fTw(inds)+M \ (UFromTorq(inds) + U0(inds) + nLvel(inds)));
    alphaU = lsqminnorm(Kt*M^(-1)*B,RHS);
    ut = K*alphaU;
    dU(inds) = Ds*ut;
    Xp1(inds) = Xt(inds)+dt*ut;
    Xsp1(inds) = Ds*Xp1(inds);
    l_m1(inds) = l_m(inds);
    l_m(inds) = M \ (ut-nLvel(inds)-U0(inds)-UFromTorq(inds))...
        -FE*(UpsampleXBCMat*(impcoeff*Xp1(inds)+(1-impcoeff)*Xt(inds))+BCShift)-fext(inds)-fTw(inds);
    U2 = M*(FE*(UpsampleXBCMat*Xp1(inds)+BCShift)+fTw(inds)+l_m(inds)+fext(inds))+UFromTorq(inds);
    % Solve theta ODE 
    force = l_m(inds)+fTw(inds)+FE*(UpsampleXBCMat*Xp1(inds)+BCShift);
    f3 = reshape(force,3,N)';
    RotFromTrans = zeros(N,1);
    if (~noRotTransAtAll)
        if (exactRPY)
            RotFromTrans = upsampleRPYRotTransSmall(X3,Xs3,f3,s,b,NForSmall,L,a,mu);
            RotFromTrans = RotFromTrans + OmFromFFPIntegral(reshape(Xt(inds),3,N)',...
                Xs3,Xss3,Xsss3,s,N,L,f3,D*f3,AllbS,mu); 
            RotFromTrans = RotFromTrans + getMlocRotlet(N,Xsarg(inds),Xss,a,L,mu,s,0)*force;
            if (bottomwall)
                f_globalup = (Rglobalupsample*f3).*wup';
                OmWallFull = Rglobaldown*single_wall_mobilityCor_rot_times_force(Xup,f_globalup,mu,a);
                OmParWall = sum(OmWallFull.*Xs3,2);
                RotFromTrans = RotFromTrans +OmParWall;
            end
        else
            RotFromTrans =MrtLoc*force;
            if (nonLocalRotTrans)
                RotFromTrans=RotFromTrans+... % Omega dot tau
                OmFromFFPIntegral(reshape(Xt(inds),3,N)',Xs3,Xss3,Xsss3,s,N,L,f3,D*f3,Allb_trueFP,mu); 
            end
        end
    end
    Omega = cross(Xs3,reshape(dU(inds),3,N)');
    OmegaDotXss = sum(Xss3.*Omega,2);
    extrawallterm = zeros(N,1);
    if (bottomwall)
        npar_globalup = (Rglobalupsample*reshape(Xsarg(inds),3,N)').*(Rglobalupsample*nparTw(scinds));
        npar_globalup = npar_globalup.*wup';
        OmFull_nparWall = Rglobaldown*single_wall_mobilityCor_rot_times_torque(Xup,npar_globalup,mu,a);
        extrawallterm = sum(OmFull_nparWall.*Xs3,2);
    end
    RHS = theta_s(scinds) + dt*(-OmegaDotXss+D*RotFromTrans+D*extrawallterm);
    ThetaBC0 = 0;
    if (clamp0)
        ThetaBC0 = TurnFreq-barymat(0,s,b)*(RotFromTrans+extrawallterm); % omega^parallel(0)
        if (TorqBC)
            ThetaBC0 = -w*diag(Mrr^(-1))*TurnFreq/twmod;
        end
    end
    if (strongthetaBC)
        theta_sp1(scinds) = [ThetaBCMat_low(1,:); ThetaImplicitMatrix(2:N-1,:); ThetaBCMat_low(2,:)] \ ...
           [ThetaBC0; RHS(2:end-1); 0];
    else
        theta_sp1(scinds) = ThetaImplicitMatrix \ (RHS+dt*D*Mrr*R2ToN*twmod*D_sp2*...
            (UpsampleThetaBCMat \ [zeros(N,1);ThetaBC0;0]));
    end
end
reler = norm(l_m-l_m1)/(max([1 norm(l_m)]));
iters=iters+1;
if (iters ==25 && reler > 1e-6)
    disp('Fixed point iteration not converging - change tolerance')
end
end
% Update with an inextensible motion
for iFib=1:nFib
    inds = (iFib-1)*3*N+1:3*N*iFib;
    scinds = (iFib-1)*N+1:N*iFib;
    [newX,newXs,OmegaPerp] = updateX(Xt(inds),(Xp1(inds)-Xt(inds))/dt,N,dt,...
        L,Xst(inds),Xstm1(inds),dU(inds),Temporal_order);
    if (max(abs(Xp1(inds)-newX)) > 1e-3)
        max(abs(Xp1(inds)-newX))
        %keyboard
    end
    Xp1(inds) = newX;
    Xsp1(inds) = newXs;
end
lambdas=l_m;
% Compute material frame (start of next time step)
% Update first material frame vector (at 0)
nparTw((iFib-1)*N+1:N*iFib) = twmod*R2ToN*D_sp2*theta_s_sp2;
OmegaPar = RotFromTrans+Mrr*nparTw;
OmegaTot = OmegaPerp+OmegaPar.*reshape(Xsp1,3,N)';
OmegaPar0_re = barymat(0,s,b)*OmegaPar;
OmegaMid_re = barymat(L/2,s,b)*OmegaTot;
Xsmid = barymat(L/2,s,b)*reshape(Xsp1,3,N)';
D1mid = rotate(D1mid,dt*OmegaMid_re);
% 1) Compute Bishop frame and rotate it
XBCNext =  UpsampleXBCMat*Xt(inds) + BCShift;
XssNext = reshape(stackMatrix(R4ToN*D_sp4^2)*XBC,3,N)';
theta = (eye(N)-barymat(L/2,s,b))*pinv(D)*theta_sp1;
[bishA,bishB,D1next,D2next] = computeBishopFrame(N,reshape(Xsp1,3,N)',XssNext,s,b,L,theta,D1mid');