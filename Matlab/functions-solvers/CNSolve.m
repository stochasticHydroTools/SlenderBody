% Crank Nicolson / AB2 hybrid
function [Xp1,lambdas,fEstar,Xsp1] = CNSolve(nFib,N,deltaLoc,chebyshevmat,I,wIt,FE,Xt,...
    Xtm1,Xs,Xsm1,U0,dt,s0,w0,Lf,epsilon,Dmat,mu,xi,L,nonLocal,lamprev,maxiters,fext,strain)
    global Periodic doFP;
    % Schur complement solve fiber by fiber
    Xp1 = zeros(3*N*nFib,1);
    Xsp1 = zeros(3*N*nFib,1);
    lambdas = zeros(3*N*nFib,1);
    fE = zeros(3*N*nFib,1);
    fEprev = zeros(3*N*nFib,1);
    nLvel = zeros(3*N*nFib,1);
    dU = zeros(3*N*nFib,1);
    l_m1 = 0*lambdas;
    l_m = lamprev;  % put in the previous lambda as the initial guess
    iters=0;
    reler = 10;
    % Precompute the relevant fE's 
    for iFib=1:nFib
        inds = (iFib-1)*3*N+1:3*N*iFib;
        fE(inds)=FE*Xt(inds);
        fEprev(inds)=FE*Xtm1(inds);
    end
    fEarg = 1.5*fE-0.5*fEprev;
    fEstar = fEarg;
    Xsarg = 1.5*Xs-0.5*Xsm1;
    Xarg = 1.5*Xt-0.5*Xtm1;
    while ((nonLocal || iters==0) && reler > 1e-6 && iters < maxiters)
        if (iters > 0) % use CN for nonlocal also if doing fixed point
            for iFib = 1:nFib
                inds = (iFib-1)*3*N+1:3*N*iFib;
                fEarg(inds)=FE*0.5*(Xt(inds)+Xp1(inds));
            end
        end
        if (nonLocal)
            if (~Periodic)
                nLvel = MNonLocalUnbounded(nFib,N,s0,w0,Lf,epsilon,reshape(fEarg+l_m+fext,3,N*nFib),...
                     reshape(Xarg,3,N*nFib)',reshape(Xsarg,3,N*nFib)',Dmat(1:3:3*N,1:3:3*N),mu,deltaLoc);
            else
                nLvel = MNonLocalPeriodic(nFib,N,s0,w0,Lf,epsilon,reshape(fEarg+l_m+fext,3,N*nFib),...
                     reshape(Xarg,3,N*nFib)',reshape(Xsarg,3,N*nFib)',Dmat(1:3:3*N,1:3:3*N),mu,xi,L,L,L,strain);
            end
            nLvel = reshape(nLvel',3*N*nFib,1);
        end
        for iFib=1:nFib % block diagonal solve
            inds = (iFib-1)*3*N+1:3*N*iFib;
            M = getMloc(N,Xsarg(inds),epsilon,Lf,mu,s0,deltaLoc);
            %M = getMlocRPY(N,Xsarg(inds),1.1204*epsilon*Lf,Lf,mu,s0);
%             if (~doFP)
%                 % Form upsampled RPY matrix column by column
%                 M = zeros(3*N);
%                 for iC=1:3*N
%                     f = zeros(3*N,1);
%                     f(iC) = 1;
%                     [sc,~,b0]=chebpts(N,[0 Lf],1);
%                     U = 1/(8*pi*mu)*upsampleRPY(reshape(Xarg,3,N*nFib)',s0,reshape(Xarg,3,N*nFib)',...
%                         reshape(f,3,N)',s0,b0,200,Lf,exp(1.5)/4*epsilon*Lf);
%                     M(:,iC) = reshape(U',3*N,1);
%                 end
%             end
            [K,Kt]=getKMats3D(Xsarg(inds),chebyshevmat,w0,N,'U');
            %[K,Kt]=getKMats3DLimited(Xsarg(inds),chebyshevmat,w0,N);
            K = [K I];   Kt = [Kt; wIt];
            B = K-0.5*dt*M*FE*K;
            RHS = Kt*fE(inds)+Kt*fext(inds)+Kt*M^(-1)*(U0(inds) + nLvel(inds));
            alphaU = lsqminnorm(Kt*M^(-1)*B,RHS);
            ut = K*alphaU;
            dU(inds) = Dmat*ut;
            Xp1(inds) = Xt(inds)+dt*ut;
            Xsp1(inds) = Dmat*Xp1(inds);
            l_m1(inds) = l_m(inds);
            l_m(inds) = M \ (ut-nLvel(inds)-U0(inds))-FE*0.5*(Xp1(inds)+Xt(inds))-fext(inds);
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
        [newX,newXs] = updateX(Xt(inds),(Xp1(inds)-Xt(inds))/dt,N,dt,...
            Lf,Xs(inds),Xsm1(inds),dU(inds),2);
        if (max(abs(Xp1(inds)-newX)) > 1e-3)
%             keyboard
            max(abs(Xp1(inds)-newX))
        end
        Xp1(inds) = newX;
        Xsp1(inds) = newXs;
    end
    lambdas=l_m;
end