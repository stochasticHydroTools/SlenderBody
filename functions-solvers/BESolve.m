% Backward Euler
function [Xp1,lambdas,fE,Xsp1] = BESolve(nFib,N,Ms,Ks,Kts,I,wIt,FE,LRLs,URLs,Xt,Xs,...
                                U0,dt,s0,w0,Lf,epsilon,Dmat,mu,xi,L,...
                                nonLocal,lamprev,maxiters,grav,fext,strain)
    % Schur complement solve fiber by fiber
    Xp1 = zeros(3*N*nFib,1);
    Xsp1 = zeros(3*N*nFib,1);
    lambdas = zeros(3*N*nFib,1);
    allalphas = zeros((2*N-2)*nFib,1);
    fE = zeros(3*N*nFib,1);
    nLvel = zeros(3*N*nFib,1);
    l_m1 = 0*lambdas;
    l_m = lamprev;  % put in the previous lambda as the initial guess
    iters=0;
    reler = 10;
    % Precompute the relevant fE's 
    for iFib=1:nFib
        inds = (iFib-1)*3*N+1:3*N*iFib;
        fE(inds) = FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt(inds)));
    end
    while ((nonLocal || iters==0) && reler > 1e-6 && iters < maxiters)
        if (nonLocal)
            %nLvel = MNonLocalSlow(nFib,N,s0,w0,Lf,epsilon,reshape(fE+l_m+fext,3,N*nFib),...
            %     reshape(Xt,3,N*nFib)',reshape(Xs,3,N*nFib)',Dmat(1:3:3*N,1:3:3*N),mu);
            nLvel = MNonLocal(nFib,N,s0,w0,Lf,epsilon,reshape(fE+l_m+fext,3,N*nFib),...
                     reshape(Xt,3,N*nFib)',reshape(Xs,3,N*nFib)',...
                     Dmat(1:3:3*N,1:3:3*N),mu,xi,L,L,L,strain);
            nLvel = reshape(nLvel',3*N*nFib,1);
        end
        for iFib=1:nFib
            inds = (iFib-1)*3*N+1:3*N*iFib;
            M = Ms{iFib};   K = Ks{iFib};   Kt = Kts{iFib};
            B=[K-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
                I-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
            C=[Kt; wIt];
            D=[zeros(2*N-2) zeros(2*N-2,3); ...
                dt*wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
                wIt*dt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
            RHS = C*fE(inds)+C*M^(-1)*U0(inds)+C*fext(inds)+...
                [Kt*repmat([0;0;grav/Lf],N,1); -wIt*fE(inds)+[0;0;grav]] + C*M^(-1)*nLvel(inds);
            alphaU = lsqminnorm(C*M^(-1)*B+D,RHS);
            alphas=alphaU(1:2*N-2);
            allalphas((2*N-2)*(iFib-1)+1:(2*N-2)*iFib)=alphas;
            Urigid=alphaU(2*N-1:2*N+1);
            ut = K*alphas+I*Urigid;
            Xp1(inds) = Xt(inds)+dt*ut;
            Xsp1(inds) = Dmat*Xp1(inds);
            l_m1(inds) = l_m(inds);
            l_m(inds) = M \ (K*alphas+I*Urigid-nLvel(inds)...
                    -M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xp1(inds)))...
                    -M*fext(inds)-U0(inds));
        end
        reler = norm(l_m-l_m1)/(max([1 norm(l_m)]));
        iters=iters+1;
    end
    for iFib=1:nFib
        % Update with an inextensible motion
        inds = (iFib-1)*3*N+1:3*N*iFib;
        [newX,newXs] = updateX(allalphas((2*N-2)*(iFib-1)+1:(2*N-2)*iFib),...
            Xt(inds),(Xp1(inds)-Xt(inds))/dt,N,dt,Lf,Xs(inds),Xs(inds));
        Xp1(inds) = newX;
        Xsp1(inds) = newXs;
    end
    lambdas=l_m;
end