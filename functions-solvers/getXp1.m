% Function that gives Xp1 for the different solvers
function [Xnp1,Xtm1,lambdas,fE,fext,lambdasm1,links,Xsp1,Xstm1] = getXp1(Xt,Xtm1,Xst,Xstm1,...
    lambdas,lambdalast,LMM,imax,solver,dt,FE,LRLs,URLs,Ds,nFib,N,eps,mu,chebyshevmat,Dinv,...
    s0,w0,Lf,nonLocal,I,wIt,gam0,omega,t,grav,nCL,links,Kspring,rl,L,xi)
    Ms = cell(nFib,1);    Ks = cell(nFib,1);    Kts = cell(nFib,1);
    if (solver == 2) % For Crank-Nicolson
        for iFib=1:nFib
            inds = (iFib-1)*3*N+1:iFib*3*N;
            Ms{iFib} = getMloc(N,1.5*Xst(inds)-0.5*Xstm1(inds),eps,Lf,mu,s0);
            % Now need the communication with the alphas
            [K,Kt]=getKMats3D(1.5*Xst(inds)-0.5*Xstm1(inds),chebyshevmat,Dinv,w0,N);
            Ks{iFib} = K; Kts{iFib}=Kt;
        end
        % Background flow, strain external force
        [~,gnetupdate] = EvalU0(gam0,omega,t,Xt,L);
        [U0,g] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1,L);
        % Making and breaking CL links over step dt/2
        links = breakCLinks(reshape(Xt,3,N*nFib)',...
            links,nFib,N,rl,Lf,L,gnetupdate);
        links = makeCLinks(links,reshape(Xt,3,N*nFib)',...
            nFib,N,rl,Lf,nCL,L,gnetupdate);
        fext = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',...
            N,s0,w0,Lf, Kspring,rl,g,L)',3*N*nFib,1);
    else % other explicit / first order solvers
        for iFib=1:nFib
            inds = (iFib-1)*3*N+1:iFib*3*N;
            %Xst(inds) = Ds*Xt(inds);
            Ms{iFib} = getMloc(N,Xst(inds),eps,Lf,mu,s0);
            % Now need the communication with the alphas
            [K,Kt]=getKMats3D(Xst(inds),chebyshevmat,Dinv,w0,N);
            Ks{iFib} = K; Kts{iFib}=Kt;
        end
        [U0,g] = EvalU0(gam0,omega,t,Xt,L);
        % Making and breaking CL links
        links = breakCLinks(reshape(Xt,3,N*nFib)',...
            links,nFib,N,rl,Lf,L,g,dt);
        links = makeCLinks(links,reshape(Xt,3,N*nFib)',nFib,N,rl,Lf,nCL,L,g,dt);
        fext = reshape(getCLforce(links,reshape(Xt,3,N*nFib)',...
            N,s0,w0,Lf, Kspring,rl,g,L)',3*N*nFib,1);
    end
    maxIters = 1;
    if (t > 0)
        maxIters=imax;
    end
    % Temporal integration
    if (solver == -1) % FE
        lambdasm1=lambdas;
        [u, lambdas, fE] = ExSolve(nFib,N,Ms,Ks,Kts,I,wIt,FE,LRLs,URLs,Xt,Xst,...
            U0,s0,w0,Lf,eps,Ds,mu,xi,L,nonLocal,lambdas,maxIters,grav,fext); 
        Xnp1 = Xt+u*dt;
    elseif (solver == -3) %RK3
        % To be fixed to add non-local functionality
        error('RK3 not currently functional - use FE for explicit method')
%         U01 = U0Strength(gam0,mu,w,t)*U0Mat*Xt;
%         [x1,f1] = ExSolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt,U01);
%         M2 = getM(N,Ds*(Xt+dt*x1),Oone,eps,mu,s0,w0);
%         [K2, Kt2]=getKMats3D(Ds*(Xt+dt*x1),chebyshevmat,Dinv,w0,N);
%         U02 = U0Strength(gam0,mu,w,t+dt)*U0Mat*(Xt+dt*x1);
%         [x2,f2] = ExSolve(N,M2,K2,Kt2,I,wIt,FE,LRLs,URLs,Xt+dt*x1,U02);
%         M3 = getM(N,Ds*(Xt+dt/4*x1+dt/4*x2),Oone,eps,mu,s0,w0);
%         [K3, Kt3]=getKMats3D(Ds*(Xt+dt/4*x1+dt/4*x2),chebyshevmat,Dinv,w0,N);
%         U03 = U0Strength(gam0,mu,w,t+dt/2)*U0Mat*(Xt+dt/4*x1+dt/4*x2);
%         [x3,f3] = ExSolve(N,M3,K3,Kt3,I,wIt,FE,LRLs,URLs,Xt+dt/4*x1+dt/4*x2,U03);
%         Xnp1=Xt+dt*(x1/6+x2/6+2*x3/3);
%         force = f1/6+f2/6+2*f3/3;
    elseif (solver==1) % BE
        lambdasm1 = lambdas;
        [Xnp1,lambdas,fE,Xsp1]=BESolve(nFib,N,Ms,Ks,Kts,I,wIt,FE,LRLs,URLs,Xt,Xst,...
                U0,dt,s0,w0,Lf,eps,Ds,mu,xi,L,nonLocal,lambdas,maxIters,grav,fext,g);
        Xstm1=Xsp1;
    elseif (solver==2) % CN
        if (t < 1.5*dt)
            maxIters=1;
        end
        lambdasm1=lambdas;
        if (t > 1.5*dt && LMM)
            lamguess = 2*lambdas-lambdalast;
        else
            lamguess = lambdas;
        end
        [Xnp1,lambdas,fE,Xsp1]=CNSolve(nFib,N,Ms,Ks,Kts,I,wIt,FE,LRLs,URLs,Xt,Xtm1,...
            Xst,Xstm1,U0,dt,s0,w0,Lf,eps,Ds,mu,xi,L,nonLocal,lamguess,maxIters,grav,fext,g);
    end
    Xtm1=Xt;
    Xstm1=Xst;
end