% Function that gives Xp1 
function [Xnp1,Xtm1,lambdas,fE,fext,lambdasm1,links,Xsp1,Xstm1] = getXp1(Xt,Xtm1,Xst,Xstm1,...
    lambdas,lambdalast,imax,dt,FE,LRLs,URLs,Ds,nFib,N,eps,mu,chebyshevmat,Dinv,...
    s0,w0,Lf,nonLocal,I,wIt,gam0,omega,t,grav,nCL,links,Kspring,rl,L,xi)
    Ms = cell(nFib,1);    Ks = cell(nFib,1);    Kts = cell(nFib,1);
    % Assuming Crank-Nicolson solver
    for iFib=1:nFib % Compute matrices M and K for each fiber
        inds = (iFib-1)*3*N+1:iFib*3*N;
        Ms{iFib} = getMloc(N,1.5*Xst(inds)-0.5*Xstm1(inds),eps,Lf,mu,s0);
        [K,Kt]=getKMats3D(1.5*Xst(inds)-0.5*Xstm1(inds),chebyshevmat,w0,N);
        Ks{iFib} = K; Kts{iFib}=Kt;
    end
    % Background flow, strain external force
    [~,gnetupdate] = EvalU0(gam0,omega,t,Xt);
    [U0,g] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1);
    % Making and breaking CL links over step dt/2
%         links = breakCLinks(reshape(Xt,3,N*nFib)',...
%             links,nFib,N,rl,Lf,L,gnetupdate,dt/2);
    links = makeCLinks(links,reshape(Xt,3,N*nFib)',...
        nFib,N,rl,Lf,nCL,L,gnetupdate,dt/2);
    fext = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',...
        N,s0,w0,Lf, Kspring,rl,g,L)',3*N*nFib,1);
    maxIters = 100;
    if (t > 1.6*dt)
        maxIters=imax;
    end
    % Temporal integration
    lambdasm1=lambdas;
    if (t > 1.5*dt)
        lamguess = 2*lambdas-lambdalast;
    else
        lamguess = lambdas;
    end
    [Xnp1,lambdas,fE,Xsp1]=CNSolve(nFib,N,Ms,Ks,Kts,I,wIt,FE,LRLs,URLs,Xt,Xtm1,...
        Xst,Xstm1,U0,dt,s0,w0,Lf,eps,Ds,mu,xi,L,nonLocal,lamguess,maxIters,grav,fext,g);
    Xtm1=Xt;
    Xstm1=Xst;
    % Another network update
    [~,gnetupdate] = EvalU0(gam0,omega,t+dt,Xnp1);
    % Making and breaking CL links over step dt/2
%         links = breakCLinks(reshape(Xnp1,3,N*nFib)',...
%             links,nFib,N,rl,Lf,L,gnetupdate,dt/2);
    links = makeCLinks(links,reshape(Xnp1,3,N*nFib)',...
        nFib,N,rl,Lf,nCL,L,gnetupdate,dt/2);
end