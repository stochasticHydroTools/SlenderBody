% Function that gives Xp1 for the different solvers
function [Xnp1,Xtm1] = getXp1(Xt,Xtm1,solver,dt,FE,LRLs,URLs,Ds,N,eps,mu,...
                        chebyshevmat,Dinv,s0,w0,Oone,I,wIt)
    if (solver == 2) % For Crank-Nicolson
        Xs = Ds*(1.5*Xt-0.5*Xtm1);
        M = getM(N,Xs,Oone,eps,mu,s0,w0);
        % Now need the communication with the alphas
        [K,Kt]=getKMats3D(Xs,chebyshevmat,Dinv,w0,N);
    else % other explicit / first order solvers
        Xs = Ds*Xt;
        M = getM(N,Xs,Oone,eps,mu,s0,w0);
        % Now need the communication with the alphas
        [K,Kt]=getKMats3D(Xs,chebyshevmat,Dinv,w0,N);
    end
    Xtm1=Xt;
    % Temporal integration
    if (solver == -1) % FE
        u = ExSolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt);
        Xnp1 = Xt+u*dt;
    elseif (solver == -3) %RK3
        x1 = ExSolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt);
        M2 = getM(N,Ds*(Xt+dt*x1),Oone,eps,mu,s0,w0);
        [K2, Kt2]=getKMats3D(Ds*(Xt+dt*x1),chebyshevmat,Dinv,w0,N);
        x2 = ExSolve(N,M2,K2,Kt2,I,wIt,FE,LRLs,URLs,Xt+dt*x1);
        M3 = getM(N,Ds*(Xt+dt/4*x1+dt/4*x2),Oone,eps,mu,s0,w0);
        [K3, Kt3]=getKMats3D(Ds*(Xt+dt/4*x1+dt/4*x2),chebyshevmat,Dinv,w0,N);
        x3 = ExSolve(N,M3,K3,Kt3,I,wIt,FE,LRLs,URLs,Xt+dt/4*x1+dt/4*x2);
        Xnp1=Xt+dt*(x1/6+x2/6+2*x3/3);
    elseif (solver==1) % BE
        Xnp1=BESolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt,dt);
    elseif (solver==2) % CN
        Xnp1=CNSolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt,dt);
    end
end