% Fully 3D slender body theory
% First half step update for the links
%links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt/2);
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            makePlot; 
        end
        Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
        forces=[forces; reshape(lambdas,3,N*nFib)'];
    end
    % Evolve system
    % Background flow, strain, external force
    [U0,g] = EvalU0(gam0,omega,t+dt/2,1.5*Xt-0.5*Xtm1,flowtype);
    fCL = reshape(getCLforce(links,reshape(1.5*Xt-0.5*Xtm1,3,N*nFib)',N,s0,w0,Lf, Kspring,rl,g,Ld)',3*N*nFib,1);
    gravF = reshape(([0 0 grav/Lf].*ones(N*nFib,3))',3*N*nFib,1);
    maxIters = 100;
    lamguess = lambdas;
    if (t > 1.6*dt)
        maxIters=maxiters;
        lamguess = 2*lambdas-lambdalast;
    end
    lambdalast=lambdas;
    [Xnp1,lambdas,fE,Xsp1]=CNSolve(nFib,N,deltaLocal,Lmat,I,wIt,FE,Xt,Xtm1,...
        Xst,Xstm1,U0,dt,s0,w0,Lf,eps,Ds,mu,xi,Ld,nonLocal,lamguess,maxIters,fCL+gravF,g);
    Xtm1=Xt;
    Xstm1=Xst;
    Xt = Xnp1;
    Xst = Xsp1;
    % Compute total force (per length) at time n+1/2
%     totforce = lambdas+fEstar;
%     totStress = 1/Ld^3*getStress(Xstar,totforce,w0,N,nFib,links,nCL,rl,Kspring,gn,Ld,s0,Lf);
%     fibstresses(count+1)=totStress(2,1);
    % Update myosin
%     if (count==stopcount-1)
%         links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt/2);
%     else
%         links=updateMovingLinks(links,reshape(Xt,3,N*nFib)',reshape(Xst,3,N*nFib)',N,s0,w0,Lf, Kspring, rl,g,Ld,dt);
%     end
end
Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
forces=[forces; reshape(lambdas,3,N*nFib)'];
if (makeMovie)
    movieframes(1)=[];
end
% tstr = gam0/omega;
% % Only take modulus over last 2 cycles
% [Gp, Gdp] = getModuli(fibstresses(stopcount/2+1:end)/tstr,stopcount/2,dt,omega);
% % Add fluid
% Gdp = Gdp + mu*omega;
% close;
%save(strcat('Loc_nFib',num2str(nFib),'_nCL',num2str(nCL),'_K',num2str(Kspring),'.mat'))