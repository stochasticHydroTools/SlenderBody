% Fully 3D slender body theory
% Parameters
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        if (makeMovie) 
            makePlot; 
        end
        Xpts=[Xpts;reshape(Xt,3,N*nFib)'];
        forces=[forces; reshape(lambdas,3,N*nFib)'];
    end
    % This returns the position X at time t+dt (Xt), the position at time t
    % (X), and the forces at time t (for CN the lambdas are at t+dt/2)
    fextprev=fext;
    Xstar = 1.5*Xt-0.5*Xtm1;
    [Xt,Xtm1,lambdas,fEstar,fext,lambdalast,links,Xst,Xstm1] = getXp1(Xt,Xtm1,Xst,Xstm1,...
        lambdas,lambdalast,maxiters,dt,FE,LRLs,URLs,Ds,nFib,N,eps,...
        mu,chebyshevmat,Dinv,s0,w0,Lf,nonLocal,I,wIt,gam0,omega,t,grav,nCL,links,Kspring,rl,Ld,xi);
    % Compute total force (per length) at time n+1/2
    totforce = lambdas+fEstar;
    [~,gnMP] = EvalU0(gam0,omega,t+dt/2,Xt);
    totStress = 1/Ld^3*getStress(Xstar,totforce,w0,N,nFib,links,nCL,rl,Kspring,gn,Ld,s0,Lf);
    fibstresses(count+1)=totStress(2,1);
end
toc
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