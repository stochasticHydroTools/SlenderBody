% Inextensible update by rotating tangent vectors and integrating
function [Xnp1,Xsp1,Omega] = updateX(Xt,ut2,N,dt,Lf,Xsin,Xsm1,dU,solver,D,Omega)
    % Resampling matrices
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmat = (cos((0:N-1).*th));
    Xsin =reshape(Xsin,3,N)';
    Xsm1 =reshape(Xsm1,3,N)';
    % Compute new Xs
%     g1=Lmat(:,1:N-1)*alphas(1:N-1);
%     g2=Lmat(:,1:N-1)*alphas(N:2*N-2);
    Xsk = Xsin;
    if (solver > 1)
        Xsk = 1.5*Xsin-0.5*Xsm1;
    end
%     [theta,phi,~] = cart2sph(Xsk(:,1),Xsk(:,2),Xsk(:,3));
%     theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
%     n1s=[-sin(theta) cos(theta) 0*theta];
%     n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
%     Omega=g1.*n2s-g2.*n1s;
    Omega = cross(Xsk,reshape(dU,3,N)');
    %max(abs(Omega-Omega1))
    % Increment Xs
    nOm = sqrt(sum(Omega.*Omega,2));
    % Have to truncate somewhere to avoid instabilities
    k = Omega./nOm;
    k(nOm < 1e-12,:) = 0;
    % Rodriguez formula on the N grid. 
    Xsp1 = Xsin.*cos(nOm*dt)+cross(k,Xsin).*sin(nOm*dt)+k.*sum(k.*Xsin,2).*(1-cos(nOm*dt));
    Xnp1 = pinv(D)*Xsp1;
    % Make the velocity at s = the first s equal to the original solver
    Xnp1=Xnp1-Xnp1(1,:)+(Xt(1:3)'+dt*ut2(1:3)');
%     [s,~,b]=chebpts(N,[0 Lf],1);
%     MPEval = barymat(Lf/2,s,b);
%     Xnp1=Xnp1-Xnp1(1,:)+(Xt(1:3)'+dt*MPEval*reshape(ut2,3,N)');
    % Evaluate velocity at midpoint (TEMP)
    Xnp1=reshape(Xnp1',3*N,1);
    Xsp1=reshape(Xsp1',3*N,1);
end