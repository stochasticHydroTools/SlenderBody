% Inextensible update by rotating tangent vectors and integrating
function [Xnp1,Xsp1] = updateX(Xt,ut2,N,dt,Lf,Xsin,Xsm1,dU)
    % Resampling matrices
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    th_up = flipud(((2*(0:2*N-1)+1)*pi/(2*2*N))');
    Lmat = (cos((0:N-1).*th));
    RS_up = (cos((0:N-1).*th_up))*Lmat^(-1);
    RS_down = (cos((0:2*N-1).*th))*(cos((0:2*N-1).*th_up))^(-1);
    Xsin =reshape(Xsin,3,N)';
    Xsm1 =reshape(Xsm1,3,N)';
    % Compute new Xs
    Xsk = RS_up*(1.5*Xsin-0.5*Xsm1);
    dU_up = RS_up*dU;
    Omega = cross(Xsk,dU_up);
    % Increment Xs
    Omega = RS_down*Omega;
    nOm = sqrt(sum(Omega.*Omega,2));
    % Have to truncate somewhere to avoid instabilities
    k = Omega./nOm;
    k(nOm < 1e-6,:) = 0;
    % Rodriguez formula on the N grid. 
    Xsp1 = Xsin.*cos(nOm*dt)+cross(k,Xsin).*sin(nOm*dt)+k.*sum(k.*Xsin,2).*(1-cos(nOm*dt));
    % Integrate Xsp1 by going to Chebyshev space
    Xshat = Lmat \ Xsp1;
    Xhat = Lf/2*FIMatrix(N)*[Xshat; zeros(2,3)];
    % Go back to real space
    Xnp1 = Lmat*Xhat;
    % Make the velocity at s = the first s equal to the original solver
    Xnp1=Xnp1-Xnp1(1,:)+(Xt(1:3)'+dt*ut2(1:3)');
    Xnp1=reshape(Xnp1',3*N,1);
    Xsp1=reshape(Xsp1',3*N,1);
end