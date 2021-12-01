function links = updateMovingLinks(links,X,Xs,N,s,w,L, K, rl,g,Ld,dt)
    [nLink,~]=size(links);
    v0Myosin=1;
    fstall=1;  % 5/25
    % Uniform points on [0, Lf]
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmat = (cos((0:N-1).*th));
    for iL=1:nLink
        fib1 = links(iL,1);
        s1star = links(iL,2);
        fib2 = links(iL,3);
        s2star = links(iL,4);
        shift = links(iL,5:7);
        % Calculate the force density on fiber 1
        inds1 = (fib1-1)*N+1:fib1*N;
        inds2 = (fib2-1)*N+1:fib2*N;
        X1=X(inds1,:);
        th1 = acos(2*s1star/L-1)';
        U1 = (cos((0:N-1).*th1));
        X1star = U1*(Lmat \ X1);
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        th2 = acos(2*s2star/L-1)';
        U2 = (cos((0:N-1).*th2));
        X2star = U2*(Lmat \ X2);
        renorm = w*deltah(s-s1star,N,L)*w*deltah(s-s2star,N,L);
        f1 = zeros(N,3);
        for iPt=1:N
            ds = X1(iPt,:)-X2;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s2star,N,L);
            f1(iPt,:)=-K*w*ig*deltah(s(iPt)-s1star,N,L);
        end
        % Renormalize f1
        f1=f1./renorm;
        % Compute tangential component of f1
        f1_t = w*(sum(f1.*Xs(inds1,:),2).*deltah(s-s1star,N,L))/(w*deltah(s-s1star,N,L));
        % Compute velocity
        vLink1 = max(v0Myosin*(1-norm(f1_t)/fstall),0);
        links(iL,2)=min(links(iL,2)+vLink1*dt,L);
         % Calculate the force density on fiber 2
        f2 = zeros(N,3);
        for iPt=1:N
            ds = X2(iPt,:)-X1;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s1star,N,L);
            f2(iPt,:)=-K*w*ig*deltah(s(iPt)-s2star,N,L);
        end
        % Renormalize
        f2=f2./renorm;
        % Calculate tangential component
        f2_t = w*(sum(f2.*Xs(inds2,:),2).*deltah(s-s2star,N,L))/(w*deltah(s-s2star,N,L));
        vLink2 = max(v0Myosin*(1-norm(f2_t)/fstall),0);
        links(iL,4)=min(links(iL,4)+vLink2*dt,L);
    end
    % Total force and torque are 0 AFTER ACCOUNTING FOR SHIFT. 
end