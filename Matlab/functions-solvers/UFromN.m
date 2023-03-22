function U = UFromN(X,n,D,Allbs,a,L,mu,s,b,nonLocal,NForSmall)
    [N,~]=size(X);
    U = reshape(getMlocRotlet(N,X,D,a,L,mu,s)'*n,3,N)';
    if (NForSmall > 0)
        U = U + upsampleRPYTransRotSmall(X,D,n,s,b,NForSmall,L,a,mu);
    end
    if (nonLocal)
        U = U + UFromNFPIntegral(X,D,s,N,L,n,Allbs,mu);
    end
end


% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. This is the velocity from scalar torque.
function Oonevel = UFromNFPIntegral(X,D,s0,N,L,n,Allbs,mu)
    Oonevel = zeros(N,3);
    Xs = D*X;
    Xss = D*Xs;
    Xsss = D*Xss;
    nprime = D*n;
    for iPt=1:N
        nXs = norm(Xs(iPt,:));
        b = Allbs(:,iPt);
        gloc = zeros(N,3);
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt,:) = (cross(n(jPt)*Xs(jPt,:),R)/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/(2*nXs^3)*cross(Xs(iPt,:),n(iPt)*Xss(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt,:)=-nprime(iPt)/(2*nXs^3)*cross(Xss(iPt,:),Xs(iPt,:))...
            -n(iPt)/(3*nXs^3)*cross(Xsss(iPt,:),Xs(iPt,:))...
            -3/4*cross(Xs(iPt,:),Xss(iPt,:))*n(iPt)/nXs^5*dot(Xs(iPt,:),Xss(iPt,:));
        Oonevel(iPt,:)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end

% Compute the integrals on |s-s'| < 2a for the trans-rot (velocity from scalar torque
% n0. It uses Nsm/2 Gauss-Legendre points on the 
% two different sides of s. 
function U = upsampleRPYTransRotSmall(X,D,n0,s0,b0,Nsm,L,a,mu)
    % Collocation pts
    N = length(s0);
    U = zeros(N,3);
    X_s = D*X;
    for iT=1:length(s0)
        t = s0(iT);
        P = X(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=legpts(floor(Nsm/2),dom);
            Rsm = barymat(ssm, s0, b0);
            Xsm = Rsm*X;
            nsm = (Rsm*n0).*(Rsm*X_s);
            R = P-Xsm;
            nR = sqrt(sum(R.*R,2));
            FcrossR = cross(nsm,R,2);
            K1 = (1/a-3*nR/(8*a^2)).*FcrossR;
            small = 1/(2*a^2)*K1;
            U(iT,:)=U(iT,:)+wsm*small;
        end
    end
    U = 1/(8*pi*mu)*U;
end