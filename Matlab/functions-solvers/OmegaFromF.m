function Omega = OmegaFromF(X,f,D,Allbs,a,L,mu,s,b,nonLocal,NForSmall)
    [N,~]=size(X);
    Omega = getMlocRotlet(N,X,D,a,L,mu,s)*reshape(f',3*N,1);
    if (NForSmall > 0)
        Omega = Omega + upsampleRPYRotTransSmall(X,D,f,s,b,NForSmall,L,a,mu);
    end
    if (nonLocal)
        Omega = Omega + OmFromFFPIntegral(X,D,s,N,L,f,Allbs,mu); 
    end
end


% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. 
% This is the integral that gives Omega from the force
function Oonevel = OmFromFFPIntegral(X,D,s0,N,L,f,Allbs,mu)
    Oonevel = zeros(N,1);
    Xs = D*X;
    Xss = D*Xs;
    Xsss = D*Xss;
    fprime=D*f;
    for iPt=1:N
        b = Allbs(:,iPt);
        gloc = zeros(N,1);
        nXs = norm(Xs(iPt,:));
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt) = (dot(cross(R,Xs(iPt,:)),f(jPt,:))/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/(2*nXs^3)*dot(cross(Xs(iPt,:),Xss(iPt,:)),f(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt)= -1/(2*nXs^3)*(dot(fprime(iPt,:),cross(Xss(iPt,:),Xs(iPt,:)))+ ...
            1/3*dot(f(iPt,:),cross(Xsss(iPt,:),Xs(iPt,:))))...
            -3/4*dot(cross(Xs(iPt,:),Xss(iPt,:)),f(iPt,:))/nXs^5*dot(Xs(iPt,:),Xss(iPt,:));
        Oonevel(iPt)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end

% Compute the integrals on |s-s'| < 2a for the rot-trans (scalar rotational
% velocity Omega from force f). % It uses Nsm/2 Gauss-Legendre points on the 
% two different sides of s. 
function Om = upsampleRPYRotTransSmall(X0,D,f0,s0,b0,Nsm,L,a,mu)
    % Collocation pts
    N = length(s0);
    X_s = D*X0;
    Om = zeros(N,1);
    for iT=1:length(s0)
        t = s0(iT);
        P = X0(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=legpts(floor(Nsm/2),dom);
            Rsm = barymat(ssm, s0, b0);
            Xsm = Rsm*X0;
            fsm = Rsm*f0;
            R = P-Xsm;
            nR = sqrt(sum(R.*R,2));
            FcrossR = sum(cross(fsm,R,2).*X_s(iT,:),2);
            K1 = (1/a-3*nR/(8*a^2)).*FcrossR;
            small = 1/(2*a^2)*K1;
            Om(iT)=Om(iT)+wsm*small;
        end
    end
    Om = 1/(8*pi*mu)*Om;
end
