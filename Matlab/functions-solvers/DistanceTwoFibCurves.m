% Find roots
function [distance,sFib1,sFib2] = DistanceTwoFibCurves(X1,X2,s1,s2,L,sNp1,bNp1,DNp1,tol)
    %% 2D projected modified Newton optimization
    rts = [s1;s2];
    Gradient = ones(2,1);
    Hessian = zeros(2);
    nIts=0;
    distanceEps = 1e-10;
    maxIts=10;
    condMax=1e6;
    MaxNormDir = 0.1*L;
    while (norm(Gradient) > tol && nIts < maxIts)
        nIts=nIts+1;
        Ev1 = barymat(rts(1),sNp1,bNp1);
        Ev2 = barymat(rts(2),sNp1,bNp1);
        X1p = Ev1*X1;
        X2p = Ev2*X2;
        dist_before=norm(X1p-X2p);
        % Derivative calculations at (s1,s2)
        DX1p = Ev1*DNp1*X1;
        D2X1p = Ev1*DNp1^2*X1;
        DX2p = Ev2*DNp1*X2;
        D2X2p = Ev2*DNp1^2*X2;
        Gradient(1) = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
        Gradient(2) = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
        Hessian(1,1) = 2*(dot(DX1p,DX1p)+dot(D2X1p,X1p)-dot(D2X1p,X2p));
        Hessian(2,1) = -2*dot(DX1p,DX2p);
        Hessian(1,2) = Hessian(2,1);
        Hessian(2,2) = 2*(dot(DX2p,DX2p)+dot(D2X2p,X2p)-dot(D2X2p,X1p));
        % Check if Hessian is positive definite and modify if not
        [V,D]=eig(Hessian);
        HessEigs = diag(D);
        [maxEig,maxInd]=max(HessEigs);
        [minEig,minInd]=min(HessEigs);
        if (maxEig < 0)
            break;
        elseif (maxEig > 0 && minEig < maxEig/condMax)
            HessEigs(minInd)=maxEig/condMax;
        end
        HessianInv = V'*diag(1./HessEigs)*V;
        % Define boundary set and project off directions
        BInds = find((rts < distanceEps & Gradient > 0) | (rts > L-distanceEps & Gradient < 0));
        HessianInv(BInds,:)=0;
        HessianInv(:,BInds)=0;
        Gradient(BInds)=0;
        pk = -HessianInv*Gradient;
        if (norm(pk) > MaxNormDir)
            pk = pk/norm(pk)*MaxNormDir;
        end
        % Back-tracking line search
        armijo=0;
        alpha=1;
        while (~armijo && alpha > 0)
            if (alpha < tol)
                alpha=0;
            end
            rtguess = rts+alpha*pk;
            % Projection
            rtguess(rtguess < 0)=0;
            rtguess(rtguess > L)=L;
            % Evaluate function
            Ev1 = barymat(rtguess(1),sNp1,bNp1);
            Ev2 = barymat(rtguess(2),sNp1,bNp1);
            X1p = Ev1*X1;
            X2p = Ev2*X2;
            distance = norm(X1p-X2p);
            % The Armijo parameter is 1/2
            if (dist_before - distance >=-1/2*alpha*Gradient'*pk)
                armijo=1;
                rts=rtguess;
            else
                alpha=alpha/2;
            end
        end
    end
    sFib1=rts(1);
    sFib2=rts(2);
end