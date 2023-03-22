function Mrr = RotRotMobilityMatrix(X,a,L,mu,s,b,D,AllbD,NForSmall,asymp,delta)
    [N,~]=size(X);
    Xs = D*X;
    Xss = D*Xs;
    if (~asymp && delta > 0)
        delta=0; % override user input for exact RPY
        warning('You input delta > 0 and said that you wanted exact RPY')
    end
    Mrr = -1/2*getMlocDoublet(N,Xs,a,L,mu,s,delta);
    if (asymp)
        Rest = getMLocRRSmallParts(N,Xs,a,L,mu,s,delta);
    else
        Rest = 1/(8*pi*mu)*upsampleRotRotSmallMatrix(X,s,b,NForSmall,L,a) ...
         -1/2*DoubletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,AllbD);
    end
    Mrr = Mrr+Rest;
    % Hit with Xsdot matrix in the front and back
    XsDot = zeros(N,3*N);
    for iP=1:N
        XsDot(iP,3*iP-2:3*iP)=Xs(iP,:);
    end
    Mrr = XsDot*Mrr*XsDot';
end

% Compute the integrals on |s-s'| < 2a for the trans-trans matrix Mtt. 
% It uses Nup/2 Gauss-Legendre points on the two different sides of s. 
% Compute the integrals on |s-s'| < 2a for the trans-trans matrix Mtt. 
% It uses Nup/2 Gauss-Legendre points on the two different sides of s. 
function Matrix = upsampleRotRotSmallMatrix(X,s,b,Nup,L,a)
    % Collocation pts
    [N,~]=size(X);
    Matrix = zeros(3*N,3*N);
    Nhalf = floor(Nup/2);
    %AllRS = [];
    for iT=1:N
        t = s(iT);
        P = X(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            if (max(dom)-min(dom) > 0)
                [ssm,wsm,~]=legpts(Nhalf,dom);
                Rsm = barymat(ssm, s, b);
                %AllRS = [AllRS; diag(wsm)*Rsm];
                Xsm = Rsm*X;
                stRsm = stackMatrix(diag(wsm)*Rsm);
                % Formulate as row of a matrix
                RowVec = zeros(3,3*Nhalf);
                for jPt=1:Nhalf
                    R = P-Xsm(jPt,:);
                    nR = norm(R);
                    Rhat = R/nR;
                    RowVec(:,3*(jPt-1)+1:3*jPt)=...
                        1/a^3*((1-27*nR/(32*a)+5*nR^3/(64*a^3))*eye(3)...
                        +(9*nR/(32*a)-3*nR^3/(64*a^3))*(Rhat'*Rhat));
                end
                Matrix(3*(iT-1)+1:3*iT,:) = Matrix(3*(iT-1)+1:3*iT,:) +RowVec*stRsm;
            end
        end
    end
end

function MRR = getMLocRRSmallParts(N,Xs,a,L,mu,s0,delta)
    % Regularized version
    s = RegularizeS(s0,delta,L);
    MRR = zeros(3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        tau = Xs(iPt,:);
        nXs = norm(tau);
        Xshat = tau/nXs;
        XsXs = Xshat'*Xshat;
        if (t < 2*a)
            intI = (512*a^4-432*a^4*nXs+80*a^4*nXs^3+256*a^3*t...
                -108*a^2*nXs*t^2+5*nXs^3*t^4)/(256*a^6);
            intTau = -3*(-48*a^4*nXs+16*a^4*nXs^3-12*a^2*nXs*t^2+nXs^3*t^4)/(256*a^6);
        elseif (t > L-2*a)
            sbar = (L-t);
            intI = (512*a^4-432*a^4*nXs+80*a^4*nXs^3+256*a^3*sbar...
                -108*a^2*nXs*sbar^2+5*nXs^3*sbar^4)/(256*a^6);
            intTau = -3*(-48*a^4*nXs+16*a^4*nXs^3-12*a^2*nXs*sbar^2+nXs^3*sbar^4)/(256*a^6);
        else
            intI = 1/(8*a^2)*(32-27*nXs+5*nXs^3);
            intTau = -3/(8*a^2)*(-3*nXs+nXs^3);
        end
        MRR(inds,inds)=intI*eye(3)+intTau*XsXs;
    end
    MRR = 1/(8*pi*mu)*MRR;
end