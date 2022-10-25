 % 3N x 3N Matrix for the remaining Stokeslet contribution to the RPY
% trans-trans mobility. Specifically, the integral of the Stokeslet is
% written as int_D = M_stokeslet,loc + M_stokeslet, nonloc
% This is the nonlocal part of that matrix. It uses the precomputed
% integrals "Allbs" that are precomputed in precomputeStokesletInts.m
function FPMat = StokesletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,Allbs)
    FPMatrix = Allbs';
    ActualFPMat = zeros(3*N);
    DfPart = zeros(3*N);
    for iPt=1:N
        indsi = 3*iPt-2:3*iPt;
        nXs = norm(Xs(iPt,:));
        XsDotXss = dot(Xs(iPt,:),Xss(iPt,:));
        Xshat = Xs(iPt,:)/nXs;
        for jPt=1:N
            indsj = 3*jPt-2:3*jPt;
            if (iPt==jPt)
                % Diagonal block
                DiagPart1 = 0.5*(Xs(iPt,:)'*Xss(iPt,:)+Xss(iPt,:)'*Xs(iPt,:))/nXs^3 ...
                 - XsDotXss/(2*nXs^3)*eye(3)-3*XsDotXss/(2*nXs^5)*Xs(iPt,:)'*Xs(iPt,:);
                ActualFPMat(indsi,indsi) =ActualFPMat(indsi,indsi)+...
                    DiagPart1*FPMatrix(iPt,iPt);
                % Derivative part
                DfPart(indsi,indsi) = (eye(3)+Xshat'*Xshat)/nXs*FPMatrix(iPt,iPt);
            else
                rvec = X(iPt,:)-X(jPt,:);
                r = norm(rvec);
                oneoverr = 1.0/r;
                ds = s(jPt)-s(iPt);
                oneoverds = 1.0/ds;
                ActualFPMat(indsi,indsj) = (eye(3) + rvec'*rvec*oneoverr^2)...
                        *oneoverr*abs(ds)*oneoverds*FPMatrix(iPt,jPt);
                ActualFPMat(indsi,indsi)=ActualFPMat(indsi,indsi)-...
                    (eye(3)+Xshat'*Xshat)*oneoverds/nXs*FPMatrix(iPt,jPt);
            end
        end
    end
    FPMat = 1/(8*pi*mu)*0.5*L*(ActualFPMat+DfPart*stackMatrix(D));
end