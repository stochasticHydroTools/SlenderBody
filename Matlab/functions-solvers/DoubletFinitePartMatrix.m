 % 3N x 3N Matrix for the remaining doublet contribution to the RPY
% trans-trans mobility. Specifically, the integral of the doublet is
% written as int_D = M_doub,loc + M_doub, nonloc
% This is the nonlocal part of that matrix. It uses the precomputed
% integrals "Allbs" that are precomputed in precomputeDoubletInts.m
function FPMat = DoubletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,Allbs)
    FPMatrix = Allbs';
    ActualFPMat = zeros(3*N);
    DfPart = zeros(3*N);
    for iPt=1:N
        indsi = 3*iPt-2:3*iPt;
        M1 = Xs(iPt,:)'*Xs(iPt,:);
        M2 = 1/2*(Xs(iPt,:)'*Xss(iPt,:)+Xss(iPt,:)'*Xs(iPt,:));
        for jPt=1:N
            indsj = 3*jPt-2:3*jPt;
            if (iPt==jPt)
                % Diagonal block
                ActualFPMat(indsi,indsi) = ActualFPMat(indsi,indsi)-3*M2*FPMatrix(iPt,iPt);
                % Derivative part
                DfPart(indsi,indsi) = (eye(3)-3*M1)*FPMatrix(iPt,iPt);
            else
                rvec = X(iPt,:)-X(jPt,:);
                r = norm(rvec);
                oneoverr = 1.0/r;
                ds = s(jPt)-s(iPt);
                oneoverds = 1.0/ds;           
                ActualFPMat(indsi,indsj) = (eye(3) -3*(rvec'*rvec)*oneoverr^2)...
                        *oneoverr^3*abs(ds)^3*oneoverds*FPMatrix(iPt,jPt);
                ActualFPMat(indsi,indsi)=ActualFPMat(indsi,indsi)-...
                    (eye(3)-3*Xs(iPt,:)'*Xs(iPt,:))*oneoverds*FPMatrix(iPt,jPt);
            end
        end
    end
    FPMat = 1/(8*pi*mu)*0.5*L*(ActualFPMat+DfPart*stackMatrix(D));
end