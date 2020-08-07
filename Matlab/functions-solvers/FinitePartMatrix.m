function FPMat = FinitePartMatrix(X,Xs,LittleDiff,BigDiff,s,L,N,mu)
    % 3N x 3N Matrix for the finite part integral M_FP*f, formed explicitly
    % See python for documentation
    k = 0:N-1;
    s_scaled = -1+2*s/L;
    q=(1+(-1).^(k+1.0)-2*s_scaled.^(k+1.0))./(k+1.0);
    VanderMat = fliplr(vander(s_scaled));
    FPMatrix = VanderMat' \ q';
    FPMatrix = FPMatrix';
    ActualFPMat = zeros(3*N);
    DfPart = zeros(3*N);
    Xss = LittleDiff*Xs;
    for iPt=1:N
        indsi = 3*iPt-2:3*iPt;
        for jPt=1:N
            indsj = 3*jPt-2:3*jPt;
            if (iPt==jPt)
                % Diagonal block
                ActualFPMat(indsi,indsi) =ActualFPMat(indsi,indsi)+...
                    0.5*(Xs(iPt,:)'*Xss(iPt,:)+Xss(iPt,:)'*Xs(iPt,:))*FPMatrix(iPt,iPt);
                % Derivative part
                DfPart(indsi,indsi) = (eye(3)+Xs(iPt,:)'*Xs(iPt,:))*FPMatrix(iPt,iPt);
            else
                rvec = X(iPt,:)-X(jPt,:);
                r = norm(rvec);
                oneoverr = 1.0/r;
                ds = s(jPt)-s(iPt);
                oneoverds = 1.0/ds;
                ActualFPMat(indsi,indsj) = (eye(3) + rvec'*rvec*oneoverr^2)...
                        *oneoverr*abs(ds)*oneoverds*FPMatrix(iPt,jPt);
                ActualFPMat(indsi,indsi)=ActualFPMat(indsi,indsi)-...
                    (eye(3)+Xs(iPt,:)'*Xs(iPt,:))*oneoverds*FPMatrix(iPt,jPt);
            end
        end
    end
    FPMat = 1/(8*pi*mu)*0.5*L*(ActualFPMat+DfPart*BigDiff);
end