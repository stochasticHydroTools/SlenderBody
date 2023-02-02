% Compute the action of the non-local mobility matrix in an UNBOUNDED DOMAIN. 
% This code just performs an O(N^2) sum
function U = UnboundedRPYSum(nFib,X,F,a,mu,Rupsample,wup,WTildeInverse,direct,SelfOnly)
    % Contibutions from the other fibers
    [NupsampleforNL,Nx]=size(Rupsample);
    if (direct)
        NupsampleforNL=Nx;
    end
    U = zeros(Nx*nFib,3);
    for iFib=1:nFib
        U_up = zeros(NupsampleforNL,3);
        iinds = (iFib-1)*Nx+1:iFib*Nx;
        if (direct)
            ThisX = X(iinds,:);
        else
            ThisX = Rupsample*X(iinds,:);
        end
        for iPt=1:NupsampleforNL
            otherFibs = 1:nFib;
            if (SelfOnly)
                otherFibs=iFib;
            end
            for jFib=otherFibs
                jinds = (jFib-1)*Nx+1:jFib*Nx;
                if (direct)
                    otherPts = X(jinds,:);
                    otherForces = F(jinds,:);
                else
                    otherPts = Rupsample*X(jinds,:);
                    otherForces = diag(wup)*Rupsample*WTildeInverse*F(jinds,:);
                end
                for jPt=1:NupsampleforNL
                    rvec = ThisX(iPt,:)-otherPts(jPt,:);
                    MRPY = RPYTot(rvec,a,mu);
                    U_up(iPt,:)=U_up(iPt,:)+otherForces(jPt,:)*MRPY;
                end
            end
        end
        % Downsample the velocity
        if (direct)
            U(iinds,:) = U_up;
        else
            U(iinds,:) = WTildeInverse*Rupsample'*diag(wup)*U_up;
        end
    end
end