% Compute the integrals on |s-s'| < 2a for the trans-trans matrix Mtt. 
% It uses Nup/2 Gauss-Legendre points on the two different sides of s. 
function Matrix = upsampleRPYSmallMatrix(Targs,starg,X0,s0,b0,Nup,L,a)
    % Collocation pts
    N =length(Targs);
    Matrix = zeros(3*N,3*N);
    Nhalf = floor(Nup/2);
    %AllRS = [];
    for iT=1:N
        t = starg(iT);
        P = Targs(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=legpts(Nhalf,dom);
            Rsm = barymat(ssm, s0, b0);
            %AllRS = [AllRS; diag(wsm)*Rsm];
            Xsm = Rsm*X0;
            stRsm = stackMatrix(diag(wsm)*Rsm);
            % Formulate as row of a matrix
            RowVec = zeros(3,3*Nhalf);
            for jPt=1:Nhalf
                R = P-Xsm(jPt,:);
                nR = norm(R);
                Rhat = R/nR;
                RowVec(:,3*(jPt-1)+1:3*jPt)=...
                    4/(3*a)*((1-9*nR/(32*a))*eye(3)+3/(32*a)*(R'*Rhat));
            end
            Matrix(3*(iT-1)+1:3*iT,:) = Matrix(3*(iT-1)+1:3*iT,:) +RowVec*stRsm;
        end
    end
end