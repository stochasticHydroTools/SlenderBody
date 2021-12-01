function Matrix = upsampleRPYSmallStraightMatrix(starg,s0,b0,Xs,Nup,L,a)
    % Collocation pts
    N =length(starg);
    Matrix = zeros(3*N,3*N);
    Nhalf = floor(Nup/2);
    IdentityPart = zeros(3*N);
    TauPart = zeros(3*N);
    for iT=1:N
        t = starg(iT);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=chebpts(Nhalf,dom,1);
            Rsm = barymat(ssm, s0, b0);
            stRsm = stackMatrix(Rsm);
            % Formulate as row of a matrix
            RowVec1 = zeros(3,3*Nhalf);
            RowVec2 = zeros(3,3*Nhalf);
            for jPt=1:Nhalf
                nR = abs(t-ssm(jPt));
                RowVec1(1,3*jPt-2)= 4/(3*a)*(1-9*nR/(32*a))*wsm(jPt);
                RowVec1(2,3*jPt-1)= 4/(3*a)*(1-9*nR/(32*a))*wsm(jPt);
                RowVec1(3,3*jPt)= 4/(3*a)*(1-9*nR/(32*a))*wsm(jPt);
                RowVec2(1,3*jPt-2)= 4/(3*a)*(3*nR/(32*a))*wsm(jPt);
                RowVec2(2,3*jPt-1)= 4/(3*a)*(3*nR/(32*a))*wsm(jPt);
                RowVec2(3,3*jPt)= 4/(3*a)*(3*nR/(32*a))*wsm(jPt);
            end
            IdentityPart(3*(iT-1)+1:3*iT,:) = IdentityPart(3*(iT-1)+1:3*iT,:) + RowVec1*stRsm;
            TauPart(3*(iT-1)+1:3*iT,:) = TauPart(3*(iT-1)+1:3*iT,:) + RowVec2*stRsm;
        end
    end
    for iT=1:N
        Matrix(3*(iT-1)+1:3*iT,:) = (Xs(iT,:)'*Xs(iT,:))*TauPart(3*(iT-1)+1:3*iT,:)...
            +IdentityPart(3*(iT-1)+1:3*iT,:);
    end
end