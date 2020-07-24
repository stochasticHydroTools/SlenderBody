% Matric for the finite part integral using brute force quadrature,
% skipping the singular point
function M = NLMatrix(X,Xs,s0,w,N,delta)
    M=zeros(3*N);
    for iS=1:N
        sinds=3*iS-2:3*iS;
        nsinds=1:N;
        if (delta==0) 
            nsinds=setdiff(nsinds,iS); 
        end
        M(sinds,sinds)=-(eye(3)+Xs(sinds)*Xs(sinds)')*...
            (w(nsinds)*(1./sqrt((s0(nsinds)-s0(iS)).^2+delta^2)));
        sumv=0;
        for iSprime=iS+1:N
            spinds=3*iSprime-2:3*iSprime;
            R_s=X(sinds)-X(spinds);
            nRs=norm(R_s);
            R_s=R_s/nRs;
            M(sinds,spinds)=M(sinds,spinds)+(eye(3)+R_s*R_s')/sqrt(nRs^2+delta^2)*w(iSprime);
            M(spinds,sinds)=M(spinds,sinds)+(eye(3)+R_s*R_s')/sqrt(nRs^2+delta^2)*w(iS);
            sumv=sumv+1/sqrt(nRs^2+delta^2)*w(iSprime);
        end
        if (delta > 0) % add the iS term
            M(sinds,sinds)=M(sinds,sinds)+eye(3)/delta*w(iS);
        end
    end
end