% Precomputes the integrals
% int_D (eta'-eta)/abs(eta'-eta)^3*T_k(eta') deta'
% involved in the nearly singular quadratures for the Doublet.
% See Appendix F of the paper. 
% chebpoly = 0 for monomials, chebpoly=1 for Chebyshev polynomials
function Allbs = precomputeDoubletInts(s0,L,a,N,chebpoly)
    k=0:N-1;
    sscale=-1+2*s0/L;
    for iPt=1:N
        s = s0(iPt);
        eta = sscale(iPt);
        sLow = max(s-2*a,0);
        sHi = min(s+2*a,L);
        etaLow = -1+2*sLow/L;
        etaHi = -1+2*sHi/L;
        % Compute integrals numerically to high accuracy
        q = zeros(1,N);
        for kk=k
            if (etaLow > -1)
                [n,w]=chebpts(200,[-1 etaLow],1);
                poly = n.^kk;
                if (chebpoly)
                    poly = cos(kk*acos(n));
                end
                q(kk+1)=w*((n-eta)./abs(n-eta).^3.*poly);
            end
            if (etaHi < 1)
                [n,w]=chebpts(200,[etaHi 1],1);
                poly = n.^kk;
                if (chebpoly)
                    poly = cos(kk*acos(n));
                end
                q(kk+1)=q(kk+1)+w*((n-eta)./abs(n-eta).^3.*poly);
            end
        end
        AllQs(iPt,:)=q;
    end
    PolyMat = fliplr(vander(sscale))';
    if (chebpoly)
        PolyMat = cos(k.*acos(sscale))';
    end
    Allbs= mldivide(PolyMat,AllQs');
end