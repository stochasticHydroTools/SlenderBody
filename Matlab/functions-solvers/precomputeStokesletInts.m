function Allbs = precomputeStokesletInts(s0,L,a,N,chebpoly)
    k=0:N-1;
    sscale=-1+2*s0/L;
    AllQs = zeros(N);
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
            if (chebpoly)
                if (etaLow > -1)
                    [n,w]=chebpts(100,[-1 etaLow],1);
                    poly = n.^kk;
                    if (chebpoly)
                        poly = cos(kk*acos(n));
                    end
                    q(kk+1)=w*((n-eta)./abs(n-eta).*poly);
                end
                if (etaHi < 1)
                    [n,w]=chebpts(100,[etaHi 1],1);
                    poly = n.^kk;
                    if (chebpoly)
                        poly = cos(kk*acos(n));
                    end
                    q(kk+1)=q(kk+1)+w*((n-eta)./abs(n-eta).*poly);
                end
            else
                % Monomials - can do the integrals analytically
                q(kk+1) = -1./(kk+1).*(etaLow.^(kk+1)-(-1).^(kk+1))+1./(kk+1).*(1-etaHi.^(kk+1));
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