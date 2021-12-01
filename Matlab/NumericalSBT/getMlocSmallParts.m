function MTT = getMlocSmallParts(N,Xs,a,L,mu,s0,Ds,Xss,delta,secondterm)
    % Regularized version
    s = RegularizeS(s0,delta,L);
    %s = s0;
    MTT = zeros(3*N);
    MTT_fprime = zeros(3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        XsXs = Xs(iPt,:)'*Xs(iPt,:);
        if (t < 2*a)
            intI = 23/12+4*t/(3*a)-3*t^2/(16*a^2);
            intTau = 1/4+t^2/(16*a^2);
        elseif (t > L-2*a)
            sbar = (L-t)/a;
            intI = 23/12+4*sbar/3-3*sbar^2/16;
            intTau = 1/4+sbar^2/16;
        else
            intI = 23/6;
            intTau = 1/2;
        end
        MTT(inds,inds)=intI*eye(3)+intTau*XsXs;
        if (secondterm > 0)
            if (t < 2*a)
                intIprime = (40*a^3-16*a*t^2+3*t^3)/(24*a^3);
                intTauprime = (8*a^3-t^3)/(3*a^3);
            elseif (t > L-2*a)
                intIprime = (-40*a^3+16*a*(L-t)^2-3*(L-t)^3)/(24*a^3);
                intTauprime = (-8*a^3-(t-L)^3)/(3*a^3);
            else
                intIprime = 0;
                intTauprime = 0;
            end
            MTT(inds,inds)=MTT(inds,inds)+a/16*intTauprime*...
                (Xss(iPt,:)'*Xs(iPt,:)+Xs(iPt,:)'*Xss(iPt,:));
            MTT_fprime(inds,inds)=a*(intIprime*eye(3)+1/8*intTauprime*XsXs);
        end
    end
    MTT = 1/(8*pi*mu)*(MTT+MTT_fprime*Ds);
end