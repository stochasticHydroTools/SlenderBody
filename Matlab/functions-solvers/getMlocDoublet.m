% The local drag matrix for the doublet term. These formulas are given in
% Appendix A of the paper. The input extraterms is when we want to carry the asymptotic
% expansion to the next order, and is not used. 
function MTT = getMlocDoublet(N,Xs,a,L,mu,s0)
    %s = RegularizeS(s0,delta,L);
    s = s0;
    MTT_f = zeros(3*N); 
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        nXs = norm(Xs(iPt,:));
        Xshat = Xs(iPt,:)/nXs;
        if (t < 2*a)
            intx3 = (1/(8*a^2)-1/(2*(L-t)^2));
        elseif (t > L-2*a)
            intx3 = 1/(8*a^2)-1/(2*t^2);
        else
            intx3 = 1/(4*a^2)-1/(2*t^2)-1/(2*(L-t)^2);
        end
        MTT_f(inds,inds)=intx3*(eye(3)-3*(Xshat'*Xshat))/nXs^3;
    end
    MTT = 1/(8*pi*mu)*MTT_f;
end