% The local drag matrix for the doublet term. These formulas are given in
% Appendix A of the paper. The input extraterms is when we want to carry the asymptotic
% expansion to the next order, and is not used. 
function MTT = getMlocDoublet(N,Xs,Xss,Xsss,Ds,a,L,mu,s0,delta,extraterms)
    %s = RegularizeS(s0,delta,L);
    s = s0;
    MTT_f = zeros(3*N); 
    MTT_fpr = zeros(3*N);
    MTT_fDpr = zeros(3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        kap = dot(Xss(iPt,:),Xss(iPt,:));
        M1 = Xs(iPt,:)'*Xs(iPt,:);
        M2 = 1/2*(Xs(iPt,:)'*Xss(iPt,:)+Xss(iPt,:)'*Xs(iPt,:));
        M3 = 1/6*(Xsss(iPt,:)'*Xs(iPt,:)+Xs(iPt,:)'*Xsss(iPt,:)) + ...
            1/4*Xss(iPt,:)'*Xss(iPt,:)+5*kap/24*M1;
        if (t < 2*a)
            intx3 = (1/(8*a^2)-1/(2*(L-t)^2));
            intx2 = (-1/(L-t)+1/(2*a));
            intx1 = log((L-t)/(2*a));
        elseif (t > L-2*a)
            intx3 = 1/(8*a^2)-1/(2*t^2);
            intx2 = (1/t-1/(2*a));
            intx1 = log(t/(2*a)); 
        else
            intx3 = 1/(4*a^2)-1/(2*t^2)-1/(2*(L-t)^2);
            intx2 = (L-2*t)/(t*(L-t));
            intx1 = log(t*(L-t)/(4*a^2)); 
        end
        if (extraterms==0)
            MTT_f(inds,inds)=intx3*(eye(3)-3*M1);
        else
            MTT_f(inds,inds)=intx3*(eye(3)+(-3)*M1)+intx2*(-3)*M2+intx1*(kap/8*eye(3)+(-3)*M3);
            MTT_fpr(inds,inds)=intx2*(eye(3)+(-3)*M1)+intx1*(-3)*M2;
            MTT_fDpr(inds,inds)=intx1*(eye(3)/2+(-3)*M1/2);
        end
            
    end
    MTT = 1/(8*pi*mu)*(MTT_f+MTT_fpr*Ds+MTT_fDpr*Ds^2);
end