% The local drag (or leading order singular) expansion for the Rotlet term,
% given in Appendix A of the paper. 
function MRT = getMlocRotlet(N,Xs,Xss,a,L,mu,s0,delta)
    MRT = zeros(N,3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s0(iPt);
        tau = Xs(inds);
        Xsprime = Xss(inds);
        if (t < 2*a)
            MRT(iPt,inds)=log((L-t)/(2*a))*1/2*cross(tau,Xsprime);
        elseif (t > L-2*a)
            MRT(iPt,inds)=log(t/(2*a))*1/2*cross(tau,Xsprime);
        else
            MRT(iPt,inds)=1/2*cross(tau,Xsprime)*log(t.*(L-t)./(4*a.^2));
        end
    end
    MRT = 1/(8*pi*mu)*MRT;
end