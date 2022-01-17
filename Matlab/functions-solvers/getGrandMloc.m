% Grand LOCAL DRAG ,obility matrix for the asymptotic evaluations of the RPY 
% integrals. Given regularization delta (delta=0 is unregularized), it
% computes the local drag contributions Mtt (trans-trans), Mtr
% (trans-rot), Mrt (rot-trans), and Mrr (rot-rot). 
% Here Mtr is a 3N x N matrix which gives a 3D velocity at the N nodes from
% a SCALAR parallel torque density. Likewise Mrt is an N x 3N matrix which
% gives the scalar parallel rotation rate from the N 3D forces
function [MTT, MTR, MRT, MRR,sNew] = getGrandMloc(N,Xs,Xss,a,L,mu,s0,delta)
    sNew = RegularizeS(s0,delta,L);
    cs = log(sNew.*(L-sNew)./(4*a.^2)); 
    cprimes = (L-2*sNew)./((L-sNew).*sNew);
    MTT = zeros(3*N); MTR = zeros(3*N,N); MRR = zeros(N);
    MRT = zeros(N,3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s0(iPt);
        tau = Xs(inds);
        Xsprime = Xss(inds);
        XsXs = tau*tau';
        if (delta==0) % unregularized, use special endpoint formulas and keep O(a^2) terms
            if (s0(iPt) < 2*a)
                MTT(inds,inds)=(2+4*t/(3*a)-3*t^2/(16*a^2))*eye(3)+(t^2/(16*a^2))*XsXs...
                    +log((L-t)/(2*a))*(eye(3)+XsXs);
                sbar = t/a;
                MTR(inds,iPt)=(7/12+1/6*sbar^3-3/64*sbar^4+log((L-t)/(2*a)))*1/2*cross(tau,Xsprime);    
                MRT(iPt,inds)=(7/12+1/6*sbar^3-3/64*sbar^4+log((L-t)/(2*a)))*1/2*cross(tau,Xsprime);
                aI_MRR = 1/a^2*(5/8+t/a-27*t^2/(64*a^2)+5*t^4/(256*a^4)-1/2*(1/8-a^2/(2*(L-t)^2)));
                aTau_MRR = 1/a^2*(3/8+9/64*t^2/a^2 - 3/256*t^4/a^4+3/2*(1/8-a^2/(2*(L-t)^2)));
            elseif (s0(iPt) > L-2*a)
                sbar = (L-t)/a;
                MTT(inds,inds)=(2+4/3*sbar-3/16*sbar^2)*eye(3)+1/16*sbar^2*XsXs...
                    +log(t/(2*a))*(eye(3)+XsXs);
                MTR(inds,iPt)=(7/12+1/6*sbar^3-3/64*sbar^4+log(t/(2*a)))*1/2*cross(tau,Xsprime);
                MRT(iPt,inds)=(7/12+1/6*sbar^3-3/64*sbar^4+log((L-t)/(2*a)))*1/2*cross(tau,Xsprime);
                aI_MRR = 1/a^2*(5/8+sbar-27/64*sbar^2 + 5/256*sbar^4-1/2*(1/8-a^2/(2*t^2)));
                aTau_MRR = 1/a^2*(3/8+9/64*sbar^2 - 3/256*sbar^4+3/2*(1/8-a^2/(2*t^2)));
            else
                % Trans-Trans
                MTT(inds,inds)=(4-a^2/(3*t^2)-a^2/(3*(L-t)^2))*eye(3)+...
                    (a^2/t^2+a^2/(L-t)^2)*XsXs +cs(iPt)*(eye(3)+XsXs);
                % Trans-Rot
                MTR(inds,iPt)=1/2*cross(tau,Xsprime)*(cs(iPt)+7/6);
                MRT(iPt,inds)=1/2*cross(tau,Xsprime)*(cs(iPt)+7/6);
                aI_MRR = 1/(a^2)*(5/4-1/2*(1/4-a^2/(2*t^2)-a^2/(2*(L-t)^2)));
                aTau_MRR = 1/(a^2)*(3/4+3/2*(1/4-a^2/(2*t^2)-a^2/(2*(L-t)^2)));
            end
            MRR(iPt,iPt) = aI_MRR+aTau_MRR;
        else % regularized version. Use sbar as argument
            % Trans-Trans
            MTT(inds,inds)=4*eye(3)+cs(iPt)*(eye(3)+XsXs);
            % Trans-Rot
            MTR(inds,iPt)=1/2*cross(tau,Xsprime)*(cs(iPt)+7/6);
            MRT(iPt,inds)=1/2*cross(tau,Xsprime)*(cs(iPt)+7/6);
            MRR = 9/(4*a^2)*eye(N);
        end
    end
    MTT = 1/(8*pi*mu)*MTT;
    MRR = 1/(8*pi*mu)*MRR;
    MTR = 1/(8*pi*mu)*MTR;
    MRT = 1/(8*pi*mu)*MRT;
end