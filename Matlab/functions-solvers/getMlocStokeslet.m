% The local drag matrix for the Stokeslet term. These formulas are given in
% Appendix A of the paper. The input delta is for the regularization of the
% local drag term (i.e., to use SBT). 
function MTT = getMlocStokeslet(N,Xs,a,L,mu,s0,delta)
    s = RegularizeS(s0,delta,L);
    MTT = zeros(3*N); 
    [~,b]=size(Xs);
    if (b < 3)
        error('Need to pass Xs as an N x 3 vector')
    end
    c=zeros(N,1);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        tau = Xs(iPt,:);
        XsXs = tau'*tau;
        if (t < 2*a)
            c(iPt)=log((L-t)/(2*a));
        elseif (t > L-2*a)
            c(iPt)=log(t/(2*a));
        else
            % Trans-Trans
            c(iPt)=log(t*(L-t)/(4*a.^2));
        end
        MTT(inds,inds)=c(iPt)*(eye(3)+XsXs);
    end
    MTT = 1/(8*pi*mu)*MTT;
end