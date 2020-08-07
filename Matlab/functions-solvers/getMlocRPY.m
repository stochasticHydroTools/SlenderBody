% Get the mobility matrix from Xs
function M = getMlocRPY(N,Xs,eps,Lf,mu,s0)
    k = sqrt(3/2);
    aI = 47/12-log(16*k^2);
    atau = 1/4-log(16*k^2);
    L = log(4.*s0.*(Lf-s0)./(eps*Lf)^2);      
    rows=zeros(9*N,1);
    cols=zeros(9*N,1);
    vals=zeros(9*N,1);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        v = Xs(inds);
        XsXs = v*v';
        rows((iPt-1)*9+1:iPt*9)=...
            [inds(1) inds(1) inds(1) inds(2) inds(2) inds(2) inds(3) inds(3) inds(3)]';
        cols((iPt-1)*9+1:iPt*9)=...
            [inds(1) inds(2) inds(3) inds(1) inds(2) inds(3) inds(1) inds(2) inds(3)]';
        Mloc=1/(8*pi*mu)*((aI*eye(3)+atau*XsXs)+L(iPt)*(eye(3)+XsXs));
        vals((iPt-1)*9+1:iPt*9)=Mloc(:);
    end
    M = sparse(rows,cols,vals);
end