% Get the mobility matrix from Xs
function [M,Ls] = getMloc(N,Xs,eps,L,mu,s0,delta)
    % Regularized version
    if (delta < 0.5)
        x = 2*s0/L-1;
        regwt = tanh((x+1)/delta)-tanh((x-1)/delta)-1;
        sNew = s0;
        sNew(s0 < L/2) = regwt(s0 < L/2).*s0(s0 < L/2)+(1-regwt(s0 < L/2).^2).*delta*L/2;
        sNew(s0 > L/2) = L-flipud(sNew(s0 < L/2));
    else
        sNew = L/2*ones(length(s0),1);
    end
    Ls = log(4.*sNew.*(L-sNew)./(eps*L).^2);      
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
        Mloc=1/(8*pi*mu)*((eye(3)-3*XsXs)+Ls(iPt)*(eye(3)+XsXs));
        vals((iPt-1)*9+1:iPt*9)=Mloc(:);
    end
    M = sparse(rows,cols,vals);
end