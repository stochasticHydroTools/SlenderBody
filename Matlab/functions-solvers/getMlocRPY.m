% Get the mobility matrix from Xs
function [M,aI,atau] = getMlocRPY(N,Xs,a,L,mu,s0)
    % Unregularized RPY tensor
    aI = zeros(N,1);
    atau = zeros(N,1);
    for iT=1:N
        t = s0(iT);
        if (t > 2*a && t < L-2*a)
            aI(iT) = log((4*(L-t).*t)/(a^2))-log(16)+(1/6-2/3*(a^2/(2*t^2)+a^2/(2*(L-t)^2)))+23/6;
            atau(iT) = log((4*(L-t).*t)/(a^2))-log(16)-3*(1/6-2/3*(a^2/(2*t^2)+a^2/(2*(L-t)^2)))+1/2;
        elseif (t <=2*a)
            aI(iT) = log((L-t)/(2*a))+2+4*t/(3*a)-3*t^2/(16*a^2);
            atau(iT) = log((L-t)/(2*a))+t^2/(16*a^2);
        else
            aI(iT) = log(t/(2*a))+2+4*(L-t)/(3*a)-3*(L-t)^2/(16*a^2);
            atau(iT) = log(t/(2*a))+(L-t)^2/(16*a^2);
        end
    end
    M = zeros(3,3,N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        v = Xs(inds);
        XsXs = v*v';
        Mloc=1/(8*pi*mu)*(aI(iPt)*eye(3)+atau(iPt)*XsXs);
        M(:,:,iPt)=Mloc;
    end
    rows=zeros(9*N,1);
    cols=zeros(9*N,1);
    vals=zeros(9*N,1);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        rows((iPt-1)*9+1:iPt*9)=...
            [inds(1) inds(1) inds(1) inds(2) inds(2) inds(2) inds(3) inds(3) inds(3)]';
        cols((iPt-1)*9+1:iPt*9)=...
            [inds(1) inds(2) inds(3) inds(1) inds(2) inds(3) inds(1) inds(2) inds(3)]';
        Mloc=M(:,:,iPt);
        vals((iPt-1)*9+1:iPt*9)=Mloc(:);
    end
    M = sparse(rows,cols,vals);
end