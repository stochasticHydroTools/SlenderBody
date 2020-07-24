% Get the mobility matrix from Xs
function M = getMloc(N,Xs,eps,Lf,mu,s0)
    global deltaLocal;
    as = zeros(length(s0),1);
    delta = deltaLocal;
    for iS=1:length(s0)
        s = s0(iS);
        if (s < delta*Lf || s > Lf-delta*Lf)
            as(iS) = eps*2*sqrt(s*(Lf-s)); % ellipsoidal tapering
        elseif (s < 2*delta*Lf)
            wCyl = 1/(1+exp(-23.0258/(delta*Lf)*s+34.5387));
            as(iS) = eps*(Lf*wCyl+(1-wCyl)*2*sqrt(s*(Lf-s)));
        elseif (s > Lf-2*delta*Lf)
            wCyl = 1/(1+exp(-23.0258/(delta*Lf)*(Lf-s)+34.5387));
            as(iS) = eps*(Lf*wCyl+(1-wCyl)*2*sqrt(s*(Lf-s)));
        else
            as(iS) = eps*Lf;
        end
    end
    Ls = log(4.*s0.*(Lf-s0)./as.^2);      
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
        Mloc=1/(8*pi*mu)*((eye(3)-3*XsXs)+...
            Ls(mod(iPt-1,N)+1)*(eye(3)+XsXs));
        vals((iPt-1)*9+1:iPt*9)=Mloc(:);
    end
    M = sparse(rows,cols,vals);
end