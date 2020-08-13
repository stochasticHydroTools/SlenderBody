% Get the mobility matrix from Xs
function M = getMlocRPY(N,Xs,eps,Lf,mu,s0,delta)
%     Ls = log(4.*s0.*(Lf-s0)./(eps*Lf).^2); 
    % Regularize L over lengthscale delta
%     for iS=1:length(s0)
%         s = s0(iS);
%         if (s/Lf < 0.5)
%             wCyl = 1/(1+exp(-36.84/Lf*s+9.2102)); % 10^-3 at 0 and at center
%             Ls(iS) = wCyl*Ls(iS)+(1-wCyl)*log(4*delta*(1-delta)./eps^2);
%         elseif (s/Lf > 0.5)
%             wCyl = 1/(1+exp(-36.84/Lf*(Lf-s)+9.2102));
%             Ls(iS) = wCyl*Ls(iS)+(1-wCyl)*log(4*delta*(1-delta)./eps^2);   
%         end
%     end
    x = 2*s0/Lf-1;
    Lnormal = 4.*s0.*(Lf-s0)./(eps*Lf).^2;
    regwt = tanh((x+1)/delta)-tanh((x-1)/delta)-1;
    Ldelta = 4*delta/2*(1-delta/2)/eps^2;
    Ls = log(regwt.*Lnormal+(1-regwt.^2).*Ldelta);
    aI = 1;
    atau = -3;
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
        Mloc=1/(8*pi*mu)*((aI*eye(3)+atau*XsXs)+Ls(iPt)*(eye(3)+XsXs));
        vals((iPt-1)*9+1:iPt*9)=Mloc(:);
    end
    M = sparse(rows,cols,vals);
end