% Calculate the self velocity for a fiber X. This includes the Tornberg
% scheme for the finite part integral. 
% Inputs: N = number of Chebyshev points, s0 = arclength coordinates of
% Chebyshev points, Lf = fiber length, epsilon = slenderness ratio, X =
% position of Chebyshev points, Xs = tangent vectors, f = forcing on the
% fiber, D = differentiation matrix
% Output = (8*pi*mu)*fiber velocity. So it must be divided by 8*pi*mu to
% give the correct result. Split into 2 pieces: the leading order local
% term, and the non-local finite part integral
function [Local,Oone] = calcSelf(N,s0,L,eps,X,Xs,f,D,delta)
    % The self term
    % Local part
    Local = zeros(N,3);
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
    for iPt=1:N
        XsXs=Xs(iPt,:)'*Xs(iPt,:);
        Local(iPt,:) = ((eye(3)-3*XsXs)+...
            Ls(mod(iPt-1,N)+1)*(eye(3)+XsXs))*f(:,iPt);
    end
    % Tornberg way
    Xss = D*Xs;
    fprime=(D*f')';
    Oone=NLIntegrate(X,Xs,Xss,s0,N,L,f,fprime);
end