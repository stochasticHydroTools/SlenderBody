% Calculate the self velocity for a fiber X. This includes the Tornberg
% scheme for the finite part integral. 
% Inputs: N = number of Chebyshev points, s0 = arclength coordinates of
% Chebyshev points, Lf = fiber length, epsilon = slenderness ratio, X =
% position of Chebyshev points, Xs = tangent vectors, f = forcing on the
% fiber, D = differentiation matrix
% Output = (8*pi*mu)*fiber velocity. So it must be divided by 8*pi*mu to
% give the correct result. Split into 2 pieces: the leading order local
% term, and the non-local finite part integral
function [Local,Oone] = calcSelf(N,s0,Lf,epsilon,X,Xs,f,D)
    % The self term
    % Local part
    Local = zeros(N,3);
    s=s0*2/Lf-1;
    aeps=2*epsilon;
    Ls = log((2*(1-s.^2)+2*sqrt((1-s.^2).^2+4*aeps.^2))./(aeps.^2));
    for iPt=1:N
        XsXs=Xs(iPt,:)'*Xs(iPt,:);
        Local(iPt,:) = ((eye(3)-3*XsXs)+...
            Ls(mod(iPt-1,N)+1)*(eye(3)+XsXs))*f(:,iPt);
    end
    % Tornberg way
    Xss = D*Xs;
    fprime=(D*f')';
    Oone=NLIntegrate(X,Xs,Xss,s0,N,Lf,f,fprime);
end