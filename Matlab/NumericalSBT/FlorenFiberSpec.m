function [X,Xs, Xss, Xsss, Xssss] = FlorenFiberSpec(q,s,L,b,D,numerderivs)
    syms t
    XsSym = [cos(q*t.^3 .* (t-L).^3) sin(q*t.^3.*(t - L).^3) 1]/sqrt(2);
    Xss = double(subs(diff(XsSym,t),s));
    Xsss = double(subs(diff(XsSym,t,2),s));
    Xssss = double(subs(diff(XsSym,t,3),s));
    Xs = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(length(s),1)]/sqrt(2);
    if (numerderivs)
        Xss = D*Xs;
        Xsss=D^2*Xs;
        Xssss=D^3*Xs;
    end
    X = pinv(D)*Xs;
    X=X-barymat(0,s,b)*X;
end