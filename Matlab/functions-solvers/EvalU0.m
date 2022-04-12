% Evaluate background flow and domain strain
function [U0,g] = EvalU0(gam0,w,t,X,flowtype)
    U0 = zeros(length(X),1);
    if (flowtype=='Q')
        U0(1:3:end)=gam0*X(2:3:end).^2; % parabolic flow
    elseif (flowtype=='E')
        U0(1:3:end)=gam0*X(1:3:end); % extensional flow
        U0(2:3:end)=-gam0*X(2:3:end);
    else
        U0(1:3:end)=gam0*cos(w*t)*X(2:3:end);   % shear flow
    end
    if (w==0)
        g=gam0*t;
    else
        g = gam0/w*sin(w*t); % total strain
    end
    g=g-round(g);
end