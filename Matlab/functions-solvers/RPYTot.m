% Formula for the unbounded RPY tensor
function M = RPYTot(rvec,a,mu)
    r=norm(rvec);
    rhat=rvec/r;
    if (sum(isnan(rhat)) > 0)
        rhat=[0 0 0];
    end
    RR = rhat'*rhat;
    M = 1/mu*(F(r,a)*(eye(3)-RR)+G(r,a)*RR);
end
    
function val = F(r,a)
    if (r>2*a)
        val = (2*a^2 + 3*r^2)/(24*pi*r^3);
    else
        val = (32*a - 9*r)/(192*a^2*pi);
    end
end

function val = G(r,a)
    if (r>2*a)
        val = (-2*a^2 + 3*r^2)/(12*pi*r^3);
    else
        val = (16*a - 3*r)/(96*a^2*pi);
    end
end