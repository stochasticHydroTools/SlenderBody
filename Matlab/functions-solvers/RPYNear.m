% Near field RPY tensor
function M = RPYNear(rvec,xi,a,mu)
    r=norm(rvec);
    rhat=rvec/r;
    if (sum(isnan(rhat)) > 0)
        rhat=[0 0 0];
    end
    RR = rhat'*rhat;
    M = 1/(6*pi*mu*a)*(F(r,xi,a)*(eye(3)-RR)+G(r,xi,a)*RR);
end
    
function val = F(r,xi,a)
    if (r>2*a)
        f0=0;
        f1=(18*r^2*xi^2+3)/(64*sqrt(pi)*a*r^2*xi^3);
        f2=(2*xi^2*(2*a-r)*(4*a^2+4*a*r+9*r^2)-2*a-3*r)/...
            (128*sqrt(pi)*a*r^3*xi^3);
        f3=(-2*xi^2*(2*a+r)*(4*a^2-4*a*r+9*r^2)+2*a-3*r)/...
            (128*sqrt(pi)*a*r^3*xi^3);
        f4=(3-36*r^4*xi^4)/(128*a*r^3*xi^4);
        f5=(4*xi^4*(r-2*a)^2*(4*a^2+4*a*r+9*r^2)-3)/(256*a*r^3*xi^4);
        f6=(4*xi^4*(r+2*a)^2*(4*a^2-4*a*r+9*r^2)-3)/(256*a*r^3*xi^4);
    else
        f0=-(r-2*a)^2*(4*a^2+4*a*r+9*r^2)/(32*a*r^3);
        f1=(18*r^2*xi^2+3)/(64*sqrt(pi)*a*r^2*xi^3);
        f2=(2*xi^2*(2*a-r)*(4*a^2+4*a*r+9*r^2)-2*a-3*r)/...
            (128*sqrt(pi)*a*r^3*xi^3);
        f3=(-2*xi^2*(2*a+r)*(4*a^2-4*a*r+9*r^2)+2*a-3*r)/...
            (128*sqrt(pi)*a*r^3*xi^3);
        f4=(3-36*r^4*xi^4)/(128*a*r^3*xi^4);
        f5=(4*xi^4*(r-2*a)^2*(4*a^2+4*a*r+9*r^2)-3)/...
            (256*a*r^3*xi^4);
        f6=(4*xi^4*(r+2*a)^2*(4*a^2-4*a*r+9*r^2)-3)/...
            (256*a*r^3*xi^4);
    end
    val = f0+f1*exp(-r^2*xi^2)+f2*exp(-(r-2*a)^2*xi^2)+...
        f3*exp(-(r+2*a)^2*xi^2)+f4*erfc(r*xi)+f5*erfc((r-2*a)*xi)+...
        f6*erfc((r+2*a)*xi);
    if (r < 1e-10)
        val = 1/(4*sqrt(pi)*xi*a)*(1-exp(-4*a^2*xi^2)+...
            4*sqrt(pi)*a*xi*erfc(2*a*xi));
    end
end

function val = G(r,xi,a)
    if (r>2*a)
        g0=0;
        g1=(6*r^2*xi^2-3)/(32*sqrt(pi)*a*r^2*xi^3);
        g2=(-2*xi^2*(r-2*a)^2*(2*a+3*r)+2*a+3*r)/...
            (64*sqrt(pi)*a*r^3*xi^3);
        g3=(2*xi^2*(r+2*a)^2*(2*a-3*r)-2*a+3*r)/...
            (64*sqrt(pi)*a*r^3*xi^3);
        g4=-3*(4*r^4*xi^4+1)/(64*a*r^3*xi^4);
        g5=(3-4*xi^4*(2*a-r)^3*(2*a+3*r))/(128*a*r^3*xi^4);
        g6=(3-4*xi^4*(2*a-3*r)*(2*a+r)^3)/(128*a*r^3*xi^4);
    else
        g0=(2*a-r)^3*(2*a+3*r)/(16*a*r^3);
        g1=(6*r^2*xi^2-3)/(32*sqrt(pi)*a*r^2*xi^3);
        g2=(-2*xi^2*(r-2*a)^2*(2*a+3*r)+2*a+3*r)/...
            (64*sqrt(pi)*a*r^3*xi^3);
        g3=(2*xi^2*(r+2*a)^2*(2*a-3*r)-2*a+3*r)/...
            (64*sqrt(pi)*a*r^3*xi^3);
        g4=-3*(4*r^4*xi^4+1)/(64*a*r^3*xi^4);
        g5=(3-4*xi^4*(2*a-r)^3*(2*a+3*r))/(128*a*r^3*xi^4);
        g6=(3-4*xi^4*(2*a-3*r)*(2*a+r)^3)/(128*a*r^3*xi^4);
    end
    val = g0+g1*exp(-r^2*xi^2)+g2*exp(-(r-2*a)^2*xi^2)+...
        g3*exp(-(r+2*a)^2*xi^2)+g4*erfc(r*xi)+g5*erfc((r-2*a)*xi)+...
        g6*erfc((r+2*a)*xi);
    if (r < 1e-10)
        val = 0;
    end
end