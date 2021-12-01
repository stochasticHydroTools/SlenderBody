function MRPY = calcRPYKernel(x,y,a)
    R = x-y;
    r = norm(R);
    Rhat = R/r;
    if (r==0)
        MRPY = 4/(3*a)*eye(3);
    elseif (r < 2*a)
        MRPY = (4/(3*a)-3*r/(8*a^2))*eye(3)+1/(8*a^2*r)*(R'*R);
    else
        MRPY = (eye(3)+(Rhat'*Rhat))/r+2*a^2/3*(eye(3)-3*(Rhat'*Rhat))/r^3;
    end
end