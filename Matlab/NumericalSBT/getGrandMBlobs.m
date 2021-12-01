function [Mtt, Mrt, Mrr] = getGrandMBlobs(N,X,a,mu)
    Mtt = zeros(3*N);
    for iB=1:N
        for jB = 1:N
            R = X(iB,:)-X(jB,:);
            RR = (R'*R);
            r = norm(R);
            if (r > 2*a)
                TransTrans = (eye(3)/r+RR/r^3)+ 2*a^2/3 *(eye(3)/r^3-3*RR/r^5);
                RotTrans = -CPMatrix(R)/r^3;
                RotRot = -1/2*(eye(3)/r^3-3*RR/r^5);
            elseif (r > 0)
                TransTrans = (4/(3*a)-3*r/(8*a^2))*eye(3)+1/(8*a^2*r)*RR;
                RotTrans = -CPMatrix(R)*1/(2*a^2)*(1/a-3*r/(8*a^2));
                RotRot = 1/a^3*((1-27*r/(32*a)+5*r^3/(64*a^3))*eye(3) + ...
                    (9/(32*a*r)-3*r/(64*a^3))*RR);
            else
                TransTrans = 4/(3*a)*eye(3);
                RotTrans = 0;
                RotRot = 1/a^3*eye(3);
            end
            Mtt(3*iB-2:3*iB,3*jB-2:3*jB)=1/(8*pi*mu)*TransTrans;
            Mrt(3*iB-2:3*iB,3*jB-2:3*jB)=1/(8*pi*mu)*RotTrans;
            Mrr(3*iB-2:3*iB,3*jB-2:3*jB)=1/(8*pi*mu)*RotRot;
        end
    end
end