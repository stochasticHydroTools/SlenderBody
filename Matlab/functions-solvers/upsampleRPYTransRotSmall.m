% Compute the integrals on |s-s'| < 2a for the trans-rot (velocity from scalar torque
% n0. It uses Nsm/2 Gauss-Legendre points on the 
% two different sides of s. 
function U = upsampleRPYTransRotSmall(X0,X_s,n0,s0,b0,Nsm,L,a,mu)
    % Collocation pts
    N = length(s0);
    U = zeros(N,3);
    for iT=1:length(s0)
        t = s0(iT);
        P = X0(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=legpts(floor(Nsm/2),dom);
            Rsm = barymat(ssm, s0, b0);
            Xsm = Rsm*X0;
            nsm = (Rsm*n0).*(Rsm*X_s);
            R = P-Xsm;
            nR = sqrt(sum(R.*R,2));
            FcrossR = cross(nsm,R,2);
            K1 = (1/a-3*nR/(8*a^2)).*FcrossR;
            small = 1/(2*a^2)*K1;
            U(iT,:)=U(iT,:)+wsm*small;
        end
    end
    U = 1/(8*pi*mu)*U;
end