% Compute the integrals on |s-s'| < 2a for the rot-trans (scalar rotational
% velocity Omega from force f). % It uses Nsm/2 Gauss-Legendre points on the 
% two different sides of s. 
function Om = upsampleRPYRotTransSmall(X0,X_s,f0,s0,b0,Nsm,L,a,mu)
    % Collocation pts
    N = length(s0);
    Om = zeros(N,1);
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
            fsm = Rsm*f0;
            R = P-Xsm;
            nR = sqrt(sum(R.*R,2));
            FcrossR = sum(cross(fsm,R,2).*X_s(iT,:),2);
            K1 = (1/a-3*nR/(8*a^2)).*FcrossR;
            small = 1/(2*a^2)*K1;
            Om(iT)=Om(iT)+wsm*small;
        end
    end
    Om = 1/(8*pi*mu)*Om;
end