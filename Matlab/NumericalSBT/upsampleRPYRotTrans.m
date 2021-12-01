function Om = upsampleRPYRotTrans(Targs,starg,X0,f0,s0,b0,Nup,L,a)
    % Collocation pts
    Om = 0*Targs;
    for iT=1:length(Targs)
        t = starg(iT);
        P = Targs(iT,:);
        smallds = zeros(1,3);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            [ssm,wsm,~]=chebpts(Nup,dom,1);
            Rsm = barymat(ssm, s0, b0);
            Xsm = Rsm*X0;
            fsm = Rsm*f0;
            R = P-Xsm;
            nR = sqrt(sum(R.*R,2));
            FcrossR = cross(fsm,R);
            K1 = (1/a-3*nR/(8*a^2)).*FcrossR;
            small = 1/(2*a^2)*K1;
            smallds=smallds+wsm*small;
        end

        % Two Stokeslet integrals
        int1 = zeros(1,3);
        if (t > 2*a)
            [s1,w1,~]=chebpts(Nup,[0 t-2*a],1);
            R1 = barymat(s1, s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            FCrossR = cross(f1,R);
            totIG = FCrossR./nR.^3;
            int1=w1*totIG;
        end
        
        int2 = zeros(1,3);
        if (t < L-2*a)
            [s1,w1,~]=chebpts(Nup,[t+2*a L],1);
            R1 = barymat(s1,s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            FCrossR = cross(f1,R);
            totIG = FCrossR./nR.^3;
            int2=w1*totIG;
        end
        Om(iT,:)=smallds+int1+int2;
    end
end