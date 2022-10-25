function U = upsampleRPY(Targs,starg,X0,f0,s0,b0,Nup,L,a)
    % Collocation pts
    U = 0*Targs;
    for iT=1:length(Targs)
        t = starg(iT);
        P = Targs(iT,:);
        smallds = zeros(1,3);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            if (max(dom)-min(dom) > 0)
                [ssm,wsm,~]=chebpts(Nup,dom,1);
                Rsm = barymat(ssm, s0, b0);
                Xsm = Rsm*X0;
                fsm = Rsm*f0;
                R = P-Xsm;
                nR = sqrt(sum(R.*R,2));
                Rhat = R./nR;
                Rdotf = sum(R.*fsm,2);
                K1 = (1-9*nR/(32*a)).*fsm;
                K2 = 3./(32*a).*Rdotf.*Rhat;
                small = 4/(3*a)*(K1+K2);
                smallds=smallds+wsm*small;
            end
        end

        % Two Stokeslet integrals
        int1 = zeros(1,3);
        if (t > 2*a)
            [s1,w1,~]=chebpts(Nup,[0 t-2*a],1);
            %[s1,w1,~]=chebpts(Nup,[0 t],1);
            R1 = barymat(s1, s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            Rdotf = sum(R.*f1,2);
            StokIG = f1./nR + Rdotf.*R./nR.^3;
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = StokIG+2*a^2/3*DoubIG;
            int1=w1*totIG;
        end
        
        int2 = zeros(1,3);
        if (t < L-2*a)
            [s1,w1,~]=chebpts(Nup,[t+2*a L],1);
            %[s1,w1,~]=chebpts(Nup,[t L],1);
            R1 = barymat(s1,s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            Rdotf = sum(R.*f1,2);
            StokIG = f1./nR + Rdotf.*R./nR.^3;
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = StokIG+2*a^2/3*DoubIG;
            int2=w1*totIG;
        end
        U(iT,:)=smallds+int1+int2;
    end
end