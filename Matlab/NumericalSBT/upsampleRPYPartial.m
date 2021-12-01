% RPY integrals for targets that are not necessarily on the fiber
function U = upsampleRPYPartial(Targs,starg,X0,f0,s0,b0,Nup,L,a,delta)
    % Collocation pts
    U = 0*Targs;
    for iT=1:length(Targs)
        t = starg(iT);
        P = Targs(iT,:);
        smallds = zeros(1,3);
        if (t > delta*L-2*a && t < L*(1-delta)+2*a)
            [ssm,wsm,~]=chebpts(Nup,[max(t-2*a,delta*L) min(t+2*a,L-delta*L)],2);
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
            smallds=wsm*small;
        end

        % Two Stokeslet integrals
        % Fiber can be done in one piece
        if (t <= delta*L || t >= L-delta*L)
            if (t <= delta*L)
                [s1,w1,~]=chebpts(Nup,[max(delta*L,t+2*a) L-delta*L],1);
            else
                [s1,w1,~]=chebpts(Nup,[delta*L min(L-delta*L,t-2*a)],1);
            end
            R1 = barymat(s1, s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            Rdotf = sum(R.*f1,2);
            StokIG = f1./nR + Rdotf.*R./nR.^3;
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = StokIG+2*a^2/3*DoubIG;
            int=w1*totIG;
        else % On the fiber, has to be done in two pieces
            int1 = zeros(1,3);
            if (t-delta*L > 2*a)
                [s1,w1,~]=chebpts(Nup,[delta*L t-2*a],1);
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
            if (t < L-delta*L-2*a)
                [s1,w1,~]=chebpts(Nup,[t+2*a L-delta*L],1);
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
            int = int1+int2;
        end
        U(iT,:)=smallds+int;
    end
end