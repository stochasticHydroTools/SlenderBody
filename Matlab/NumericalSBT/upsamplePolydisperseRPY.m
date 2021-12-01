function U = upsamplePolydisperseRPY(Targs,starg,X0,f0,s0,b0,Nup,L,as)
    % Collocation pts
    U = 0*Targs;
    for iT=1:length(Targs)
        a = as(iT);
        t = starg(iT);
        P = Targs(iT,:);
        [ssm,wsm,~]=chebpts(Nup,[max(t-2*max(as),0) min(t+2*max(as),L)],2);
        Rsm = barymat(ssm, s0, b0);
        Xsm = Rsm*X0;
        fsm = Rsm*f0;
        asm = Rsm*as;
        sumas = asm+a;
        disp('Small a not tested!')
        diffas = zeros(Nup,1);
        maxas = zeros(Nup,1);
        for iPt=1:Nup
            amin=min(a,asm(iPt));
            maxas(iPt)=max(a,asm(iPt));
            diffas(iPt)=maxas(iPt)-amin;
        end
        R = P-Xsm;
        nR = sqrt(sum(R.*R,2));
        tooclose = diffas >= nR;
        closenough = (sumas >= nR & ~tooclose);
        Rhat = R./nR;
        Rdotf = sum(R.*fsm,2);
        K1 = (16*nR.^3.*(asm+a)-((a-asm).^2+3*nR.^2).^2)./(32*nR.^3).*fsm;
        K2 = 3*((a-asm).^2-nR.^2).^2./(32*nR.^4).*Rdotf.*Rhat;
        small = 4./(3*a.*asm).*(K1+K2).*closenough+4./(3*maxas).*fsm.*tooclose;
        smallds=wsm*small;

        % Two Stokeslet integrals
        int1 = zeros(1,3);
        if (t > 2*min(as))
            [s1,w1,~]=chebpts(Nup,[0 t-2*min(as)],1);
            R1 = barymat(s1, s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            a1 = R1*as;
            sumas = a1+a;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            farenough = sumas < nR;
            Rdotf = sum(R.*f1,2);
            StokIG = f1./nR + Rdotf.*R./nR.^3;
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = (StokIG+(a1.^2+a.^2)/3.*DoubIG).*farenough;
            int1=w1*totIG;
        end
        
        int2 = zeros(1,3);
        if (t < L-2*min(as))
            [s1,w1,~]=chebpts(Nup,[t+2*min(as) L],1);
            R1 = barymat(s1,s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            a1 = R1*as;
            sumas = a1+a;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            farenough = sumas < nR;
            Rdotf = sum(R.*f1,2);
            StokIG = f1./nR + Rdotf.*R./nR.^3;
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = (StokIG+(a1.^2+a.^2)/3.*DoubIG).*farenough;
            int2=w1*totIG;
        end
        U(iT,:)=smallds+int1+int2;
    end
end