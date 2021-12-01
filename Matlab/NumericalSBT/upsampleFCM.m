function U = upsampleFCM(Targs,starg,X0,f0,s0,b0,Nup,L,a)
    % Collocation pts
    U = 0*Targs;
    for iT=1:length(Targs)
        t = starg(iT);
        P = Targs(iT,:);
        [sup,wup,~]=chebpts(Nup,[0 L],2);
        Rsm = barymat(sup, s0, b0);
        Xsm = Rsm*X0;
        fsm = Rsm*f0;
        R = P-Xsm;
        nR = sqrt(sum(R.*R,2));
        Rdotf = sum(R.*fsm,2);
        A = 1./nR.*((1+a^2./nR.^2).*erf(nR/(a*sqrt(2)))-2*a./nR.*(2*pi)^(-1/2).*exp(-nR.^2/(2*a^2)));
        B = 1./nR.^3.*((1-3*a^2./nR.^2).*erf(nR/(a*sqrt(2)))+6*a./nR.*(2*pi)^(-1/2).*exp(-nR.^2/(2*a^2)));
        IG = A.*fsm + B.*R.*Rdotf;
        smallds=wup*IG;
        U(iT,:)=smallds;
    end
end