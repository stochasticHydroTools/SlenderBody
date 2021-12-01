function U = upsampleDoublet(Targs,starg,X0,f0,s0,b0,Nup,L,a)
    % Collocation pts
    U = 0*Targs;
    for iT=1:length(Targs)
        t = starg(iT);
        P = Targs(iT,:);

        % Two Stokeslet integrals
        int1 = zeros(1,3);
        if (t > 2*a)
            [s1,w1,~]=chebpts(Nup,[0 t-2*a],1);
            R1 = barymat(s1, s0, b0);
            X1 = R1*X0;
            f1 = R1*f0;
            R = P-X1;
            nR = sqrt(sum(R.*R,2));
            Rdotf = sum(R.*f1,2);
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = DoubIG;
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
            Rdotf = sum(R.*f1,2);
            DoubIG = f1./nR.^3-3*Rdotf.*R./nR.^5;
            totIG = DoubIG;
            int2=w1*totIG;
        end
        U(iT,:)=int1+int2;
    end
end