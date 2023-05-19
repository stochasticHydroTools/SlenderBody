function Uncertainties = UncertainityGenMax(nTimes,params,w,fits,data,unc,Gpstart)
    M = length(w);
    nP = length(params);
    Gprime = data(1:M);
    Gdprime = data(M+1:end);
    GprimeFit = fits(1:M);
    GdprimeFit = fits(M+1:end);
    sigGprime = unc(1:M);
    sigGdprime = unc(M+1:end);
    dRdxiEL = zeros(nP,length(w));
    d2RdxidxjEL = zeros(nP,nP,length(w));
    dRdxiVISC = zeros(nP,length(w));
    d2RdxidxjVISC = zeros(nP,nP,length(w));
    for iTau=1:nTimes
        tau = params(nTimes+iTau);
        g = params(iTau);
        % Elastic part
        dRdxiEL(nTimes+iTau,:)= (2*g*tau*w.^2)./(1 + tau^2*w.^2).^2; % dR'/dtau_i 
        dRdxiEL(iTau,:) = tau^2*w.^2./(1+tau^2*w.^2); %dR'/dg_i
        d2RdxidxjEL(nTimes+iTau,nTimes+iTau,:) = (g*(2*w.^2 - 6*tau^2*w.^4))./(1 + tau^2*w.^2).^3; %d^2 R''/dtau_i^2
        d2RdxidxjEL(nTimes+iTau,iTau,:) = (2*tau*w.^2)./(1 + tau^2*w.^2).^2; % d^2 R'/dtau_i dg_i
        d2RdxidxjEL(iTau,iTau+nTimes,:) = d2RdxidxjEL(nTimes+iTau,iTau,:);
        % Viscous part
        dRdxiVISC(nTimes+iTau,:)= g*(w-tau^2*w.^3)./(1 + tau^2*w.^2).^2; % dR''/dtau_i 
        dRdxiVISC(iTau,:) = tau*w./(1+tau^2*w.^2); %dR''/dg_i
        d2RdxidxjVISC(nTimes+iTau,nTimes+iTau,:) = 2*g*tau*w.^3.*(-3+tau^2*w.^2)./(1 + tau^2*w.^2).^3; %d^2 R''/dtau_i^2
        d2RdxidxjVISC(nTimes+iTau,iTau,:) = (w-tau^2*w.^3)./(1 + tau^2*w.^2).^2; % d^2 R''/dtau_i dg_i
        d2RdxidxjVISC(iTau,nTimes+iTau,:) = d2RdxidxjVISC(nTimes+iTau,iTau,:);  % d^2 R'/dg_i dtau_i
    end
    dRdxiVISC(2*nTimes+1,:)=w; % dR''/ deta (no second derivative)
    if (nP > 2*nTimes+1) 
        dRdxiEL(2*nTimes+2,:)=1;
    end
    % Build Fisher information matrix
    FisherEL = zeros(length(params));
    FisherVISC = zeros(length(params));
    for p=1:length(params)
        for q = 1:length(params)
            CrossPartial = reshape(d2RdxidxjEL(p,q,:),1,length(w));
            FisherEL(p,q) = sum(1./sigGprime(Gpstart:end).^2 ...
                .*((Gprime(Gpstart:end)-GprimeFit(Gpstart:end)).*CrossPartial(Gpstart:end) ...
                    - dRdxiEL(p,Gpstart:end).*dRdxiEL(q,Gpstart:end)));
            CrossPartialV = reshape(d2RdxidxjVISC(p,q,:),1,length(w));
            FisherVISC(p,q) = sum(1./sigGdprime.^2.*((Gdprime-GdprimeFit).*CrossPartialV - dRdxiVISC(p,:).*dRdxiVISC(q,:)));
        end
    end
    Fisher = -(FisherEL + FisherVISC);
    Uncertainties = 2*sqrt(diag(Fisher^(-1)));
end
    