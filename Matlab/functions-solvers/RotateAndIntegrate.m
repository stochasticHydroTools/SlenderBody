function Xplus = RotateAndIntegrate(alpha,x,XFromTau,XFromTauInv)
    if (size(x,2)==3)
        x=reshape(x',[],1);
    end
    if (size(alpha,2)==1)
        alpha=reshape(alpha,3,[])';
    end
    tauMP = reshape(XFromTauInv*x,3,[])';
    tau_plus = rotateTau(tauMP(1:end-1,:),alpha(1:end-1,:),1);
    Xmp_plus = tauMP(end,:)+alpha(end,:);
    tauMP_plus = reshape([tau_plus;Xmp_plus]',[],1);
    Xplus = XFromTau*tauMP_plus;
end