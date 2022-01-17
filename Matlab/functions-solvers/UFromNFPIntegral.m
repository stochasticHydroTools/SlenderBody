% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. This is the velocity from scalar torque.
function Oonevel = UFromNFPIntegral(X,Xs,Xss,Xsss,s0,N,L,n,nprime,Allbs,mu)
    Oonevel = zeros(N,3);
    for iPt=1:N
        b = Allbs(:,iPt);
        gloc = zeros(N,3);
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt,:) = (cross(n(jPt)*Xs(jPt,:),R)/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/2*cross(Xs(iPt,:),n(iPt)*Xss(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt,:)=-nprime(iPt)/2*cross(Xss(iPt,:),Xs(iPt,:))...
            -n(iPt)/3*cross(Xsss(iPt,:),Xs(iPt,:));
        Oonevel(iPt,:)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end
