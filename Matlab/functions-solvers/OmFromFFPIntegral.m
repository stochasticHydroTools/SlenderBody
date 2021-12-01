% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. 
function Oonevel = OmFromFFPIntegral(X,Xs,Xss,Xsss,s0,N,L,f,fprime,Allbs,mu)
    Oonevel = zeros(N,1);
    for iPt=1:N
        b = Allbs(:,iPt);
        gloc = zeros(N,1);
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            gloc(jPt) = (dot(cross(R,Xs(iPt,:)),f(jPt,:))/nR^3*abs(s0(jPt)-s0(iPt))-...
                1/2*dot(cross(Xs(iPt,:),Xss(iPt,:)),f(iPt,:)))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt)= -dot(fprime(iPt,:),cross(Xss(iPt,:),Xs(iPt,:)))/2 ...
            - 1/6*dot(f(iPt,:),cross(Xsss(iPt,:),Xs(iPt,:)));
        Oonevel(iPt)=L/2*gloc'*b;
    end
    Oonevel = 1/(8*pi*mu)*Oonevel;
end
