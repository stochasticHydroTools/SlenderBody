% This function uses the method of Tornberg / Barnett / Klingenberg to
% evaluate the finite part integral to spectral accuracy using the monomial
% basis. 
function Oonevel = NLIntegrate(X,Xs,Xss,s0,N,L,f,fprime)
    Oonevel = zeros(N,3);
    k=0:N-1;
    sscale=-1+2*s0/L;
    for iPt=1:N
        q=(1+(-1).^(k+1)-2*sscale(iPt).^(k+1))./(k+1);
        % This MUST be done with a backwards stable method (mldivide)
        % Important point: note that b can be precomputed for each s. This
        % might seem like a lot but because s is the same on every fiber it
        % will just be an N x N matrix that we multiply by a vector to get
        % the nonlocal velocity everywhere. 
        b = mldivide(fliplr(vander(sscale))',q');
        gloc = zeros(N,3);
        for jPt=1:N
            R=X(iPt,:)-X(jPt,:);
            nR=norm(R);
            Rhat=R/nR;
            gloc(jPt,:) = ((eye(3)+Rhat'*Rhat)/nR*abs(s0(jPt)-s0(iPt))*f(:,jPt)-...
                (eye(3)+Xs(iPt,:)'*Xs(iPt,:))*f(:,iPt))/(s0(jPt)-s0(iPt));
        end
        gloc(iPt,:)=0.5*(Xs(iPt,:)'*Xss(iPt,:)+Xss(iPt,:)'*Xs(iPt,:))*f(:,iPt)+...
            (eye(3)+Xs(iPt,:)'*Xs(iPt,:))*fprime(:,iPt);
        Oonevel(iPt,:)=L/2*gloc'*b;
    end
end