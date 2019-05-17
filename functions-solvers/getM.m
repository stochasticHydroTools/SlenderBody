% Get the mobility matrix from Xs
function M = getM(N,Xs,Oone,eps,mu,s0,w0)
    XsXs = zeros(3*N);
    for iPt=1:N
        v = Xs((iPt-1)*3+1:3*iPt);
        XsXs((iPt-1)*3+1:3*iPt,(iPt-1)*3+1:3*iPt)=v*v';
    end
    M=-log(exp(1)*eps^2)*(eye(3*N)+XsXs);
    if (Oone)
        M=M+2*(eye(3*N)-XsXs);
%         M=M+NLMatrix(X,Xs,s0,w0,N,0);
    end
    M=M/(8*pi*mu);
end