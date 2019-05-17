function [K,Kt]=getKMats3D(Xts,chebyshevmat,Dinv,w0,N)
    n1s=[-Xts(2:3:end) Xts(1:3:end) zeros(N,1)];
    normn1s=(sqrt(n1s(:,1).^2+n1s(:,2).^2+n1s(:,3).^2));
    if (min(normn1s) < 1e-5) % Check that the normal we formed is ok
        n1s=[zeros(N,1) -Xts(3:3:end) Xts(2:3:end)];
    end
    n2s=cross((reshape(Xts,3,N))',n1s);
    K=zeros(3*N,2*(N-1));
    K(1:3:3*N,1:N-1)=n1s(:,1).*chebyshevmat;
    K(2:3:3*N,1:N-1)=n1s(:,2).*chebyshevmat;
    K(3:3:3*N,1:N-1)=n1s(:,3).*chebyshevmat;
    K(1:3:3*N,N:2*(N-1))=n2s(:,1).*chebyshevmat;
    K(2:3:3*N,N:2*(N-1))=n2s(:,2).*chebyshevmat;
    K(3:3:3*N,N:2*(N-1))=n2s(:,3).*chebyshevmat;
    K=Dinv*K;
    Kt=zeros(3*N,2*(N-1));
    Kt(1:3:3*N,:)=K(1:3:3*N,:).*w0';
    Kt(2:3:3*N,:)=K(2:3:3*N,:).*w0';
    Kt(3:3:3*N,:)=K(3:3:3*N,:).*w0';
    Kt=Kt';
end