function [K,Kt]=getKMats3D(Xts,chebyshevmat,Dinv,w0,N)
    [theta,phi,~] = cart2sph(Xts(1:3:end),Xts(2:3:end),Xts(3:3:end));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1s=[-sin(theta) cos(theta) zeros(N,1)];
    n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
    normn1s=(sqrt(n1s(:,1).^2+n1s(:,2).^2+n1s(:,3).^2));
    normn2s=(sqrt(n2s(:,1).^2+n2s(:,2).^2+n2s(:,3).^2));
    if ((min(abs(normn1s-1)) > 1e-5) ||(min(abs(normn2s-1)) > 1e-5))
        % Check that the normal we formed is ok
        keyboard
    end
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