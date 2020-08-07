function [K,Kt]=getKMats3D(Xts,chebyshevmat,w0,N)
    [s,~,b]=chebpts(N,[0 sum(w0)],1);
    [su,~,bu]=chebpts(2*N,[0 sum(w0)],2);
    Rup = barymat(su,s,b);
    Rdwn = barymat(s,su,bu);
    
    [theta,phi,~] = cart2sph(Rup*Xts(1:3:end),Rup*Xts(2:3:end),Rup*Xts(3:3:end));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1s=[-sin(theta) cos(theta) 0*theta];
    n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];

    K=zeros(3*N,2*(N-1));
    Dinvup = pinv(diffmat(2*N,1,[0 sum(w0)],'chebkind2'));
    IndefInts11 = Dinvup*(n1s(:,1).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts12 = Dinvup*(n1s(:,2).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts13 = Dinvup*(n1s(:,3).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts21 = Dinvup*(n2s(:,1).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts22 = Dinvup*(n2s(:,2).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts23 = Dinvup*(n2s(:,3).*(Rup*chebyshevmat(:,1:N-1)));
    
    % To fix the constant to zero by finding value at s = 0 and setting to 0
    IndefInts11 = IndefInts11-IndefInts11(1,:);
    IndefInts12 = IndefInts12-IndefInts12(1,:);
    IndefInts13 = IndefInts13-IndefInts13(1,:);
    IndefInts21 = IndefInts21-IndefInts21(1,:);
    IndefInts22 = IndefInts22-IndefInts22(1,:);
    IndefInts23 = IndefInts23-IndefInts23(1,:);
    
    K(1:3:3*N,1:N-1)=Rdwn*IndefInts11;
    K(2:3:3*N,1:N-1)=Rdwn*IndefInts12;
    K(3:3:3*N,1:N-1)=Rdwn*IndefInts13;
    K(1:3:3*N,N:2*(N-1))=Rdwn*IndefInts21;
    K(2:3:3*N,N:2*(N-1))=Rdwn*IndefInts22;
    K(3:3:3*N,N:2*(N-1))=Rdwn*IndefInts23;
    
    Kt=zeros(3*N,2*(N-1));
    Kt(1:3:3*N,:)=K(1:3:3*N,:).*w0';
    Kt(2:3:3*N,:)=K(2:3:3*N,:).*w0';
    Kt(3:3:3*N,:)=K(3:3:3*N,:).*w0';
    Kt=Kt';
end