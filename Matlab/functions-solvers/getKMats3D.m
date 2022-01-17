% Function to compute the matrix K. 
function [K,Kt,nPolys]=getKMats3D(Xts,chebyshevmat,w0,N,I,wIt,Kttype,rigid)
    [s,~,b]=chebpts(N,[0 sum(w0)],1);
    [su,wu,bu]=chebpts(2*N,[0 sum(w0)],2);
    Rup = barymat(su,s,b);
    Rdwn = barymat(s,su,bu);
    
    [theta,phi,~] = cart2sph(Rup*Xts(1:3:end),Rup*Xts(2:3:end),Rup*Xts(3:3:end));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1s=[-sin(theta) cos(theta) 0*theta];
    n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];

    Dinvup = pinv(diffmat(2*N,1,[0 sum(w0)],'chebkind2'));
    IndefInts11 = Dinvup*((n1s(:,1)).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts12 = Dinvup*((n1s(:,2)).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts13 = Dinvup*((n1s(:,3)).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts21 = Dinvup*((n2s(:,1)).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts22 = Dinvup*((n2s(:,2)).*(Rup*chebyshevmat(:,1:N-1)));
    IndefInts23 = Dinvup*((n2s(:,3)).*(Rup*chebyshevmat(:,1:N-1)));
    
    % To fix the constant to zero by finding value at s = 0 and setting to 0
    IndefInts11 = IndefInts11-IndefInts11(1,:);
    IndefInts12 = IndefInts12-IndefInts12(1,:);
    IndefInts13 = IndefInts13-IndefInts13(1,:);
    IndefInts21 = IndefInts21-IndefInts21(1,:);
    IndefInts22 = IndefInts22-IndefInts22(1,:);
    IndefInts23 = IndefInts23-IndefInts23(1,:);
%     
    if (rigid)
        nPolys = 1;
    else
        nPolys = N-1;
    end
    %disp('K set up for rigid body motion!')
    J = zeros(6*N,2*nPolys);
    J(1:3:6*N,1:nPolys)=IndefInts11(:,1:nPolys);
    J(2:3:6*N,1:nPolys)=IndefInts12(:,1:nPolys);
    J(3:3:6*N,1:nPolys)=IndefInts13(:,1:nPolys);
    J(1:3:6*N,nPolys+1:2*nPolys)=IndefInts21(:,1:nPolys);
    J(2:3:6*N,nPolys+1:2*nPolys)=IndefInts22(:,1:nPolys);
    J(3:3:6*N,nPolys+1:2*nPolys)=IndefInts23(:,1:nPolys);
    
    W = diag(reshape([wu; wu; wu],6*N,1));
    U=zeros(6*N,3*N);
    R = zeros(3*N,6*N);
    for iD=1:3
        U(iD:3:end,iD:3:end)=Rup;
        R(iD:3:end,iD:3:end)=Rdwn;
    end
    if (Kttype=='U')
        Kt = J'*W*U;
    else
        W1 = diag(reshape([w0; w0; w0],3*N,1));
        Kt = J'*R'*W1;
    end
    %K = R*J;
    %disp('Downsampled K')
    K = (U'*W*U) \ (U'*W*J);
    K = [K I];   Kt = [Kt; wIt];
end