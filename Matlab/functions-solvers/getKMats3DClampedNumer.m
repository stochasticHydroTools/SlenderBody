function [K,Kt,nPolys]=getKMats3DClampedNumer(Xts,chebyshevmat,w0,N,I,wIt,Kttype,clamped)
    L = sum(w0);
    [s,~,b]=chebpts(N,[0 L],1);
    [su,wu,bu]=chebpts(2*N,[0 L],2);
    Rup = barymat(su,s,b);
    Rdwn = barymat(s,su,bu);
    clamp0 = clamped(1);
    clampL = clamped(2);
    
    [theta,phi,~] = cart2sph(Rup*Xts(1:3:end),Rup*Xts(2:3:end),Rup*Xts(3:3:end));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1s=[-sin(theta) cos(theta) 0*theta];
    n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
    
    % Modify polynomial basis so that it is 0 at 0 (and 0 at L)
    nPolys=N-1;
    Basis = Rup*chebyshevmat(:,1:N-1);
    if (clamp0)
        Basis = Basis - Basis(1,:);
        Basis(:,1)=[];
        nPolys=nPolys-1;
    end
    if (clampL)
        Basis = Basis - Basis(end,:).*su/L;
        Basis(:,1)=[];
        nPolys=nPolys-1;
    end
    
    Dup = diffmat(2*N,1,[0 sum(w0)],'chebkind2');
    Dinvup = pinv(Dup);
    IndefInts11 = Dinvup*((n1s(:,1)).*Basis);
    IndefInts12 = Dinvup*((n1s(:,2)).*Basis);
    IndefInts13 = Dinvup*((n1s(:,3)).*Basis);
    IndefInts21 = Dinvup*((n2s(:,1)).*Basis);
    IndefInts22 = Dinvup*((n2s(:,2)).*Basis);
    IndefInts23 = Dinvup*((n2s(:,3)).*Basis);
    
    % To fix the constant to zero by finding value at s = 0 and setting to 0
    IndefInts11 = IndefInts11-IndefInts11(1,:);
    IndefInts12 = IndefInts12-IndefInts12(1,:);
    IndefInts13 = IndefInts13-IndefInts13(1,:);
    IndefInts21 = IndefInts21-IndefInts21(1,:);
    IndefInts22 = IndefInts22-IndefInts22(1,:);
    IndefInts23 = IndefInts23-IndefInts23(1,:);
    
    J = zeros(6*N,2*nPolys);
    J(1:3:6*N,1:nPolys)=IndefInts11(:,1:nPolys);
    J(2:3:6*N,1:nPolys)=IndefInts12(:,1:nPolys);
    J(3:3:6*N,1:nPolys)=IndefInts13(:,1:nPolys);
    J(1:3:6*N,nPolys+1:2*nPolys)=IndefInts21(:,1:nPolys);
    J(2:3:6*N,nPolys+1:2*nPolys)=IndefInts22(:,1:nPolys);
    J(3:3:6*N,nPolys+1:2*nPolys)=IndefInts23(:,1:nPolys);
    
    % Fix the velocity at L to zero
    % Same thing for the velocity at L
%     if (clampL)
%         % First set of normals
%         J1ends = J(end-2:end,1:nPolys);
%         r1 = rank(J1ends,1e-10);
%         basisr1 = J1ends(:,1);
%         J1cols = [1];
%         addCol=2;
%         while (rank(basisr1,1e-10) < r1)
%             % Add column if it increases rank 
%             trythis = [basisr1 J1ends(:,addCol)];
%             if (rank(trythis,1e-10) > rank(basisr1,1e-10))
%                 basisr1=trythis;
%                 J1cols = [J1cols addCol];
%             end
%             addCol=addCol+1;
%         end
%         % Subtract multiple of first column from others
%         for iPoly=setdiff(1:nPolys,J1cols)
%             wts = basisr1 \ J1ends(:,iPoly);
%             J(:,iPoly)=J(:,iPoly)-sum(wts'.*J(:,J1cols),2);
%         end
%         % Repeat for second set of normals
%         J2ends = J(end-2:end,nPolys+1:end);
%         r2 = rank(J2ends,1e-10);
%         basisr2 = J2ends(:,1);
%         J2cols = [1];
%         addCol=2;
%         while (rank(basisr2,1e-10) < r2)
%             % Add column if it increases rank 
%             trythis = [basisr2 J2ends(:,addCol)];
%             if (rank(trythis,1e-10) > rank(basisr2,1e-10))
%                 basisr2=trythis;
%                 J2cols = [J2cols addCol];
%             end
%             addCol=addCol+1;
%         end
%         % Subtract multiple of first column from others
%         for iPoly=setdiff(1:nPolys,J2cols)
%             wts = basisr2 \ J2ends(:,iPoly);
%             J(:,nPolys+iPoly)=J(:,nPolys+iPoly)-sum(wts'.*J(:,nPolys+J2cols),2);
%         end
%         J(:,nPolys+J2cols)=[];
%         J(:,J1cols)=[];
%     end
    
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
    if (~clamp0)
        K = [K I]; Kt = [Kt; wIt];
    end
end