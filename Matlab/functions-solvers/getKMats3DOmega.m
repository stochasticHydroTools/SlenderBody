% Function to compute the matrix K. 
function [K,Kt,CMat]=getKMats3DOmega(Xts,L,N,I,wIt,Dinv,s2,W,b2,U,clamp0,s)
    % Upsample
    lstsquares=1;
    Oversamp=length(W)/(3*N);
    CMat = zeros(Oversamp*3*N);
    CMatN = zeros(3*N);
    for iR=1:N
        inds = (iR-1)*3+1:iR*3;
        tau = Xts(inds);
        CMatN(inds,inds)=CPMatrix(tau);
    end
    % projection
    ProjXs = -CMatN*CMatN;
    Xsup = reshape(U*Xts,3,[])';
    Xsup = Xsup./sum(Xsup.*Xsup,2);
    for iR=1:Oversamp*N
        inds = (iR-1)*3+1:iR*3;
        tau = Xsup(iR,:);
        CMat(inds,inds)=CPMatrix(tau);
    end
    B0 = stackMatrix(barymat(0,s2,b2));
    BL = stackMatrix(barymat(L,s2,b2));
    R = (U'*W*U) \ (U'*W);
    % Define J, etc.
    J = -(eye(Oversamp*3*N)-repmat(B0,Oversamp*N,1))*Dinv*CMat;
    Jst = CMat*(eye(Oversamp*3*N)-repmat(BL,Oversamp*N,1))*Dinv;
    %K = R*J*U*Lmat;
    %Kt = Lmat'*U'*W*Jst*U;
    if (lstsquares)
        K = R*J*U;%*ProjXs;
        Kt = U'*W*Jst*U;
    end
    if (~clamp0)
        K = [K I];
        Kt = [Kt; wIt];
    end
end