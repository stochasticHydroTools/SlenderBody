% Function to compute the Bishop frame by solving the ODE
% da/ds = (tau cross tau_s) cross a
% a(L/2) = D1mid
% Returns the bishop frame (a,b) and material frame (D1,D2). The other axis
% is Xs in both cases.
function [bishA,bishB,D1,D2] = computeBishopFrame(N,Xs,Xss,Dinv,BMP,theta,D1mid)
    eyeC = zeros(3*N,3);
    for iC=1:N
        eyeC(3*iC-2:3*iC,:)=eye(3);
    end
    AllCPMat2 = zeros(3*N); % Matrix which gives (Xs x Xss) x a at all pts
    for iPt=1:N
        AllCPMat2(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt)=CPMatrix(cross(Xs(iPt,:),Xss(iPt,:)));
    end
    Matrix2 = [eye(3*N)-stackMatrix(Dinv)*AllCPMat2 eyeC; stackMatrix(BMP) zeros(3)];
    RHS2 = [zeros(3*N,1); D1mid];
    a2=Matrix2\RHS2;
    allas = reshape(a2(1:3*N),3,N)';
    bishA = allas./sqrt(sum(allas.*allas,2)); % NORMALIZE
    bishB = cross(Xs,bishA);
    bishB = bishB./sqrt(sum(bishB.*bishB,2));
    % Material frame from Bishop frame
    D1 = bishA.*cos(theta)+bishB.*sin(theta);
    D2 = -bishA.*sin(theta)+bishB.*cos(theta);
end
    
    
        
        