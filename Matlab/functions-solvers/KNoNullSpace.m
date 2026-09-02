function [K,NormalsMat] = KNoNullSpace(Xs3,XonNp1Mat)
    [N,~]=size(Xs3);
    TauCross = zeros(3*N);
    NormalsMat = zeros(3*N,2*N);
    for iR=1:N
        inds = (iR-1)*3+1:iR*3;
        TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
        if (abs(Xs3(iR,3)) < 0.99)
            n1 = [0 0 1];
        else 
            n1 = [1 0 0];
        end
        n2 = cross(Xs3(iR,:),n1);
        NormalsMat(inds,2*iR-1)=n1;
        NormalsMat(inds,2*iR)=n2;
    end
    K = [XonNp1Mat(1:3*(N+1),1:3*N)*-TauCross*NormalsMat XonNp1Mat(:,end-2:end)];
    NormalsMat = [NormalsMat zeros(3*N,3); zeros(3,2*N) eye(3)];
end