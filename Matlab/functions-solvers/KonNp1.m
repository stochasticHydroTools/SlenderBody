function K = KonNp1(Xs3,XonNp1Mat,I)
    [N,~]=size(Xs3);
    TauCross = zeros(3*N);
    for iR=1:N
        inds = (iR-1)*3+1:iR*3;
        TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
    end
    K = [XonNp1Mat(1:3*(N+1),1:3*N)*-TauCross I];
end