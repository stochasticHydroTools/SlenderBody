function KInv = KInvonNp1(Xs3,InvXonNp1Mat,BMNp1)
    [N,~]=size(Xs3);
    TauCross = zeros(3*N);
    for iR=1:N
        inds = (iR-1)*3+1:iR*3;
        TauCross(inds,inds)=CPMatrix(Xs3(iR,:));
    end
    KInv = [TauCross*InvXonNp1Mat(1:3*N,:); BMNp1];
end