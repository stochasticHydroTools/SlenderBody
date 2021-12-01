function Ds = stackMatrix(D)
    [N,C]=size(D);
    Ds = zeros(3*N,3*C);
    for iD=1:3
        Ds(iD:3:end,iD:3:end)=D;
    end
end