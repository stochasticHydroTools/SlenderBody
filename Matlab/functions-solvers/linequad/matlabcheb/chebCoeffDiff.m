function Df = chebCoeffDiff(fhat,Nz)
    Df = zeros(size(fhat));
    Df(Nz-1) = 2*(Nz-1)*fhat(Nz);
    for  j = 2:Nz-1
        Df(Nz-j) = Df(Nz-j+2) + 2*(Nz-j)*fhat(Nz-j+1);
    end
    Df(1)=Df(1)/2;
end