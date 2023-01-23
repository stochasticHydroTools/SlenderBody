function newXs = rotateTau(Xsin,Omega,dt)
    nOm = sqrt(sum(Omega.*Omega,2));
    % Have to truncate somewhere to avoid instabilities
    k = Omega./nOm;
    k(nOm < 1e-12,:) = 0;
    % Rodriguez formula on the N grid. 
    newXs = Xsin.*cos(nOm*dt)+cross(k,Xsin).*sin(nOm*dt)...
        +k.*sum(k.*Xsin,2).*(1-cos(nOm*dt));
end