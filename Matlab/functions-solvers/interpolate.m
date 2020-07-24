function pPtsDP = interpolate(I,pgrid,Nx,Ny,Nz)
    pPtsDP=I'*reshape(permute(pgrid,[2 1 3]),Nx*Ny*Nz,1);
end
