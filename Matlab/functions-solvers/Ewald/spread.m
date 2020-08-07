function [gridfx, gridfy, gridfz] = spread(S,forces,Nx,Ny,Nz)
    gridf=full(S*forces);
    gridfx=permute(reshape(gridf(:,1),Nx,Ny,Nz),[2 1 3]);
    gridfy=permute(reshape(gridf(:,2),Nx,Ny,Nz),[2 1 3]);
    gridfz=permute(reshape(gridf(:,3),Nx,Ny,Nz),[2 1 3]);
end