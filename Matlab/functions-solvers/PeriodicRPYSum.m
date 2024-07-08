% Compute the action of the non-local mobility matrix in a periodic DOMAIN. 
% This code just performs an O(N^2) sum
function U = PeriodicRPYSum(nFib,X,F,Lds,xi,g,a,mu,Rupsample,wup,WTildeInverse,directquad)
    FINUUFT=1; % use FINUFFT
    % Compile list of upsampled points and forces
    [Nup,Nx]=size(Rupsample);
    if (directquad)
        Nup=Nx;
    end
    Forces = zeros(Nup*nFib,3);
    UpPoints = zeros(Nup*nFib,3);
    U = zeros(Nx*nFib,3);
    for iFib=1:nFib
        inds = (iFib-1)*Nx+1:iFib*Nx;
        UpInds = (iFib-1)*Nup+1:iFib*Nup;
        if (directquad)
            Forces(inds,:)=F(inds,:);
            UpPoints(inds,:)=X(inds,:);
        else
            Forces(UpInds,:)=diag(wup)*Rupsample*WTildeInverse*F(inds,:);
            UpPoints(UpInds,:)=Rupsample*X(inds,:);
        end
    end
    % Ewald splitting calculation
    if (FINUUFT)
        velFar = EwaldFarVelFI(UpPoints,Forces,mu,Lds(1),Lds(2),Lds(3),xi,a,g)';
    else
        velFar = EwaldFarVel(UpPoints,Forces,mu,Lds(1),Lds(2),Lds(3),xi,a,g);
    end
    velNear = EwaldNearSum(nFib*Nup,UpPoints,Forces,xi,Lds(1),Lds(2),Lds(3),a,mu,g);
    velEwald=velNear+velFar; % This is on the updampled grid
    % Apply L^2 downsampling to go back to the N point grid
    for iFib=1:nFib
        inds = (iFib-1)*Nx+1:iFib*Nx;
        UpInds = (iFib-1)*Nup+1:iFib*Nup;
        if (directquad)
            U(inds,:)=velEwald(UpInds,:);
        else
            U(inds,:) = WTildeInverse*Rupsample'*diag(wup)*velEwald(UpInds,:);
        end
    end
end