function [sNp1,bNp1,XonNp1Mat,RNp1ToN,DNp1,I,BendingEnergyMatrix_Np1,WTilde_Np1] = ...
    InitializeXMatrices(Nx,s,b,L,Eb,TrkLoc)
    

    % Hydrodynamics
    % NupsampleHydro = 200;%ceil(2/a);
    % [sup,wup,~] = chebpts(NupsampleHydro, [0 L],2);
    % RupsampleHydro = stackMatrix(barymat(sup,sNp1,bNp1));
    % WUpHydro = stackMatrix(diag(wup));
    % BDCell = repmat({RupsampleHydro},nFib,1);
    % RupsampleHydro_BD = blkdiag(BDCell{:});
    % BDCell = repmat({WUpHydro},nFib,1);
    % WUpHydro_BD = blkdiag(BDCell{:});
    % BDCell = repmat({WTilde_Np1_Inverse},nFib,1);
    % WTInv_BD = blkdiag(BDCell{:});
    % AllbS_Np1 = precomputeStokesletInts(sNp1,L,a,N+1,1);
    % AllbD_Np1 = precomputeDoubletInts(sNp1,L,a,N+1,1);
    % NForSmall = 8; % # of pts for R < 2a integrals for exact RPY
    % eigThres = 1e-3;
end