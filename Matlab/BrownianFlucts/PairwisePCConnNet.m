% Apply preconditioner (precomps elsewhere)
function x = PairwisePCConnNet(RHS,PCMats,AllRHSInds, ...
    AllUpClamped,clampedTau,Nxx,nFib)

    Lam_All = zeros(Nxx,1);
    alphaU_All = zeros(length(RHS)-Nxx,1);
    
    U = RHS(1:Nxx);
    V = RHS(Nxx+1:end);

    % First solve pretending that the master connections are the only ones
    for iC=1:size(PCMats,1)
      
        stAlphInds = AllRHSInds{iC,2};
        UpClamped = AllUpClamped(iC);
        if (UpClamped)
            VThis = V(stAlphInds);
        elseif (clampedTau>0)
            VThis = [V(stAlphInds); zeros(3,1)];
        else
            VThis = [V(stAlphInds); V(end-2:end)];
        end
        UInds = AllRHSInds{iC,1};
        UThis = U(UInds);

        % Form and solve linear system for this pair
        M = PCMats{iC,1};
        K = PCMats{iC,2};
        KWithImp = PCMats{iC,3};
        NMat = PCMats{iC,4};
        KprimeMinvU = K' * (M \ UThis);
        alphaU = NMat*(KprimeMinvU+VThis);
        Lam = M \ (KWithImp*alphaU - UThis);

        Lam_All(UInds)=Lam_All(UInds)+Lam;
        if (UpClamped)
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU;
        elseif (clampedTau>0)
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
        else
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
            alphaU_All(end-2:end)=alphaU_All(end-2:end)+1/(nFib-1)*alphaU(end-2:end);
        end
    end
    x=[Lam_All;alphaU_All];
end