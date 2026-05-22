function OmegaPerp = KInvApplyConnNet(U,Xt,InvXFcn,BranchIndices)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(U,2)==1)
        U = reshape(U,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);
    Alpha = InvXFcn(U);
    OmegaPerp = zeros(size(TausAndXBar,1),3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(TausAndXBar,1)-1
        OmegaPerp(iR,:) =  -cross(Alpha(iR,:),TausAndXBar(iR,:));
    end
    % The COM
    OmegaPerp(end,:)=Alpha(end,:);
    % Overwrite the branches
    for iBr=1:size(BranchIndices,1)
        % Construct local ONB
        tauM = TausAndXBar(BranchIndices(iBr,1),:);
        tauD = TausAndXBar(BranchIndices(iBr,2),:);
        OmTot = OmegaPerp(BranchIndices(iBr,1),:)+OmegaPerp(BranchIndices(iBr,2),:);
        crossMD = cross(tauM,tauD);
        % 3 x 3 matrix going from P1Omega+P2Omega -> (Omega1,Omega2,Omega3)
        InvertMe = [tauM'-tauD'*dot(tauM,tauD) tauD'-tauM'*dot(tauD,tauM) ...
            2*crossMD'];
        ActOmega = [tauM' tauD' crossMD'];
        OmTrue = ActOmega*(InvertMe \ OmTot');
        OmegaPerp(BranchIndices(iBr,1),:)=OmTrue;
    end
    OmegaPerp(BranchIndices(:,2),:)=[];
end