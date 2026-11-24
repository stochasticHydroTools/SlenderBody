function [KTogether,KTogetherInv] = KWithLink(Xt,XMat,InvXMat,...
    AssignMat,BranchIndices,clampedTau)
    TausAndXBar = InvXMat*Xt;
    if (clampedTau <=0)
        Tau3 = reshape(TausAndXBar(1:end-3),3,[])';
        TauVelocity = zeros(3*size(Tau3,1)+3);
        InvTauVelocity = zeros(3*size(Tau3,1)+3);
    else
        Tau3 = reshape(TausAndXBar,3,[])';
        TauVelocity = zeros(3*size(Tau3,1));
        InvTauVelocity = zeros(3*size(Tau3,1));
    end
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
        InvTauVelocity(inds,inds) = CMat;
    end
    % The COM
    if (clampedTau<=0)
        TauVelocity(end-2:end,end-2:end)=eye(3);
        InvTauVelocity(end-2:end,end-2:end)=eye(3);
    end
    KTogether = XMat*TauVelocity*AssignMat;
    KTogetherInv = AssignMat'*InvTauVelocity*InvXMat;
    OmegaFromProjections = eye(length(TausAndXBar));
    for iBr=1:size(BranchIndices,1)
        % Construct local ONB
        tauM = Tau3(BranchIndices(iBr,1),:);
        tauD = Tau3(BranchIndices(iBr,2),:);
        crossMD = cross(tauM,tauD);
        % 3 x 3 matrix going from P1Omega+P2Omega -> (Omega1,Omega2,Omega3)
        InvertMe = [tauM'-tauD'*dot(tauM,tauD) tauD'-tauM'*dot(tauD,tauM) ...
            2*crossMD'];
        ActOmega = [tauM' tauD' crossMD'];
        brInds = 3*BranchIndices(iBr,1)+(-2:0);
        OmegaFromProjections(brInds,brInds)=ActOmega*InvertMe^(-1);
    end
    brInds=[];
    for iBr=1:size(BranchIndices,1)
        brInds = [brInds;3*BranchIndices(iBr,2)+(-2:0)'];
    end
    if (clampedTau>0)
        brInds = [brInds;(3*clampedTau-2:3*clampedTau)'];
    end
    OmegaFromProjections(brInds,:)=[];
    OmegaFromProjections(:,brInds)=[];
    KTogetherInv=OmegaFromProjections*KTogetherInv;
end