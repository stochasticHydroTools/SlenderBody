% K^-T = X^-T C A^(-T) = X^(-T) C A ProjMat' 
function Xperp = KInvTApplyConnNet(OmPerp,Xt,XTrInvFcn,InvXFcn,BranchIndices,clampedTau)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(OmPerp,2)==1)
        OmPerp = reshape(OmPerp,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);


    % Figure out where the branch points are
    NBranch = size(BranchIndices,1);
    if (NBranch>0)
        delInds=[BranchIndices(:,2)];
        if (clampedTau>0)
            delInds=[delInds;clampedTau];
        end
        AllInds = 1:max(BranchIndices(:));
        AllInds(delInds)=[];
        BrInd=[];
        for iBr=1:NBranch
            BrInd=[BrInd;find(AllInds==BranchIndices(iBr,1))];
        end
        % Apply projection matrix to recover Omega1, Omega2, Omega3 
        for iBr=1:NBranch
            % Construct local ONB
            tauM = TausAndXBar(BranchIndices(iBr,1),:);
            tauD = TausAndXBar(BranchIndices(iBr,2),:);
            crossMD = cross(tauM,tauD);
            % 3 x 3 matrix going from P1Omega+P2Omega -> (Omega1,Omega2,Omega3)
            InvertMe = [tauM'-tauD'*dot(tauM,tauD) tauD'-tauM'*dot(tauD,tauM) ...
                2*crossMD'];
            ActOmega = [tauM' tauD' crossMD'];
            OmTrue = InvertMe' \ (ActOmega'*OmPerp(BrInd(iBr,:),:)');
            OmPerp(BrInd(iBr,:),:)=OmTrue;
        end
    end
    if (NBranch>0 || clampedTau>0)
        OmPerp=AssignBranchNodes(OmPerp,BranchIndices,clampedTau);
    end
    
    OmegaPerpCross = zeros(size(TausAndXBar,1),3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(TausAndXBar,1)-1
        OmegaPerpCross(iR,:) =  cross(OmPerp(iR,:),TausAndXBar(iR,:));
    end
    if (clampedTau>0)
        OmegaPerpCross(end,:) =  cross(OmPerp(end,:),TausAndXBar(end,:));
    else
        % The COM
        OmegaPerpCross(end,:)=OmPerp(end,:);
    end
    Xperp = XTrInvFcn(OmegaPerpCross);
end