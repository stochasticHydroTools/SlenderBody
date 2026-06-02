function OmRFD = AssignBranchNodes(g3,BranchIndices,clampedTau)
    OmRFD=g3;
    if (clampedTau>0)
        BranchIndices=[BranchIndices; 0 clampedTau];
    end
    NBranch = size(BranchIndices,1);
    if (NBranch>0)
        % Sort branches 
        [~,branchorder]=sort(BranchIndices(:,2),'ascend');
        % Assign branch nodes
        for iBr=branchorder'
            masternode = BranchIndices(iBr,1);
            if (masternode==0)
                MasterOm = [0 0 0];
            else
                MasterOm = OmRFD(masternode,:);
            end
            slavenode = BranchIndices(iBr,2);
            OmRFD = [OmRFD(1:slavenode-1,:); MasterOm; ...
                OmRFD(slavenode:end,:)];
        end
    end
end
