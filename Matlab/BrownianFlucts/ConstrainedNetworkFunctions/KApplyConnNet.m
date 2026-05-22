function Kalpha = KApplyConnNet(Alpha,Xt,XFcn,InvXFcn,BranchIndices)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(Alpha,2)==1)
        Alpha = reshape(Alpha,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);
    NBranch=size(BranchIndices,1);
    if (NBranch>0)
        % Sort branches 
        [~,branchorder]=sort(BranchIndices(:,2),'ascend');
        % Assign branch nodes
        for iBr=branchorder'
            masternode = BranchIndices(iBr,1);
            slavenode = BranchIndices(iBr,2);
            Alpha = [Alpha(1:slavenode-1,:); Alpha(masternode,:); ...
                Alpha(slavenode:end,:)];
        end
    end
    TauVelocity = zeros(size(TausAndXBar));
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(TauVelocity,1)-1
        TauVelocity(iR,:) =  cross(Alpha(iR,:),TausAndXBar(iR,:));
    end
    % The COM
    TauVelocity(end,:)=Alpha(end,:);
    Kalpha = XFcn(TauVelocity);
end