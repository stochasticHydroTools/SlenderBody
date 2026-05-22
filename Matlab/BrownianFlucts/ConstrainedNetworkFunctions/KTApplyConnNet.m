% K^T = A^T C^T X^T = A^T (-C) X^T
function KTLam = KTApplyConnNet(Lam,Xt,XTrFcn,InvXFcn,BranchIndices)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(Lam,2)==1)
        Lam = reshape(Lam,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);
    XTLam = XTrFcn(Lam);

    CrossXTLam = zeros(size(XTLam));
    for iR =1:size(CrossXTLam,1)-1
        CrossXTLam(iR,:) =  -cross(XTLam(iR,:),TausAndXBar(iR,:));
    end
    % The COM
    CrossXTLam(end,:)=XTLam(end,:);

    % Overwrite the master branch nodes with the sums
    KTLam = CrossXTLam;
    NBranch=size(BranchIndices,1);
    if (NBranch>0)
        KTLam(BranchIndices(:,1),:)=...
            KTLam(BranchIndices(:,1),:)+KTLam(BranchIndices(:,2),:);
        % Sort branches 
        KTLam(BranchIndices(:,2),:)=[];
    end
end