function Kalpha = KApplyConnNet(Alpha,Xt,XFcn,InvXFcn,BranchIndices)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(Alpha,2)==1)
        Alpha = reshape(Alpha,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);
    Alpha=AssignBranchNodes(Alpha,BranchIndices);
    TauVelocity = zeros(size(TausAndXBar));
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(TauVelocity,1)-1
        TauVelocity(iR,:) =  cross(Alpha(iR,:),TausAndXBar(iR,:));
    end
    % The COM
    TauVelocity(end,:)=Alpha(end,:);
    Kalpha = XFcn(TauVelocity);
end