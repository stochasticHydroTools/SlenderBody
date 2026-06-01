function Kalpha = KApplyConnNet(Alpha,Xt,XFcn,InvXFcn,BranchIndices,clampedTau)
    if (size(Xt,2)==1)
        Xt = reshape(Xt,3,[])';
    end
    if (size(Alpha,2)==1)
        Alpha = reshape(Alpha,3,[])';
    end
    TausAndXBar = InvXFcn(Xt);
    Alpha=AssignBranchNodes(Alpha,BranchIndices,clampedTau);
    TauVelocity = zeros(size(TausAndXBar));
    for iR =1:size(TauVelocity,1)-1
        TauVelocity(iR,:) =  cross(Alpha(iR,:),TausAndXBar(iR,:));
    end
    if (clampedTau>0)
        % The matrix for all the taus (incl links) to evolve
        TauVelocity(end,:) =  cross(Alpha(end,:),TausAndXBar(end,:));
    else
        % The COM
        TauVelocity(end,:)=Alpha(end,:);
    end
    Kalpha = XFcn(TauVelocity);
end