% Compute/apply matrix X^(-1)

function [DOFs,XInvMat] = XInvConnectedNetwork(X,MasterConnections,...
    SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrixInv,DiffMatrix,clamp0)
    
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    TauStart = ones(nFib,1);
    NLinks = nnz(SlaveConnections(:,5))+nnz(MasterConnections(:,5));
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(DiffMatrix{iFib},1);
    end
    DOFs = zeros(TauStart(end)-clamp0+NLinks,3);
    if (nargout>1)
        XInvMat = zeros(TauStart(end)+NLinks-clamp0,nFib*Nx+1-clamp0);
        XToIrreg = zeros(nFib*Nx+1-clamp0,nFib*Nx);
        if (~clamp0)
            XInvMat(end,end)=1;
        end
    end
    XIrreg = zeros(nFib*Nx,3);
    % Compute the center and tangent vectors on each fib
    for iFib=1:nFib
        LeadIndices = LeadIndicesByFib{iFib};
        XInds = Nx*(iFib-1)+(1:Nx);
        DOFInds = TauStart(iFib):TauStart(iFib+1)-1;
        if (~clamp0)
            DOFs(end,:)=DOFs(end,:)+1/(L*nFib)*wX*X(XInds,:);
        end
        XIrreg(XInds,:)=RegGridMatrixInv{iFib}*X(XInds,:);
        DOFs(DOFInds,:) = DiffMatrix{iFib}*XIrreg(XInds(LeadIndices),:);
        if (nargout>1)
            if (~clamp0)
                XToIrreg(end,XInds)=1/(L*nFib)*wX;
            end
            XToIrreg(XInds,XInds)=RegGridMatrixInv{iFib};
            XInvMat(DOFInds,XInds(LeadIndices))=DiffMatrix{iFib};
        end
    end

    % Compute the link vectors 
    LinkNum=0;
    AllConnections=[MasterConnections;SlaveConnections];
    for iC=1:size(AllConnections,1)
        ConnRow = AllConnections(iC,:);
        if (ConnRow(5)>0) % Cross link
            XIndUp =  Nx*(ConnRow(1)-1)+ConnRow(2);
            XIndDwn =  Nx*(ConnRow(3)-1)+ConnRow(4);
            LinkNum=LinkNum+1;
            LinkVec=(XIrreg(XIndDwn,:)-XIrreg(XIndUp,:))/ConnRow(6);
            DOFs(TauStart(end)-1+LinkNum,:)=LinkVec;
            if (nargout>1)
                XInvMat(TauStart(end)-1+LinkNum,XIndUp)=-1/ConnRow(6);
                XInvMat(TauStart(end)-1+LinkNum,XIndDwn)=1/ConnRow(6);
            end
        end
    end
    if (nargout>1)
        XInvMat = XInvMat*XToIrreg;
    end
end