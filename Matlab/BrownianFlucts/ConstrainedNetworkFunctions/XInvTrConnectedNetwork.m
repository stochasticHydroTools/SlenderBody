% Compute the matrix X assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 

% Split this into two functions
function X = XInvTrConnectedNetwork(DOFs,MasterConnections,...
    SlaveConnections,LeadIndicesByFib,Nx,nFib,L,RegGridMatrix,DiffMatrix,clamp0)

    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    XIrreg = zeros(nFib*Nx,3);
    TauStart = ones(nFib,1);
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(DiffMatrix{iFib},1);
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
            LinkVec=DOFs(TauStart(end)-1+LinkNum,:);
            XIrreg(XIndDwn,:)=LinkVec/ConnRow(6);
            XIrreg(XIndUp,:)=-LinkVec/ConnRow(6);
        end
    end

    % Compute the center and tangent vectors on each fib
    if (clamp0)
        X = zeros(Nx*nFib,3);
    else
        X = 1/(L*nFib)*repmat(wX',nFib,1).*DOFs(end,:);
    end
    for iFib=1:nFib
        LeadIndices = LeadIndicesByFib{iFib};
        XInds = Nx*(iFib-1)+(1:Nx);
        DOFInds = TauStart(iFib):TauStart(iFib+1)-1;
        XIrreg(XInds(LeadIndices),:)=XIrreg(XInds(LeadIndices),:)+...
            DiffMatrix{iFib}'*DOFs(DOFInds,:);
        X(XInds,:)=X(XInds,:)+(RegGridMatrix{iFib}') \ XIrreg(XInds,:);
    end
end