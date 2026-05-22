function XTLam=XTrConnectedNetwork(Lams,MasterConnections,SlaveConnections,...
    Nx,nFib,L,RegGridMatrix,IntegrationMatrix)
    % Apply X^T to X:
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    SubAvgTX = zeros(Nx*nFib+1,3);
    TauStart = ones(nFib,1);
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(IntegrationMatrix{iFib},2);
    end
    NLinks = nnz(SlaveConnections(:,5))+nnz(MasterConnections(:,5));
    LinkNum=NLinks;
    for jFib=1:nFib
        SubAvgTX((jFib-1)*Nx+(1:Nx),:)=Lams((jFib-1)*Nx+(1:Nx),:)-...
            1/(L*nFib)*wX'.*sum(Lams);
    end
    SubAvgTX(end,:)=sum(Lams);
    % Map to regular Chebyshev grid
    for iFib=1:nFib
        XInds=Nx*(iFib-1)+(1:Nx);
        SubAvgTX(XInds,:) = RegGridMatrix{iFib}'*SubAvgTX(XInds,:);
    end
    XTLam = zeros(TauStart(end)+NLinks,3);
    XTLam(end,:)=SubAvgTX(end,:);
    % Slave connections 
    for iC=size(SlaveConnections,1):-1:1
        ConnRow = SlaveConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConnRow(2);
        DownFib = ConnRow(3);
        PtOnDown = ConnRow(4);
        XInd = Nx*(DownFib-1)+PtOnDown;
        XIndUp =  Nx*(UpFib-1)+PtOnUp;
        XTLam(TauStart(end)-1+LinkNum,:)=ConnRow(6)*SubAvgTX(XInd,:);
        SubAvgTX(XIndUp,:)=SubAvgTX(XIndUp,:)+SubAvgTX(XInd,:);
        LinkNum=LinkNum-1;
    end

    % Master connections
    for iC=size(MasterConnections,1):-1:1
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConnRow(2);
        DownFib = ConnRow(3);
        PtOnDown = ConnRow(4);
        DOFInds = TauStart(DownFib):TauStart(DownFib+1)-1;
        LeadIndices = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==DownFib,4));
        XInds = Nx*(DownFib-1)+(1:Nx);
        XIndUp =  Nx*(UpFib-1)+PtOnUp;
        if (ConnRow(5)>0)
            XTLam(TauStart(end)-1+LinkNum,:)=ConnRow(6)*sum(SubAvgTX(XInds(LeadIndices),:));
            LinkNum=LinkNum-1;
        end
        SubAvgTX(XIndUp,:)=SubAvgTX(XIndUp,:)+sum(SubAvgTX(XInds(LeadIndices),:));
        SubAvgTX(XInds(PtOnDown),:)=SubAvgTX(XInds(PtOnDown),:)-sum(SubAvgTX(XInds(LeadIndices),:));
        XTLam(DOFInds,:)=IntegrationMatrix{DownFib}'*SubAvgTX(XInds(LeadIndices),:);
    end

    % End with the master filament
    AnchorFil = MasterConnections(1,1);
    DOFInds = TauStart(AnchorFil):TauStart(AnchorFil+1)-1;
    % Remove slave nodes on first filament
    LeadIndices = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==AnchorFil,4));
    XInds = Nx*(AnchorFil-1)+LeadIndices;
    XTLam(DOFInds,:) = IntegrationMatrix{AnchorFil}'*SubAvgTX(XInds,:);
end