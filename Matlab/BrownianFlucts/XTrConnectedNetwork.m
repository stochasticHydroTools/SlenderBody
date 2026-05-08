function XTLam=XTrConnectedNetwork(Connections,nFib,N,L,ell,...
   paths,Lams,IntegrationMatrix)
    % Apply X^T to X:
    Nx = N+1;
    SubAvgTX = zeros(Nx*nFib+1,3);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    NLinks = nnz(Connections(:,5));
    LinkNum=NLinks;
    nPaths=length(paths);
    for jFib=1:nFib
        SubAvgTX((jFib-1)*Nx+(1:Nx),:)=Lams((jFib-1)*Nx+(1:Nx),:)-...
            1/(L*nFib)*wX'.*sum(Lams);
    end
    SubAvgTX(end,:)=sum(Lams);
    XTLam = zeros(N*nFib+NLinks+1,3);
    XTLam(end,:)=SubAvgTX(end,:);
    for iPath=nPaths:-1:1
        FilsInPath = paths{iPath};
        for j=length(FilsInPath):-1:2
            jFib = FilsInPath(j);
            Upstream = FilsInPath(j-1);
            dInds = (jFib-1)*Nx+(1:Nx);
            mInds = (Upstream-1)*Nx+(1:Nx);
            % Find the corresponding row in the connection matrix
            Row = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            ConnRow = Connections(Row,:);
            if (ConnRow(1)==jFib)
                FixPt = ConnRow(2);
                MotherPt = ConnRow(4);
            else
                MotherPt = ConnRow(2);
                FixPt = ConnRow(4);
            end
            if (ConnRow(5)>0)
                XTLam(N*nFib+LinkNum,:)=ell*sum(SubAvgTX(dInds,:));
                LinkNum=LinkNum-1;
            end
            SubAvgTX(mInds,:)=SubAvgTX(mInds,:)...
                    +sum(SubAvgTX(dInds,:)).*barymat(MotherPt,sX,bX)';
            SubAvgTX(dInds,:)=SubAvgTX(dInds,:)...
                -sum(SubAvgTX(dInds,:)).*barymat(FixPt,sX,bX)';
            XTLam((jFib-1)*N+(1:N),:)=IntegrationMatrix{jFib}'*SubAvgTX(dInds,:);
            
        end
    end
    XTLam(1:N,:)=IntegrationMatrix{1}'*SubAvgTX(1:Nx,:);
end