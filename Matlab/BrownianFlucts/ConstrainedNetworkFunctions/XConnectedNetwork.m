% Compute/apply the matrix X 
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
function [X,XMat]=XConnectedNetwork(DOFs,MasterConnections,SlaveConnections,...
    Nx,nFib,L,RegGridMatrix,IntegrationMatrix,clamp0,makePlot)
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    X = zeros(Nx*nFib,3);
    TauStart = ones(nFib,1);
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(IntegrationMatrix{iFib},2);
    end
    nDOFs = size(DOFs,1);

    AnchorFil = MasterConnections(1,1);
    DOFInds = TauStart(AnchorFil):TauStart(AnchorFil+1)-1;
    % Remove slave nodes on first filament
    LeadIndices = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==AnchorFil,4));
    XInds = Nx*(AnchorFil-1)+LeadIndices;
    X(XInds,:) = IntegrationMatrix{AnchorFil}*DOFs(DOFInds,:);
    if (nargout>1)
        XMat = eye(nDOFs + Nx*nFib); % The positions in the top and DOFs in the bottom
        XMat(XInds,Nx*nFib+DOFInds)=IntegrationMatrix{AnchorFil};
    end
    LinkNum=0;
    % Use the master connections to assign the lead nodes of the subsequent
    % filaments
    for iC=1:size(MasterConnections,1)
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConnRow(2);
        DownFib = ConnRow(3);
        PtOnDown = ConnRow(4);
        DOFInds = TauStart(DownFib):TauStart(DownFib+1)-1;
        LeadIndices = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==DownFib,4));
        XInds = Nx*(DownFib-1)+(1:Nx);
        XIndUp =  Nx*(UpFib-1)+PtOnUp;
        X(XInds(LeadIndices),:) = IntegrationMatrix{DownFib}*DOFs(DOFInds,:);
        X(XInds(LeadIndices),:) = X(XInds(LeadIndices),:)-X(XInds(PtOnDown),:)+X(XIndUp,:);
        if (ConnRow(5)>0)
            LinkNum=LinkNum+1;
            LinkVec=DOFs(TauStart(end)-1+LinkNum,:);
            X(XInds(LeadIndices),:)=X(XInds(LeadIndices),:)+LinkVec*ConnRow(6);
        end
        if (nargout>1)
            XFactor = eye(nDOFs + Nx*nFib);
            XFactor(XInds(LeadIndices),Nx*nFib+DOFInds)=IntegrationMatrix{DownFib};
            XMat = XFactor*XMat;
            XFactor = eye(nDOFs + Nx*nFib);
            XFactor(XInds(LeadIndices),XInds(PtOnDown)) = XFactor(XInds(LeadIndices),XInds(PtOnDown))-1;
            XFactor(XInds(LeadIndices),XIndUp) = 1;
            if (ConnRow(5)>0)
                XFactor(XInds(LeadIndices),Nx*nFib+TauStart(end)-1+LinkNum)=ConnRow(6);
            end
            XMat = XFactor*XMat;
        end
    end

    % Slave connections 
    for iC=1:size(SlaveConnections,1)
        LinkNum=LinkNum+1;
        ConnRow = SlaveConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConnRow(2);
        DownFib = ConnRow(3);
        PtOnDown = ConnRow(4);
        XInd = Nx*(DownFib-1)+PtOnDown;
        XIndUp =  Nx*(UpFib-1)+PtOnUp;
        LinkVec = DOFs(TauStart(end)-1+LinkNum,:);
        X(XInd,:) = X(XIndUp,:)+LinkVec*ConnRow(6);
        if (nargout>1)
            XFactor = eye(nDOFs + Nx*nFib);
            XFactor(XInd,XInd) = 0;
            XFactor(XInd,XIndUp) = 1;
            XFactor(XInd,Nx*nFib+TauStart(end)-1+LinkNum)=ConnRow(6);
            XMat = XFactor*XMat;
        end
    end

    % Map everything to a regular chebyshev grid
    for iFib=1:nFib
        XInds=Nx*(iFib-1)+(1:Nx);
        X(XInds,:) = RegGridMatrix{iFib}*X(XInds,:);
        if (nargout > 1)
            XFactor = eye(nDOFs + Nx*nFib);
            XFactor(XInds,XInds) = RegGridMatrix{iFib};
            XMat = XFactor*XMat;
        end
    end

    % Remove the average
    if (~clamp0)
        AvgPt = zeros(1,3);
        for iFib=1:nFib
            AvgPt = AvgPt+1/(L*nFib)*wX*X((iFib-1)*Nx+(1:Nx),:);
        end
        X = X-AvgPt + DOFs(end,:);
        if (nargout>1)
            XMat = XMat([1:Nx*nFib end],Nx*nFib+1:end);
            AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
            SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
            SubAvg = [SubAvg ones(Nx*nFib,1)];
            XMat = SubAvg*XMat;
            XMatEr=max(abs(XMat*DOFs-X))
        end
    else
        if (nargout>1)
            XMat = XMat(1:Nx*nFib,Nx*nFib+1:end);
            XMatEr=max(abs(XMat*DOFs-X))
        end
    end

    if (makePlot)
        figure;
        for iFib=1:nFib
            plot3(X((iFib-1)*Nx+(1:Nx),1),X((iFib-1)*Nx+(1:Nx),2),X((iFib-1)*Nx+(1:Nx),3))
            hold on
        end
        
        % Fix this later
        % for iConn=1:size(Connections,1)
        %     iFib = Connections(iConn,1);
        %     iS = Connections(iConn,2);
        %     jFib = Connections(iConn,3);
        %     jS = Connections(iConn,4);
        %     pts = [barymat(iS,sX,bX)*X((iFib-1)*Nx+(1:Nx),:); barymat(jS,sX,bX)*X((jFib-1)*Nx+(1:Nx),:)];
        %     plot3(pts(:,1),pts(:,2),pts(:,3),':ko')
        % end
        % PlotAspect
    end
end