% Compute the matrix X assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
function [X,XMat]=XConnectedNetwork(Connections,nFib,N,L,ell,...
   paths,DOFs,IntegrationMatrix,makePlot)
    Nx = N+1;
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    X = zeros(Nx*nFib,3);
    NLinks = nnz(Connections(:,5));

    % Chebyshev points of first filament
    X(1:Nx,:) = IntegrationMatrix{1}*DOFs(1:N,:);
    if (nargout>1)
        XMat = eye(nFib*N+NLinks+1 + Nx*nFib); % The positions in the top and DOFs in the bottom
        XMat(1:Nx,nFib*Nx+(1:N))=IntegrationMatrix{1};
    end
    nPaths=length(paths);
    LinkNum=0;
    for iPath=1:nPaths
        FilsInPath = paths{iPath};
        for j=2:length(FilsInPath)
            jFib = FilsInPath(j);
            Upstream = FilsInPath(j-1);
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
            if (ConnRow(5)>0) % Cross link
                LinkNum=LinkNum+1;
                LinkVec=DOFs(N*nFib+LinkNum,:);
            end
            X((jFib-1)*Nx+(1:Nx),:) = IntegrationMatrix{jFib}*DOFs((jFib-1)*N+(1:N),:);
            X((jFib-1)*Nx+(1:Nx),:) = X((jFib-1)*Nx+(1:Nx),:)-barymat(FixPt,sX,bX)*X((jFib-1)*Nx+(1:Nx),:)+ ...
                barymat(MotherPt,sX,bX)*X((Upstream-1)*Nx+(1:Nx),:);
            if (ConnRow(5)>0)
                X((jFib-1)*Nx+(1:Nx),:)=X((jFib-1)*Nx+(1:Nx),:)+LinkVec*ell;
            end
            if (nargout>1)
                XFactor = eye(nFib*N+NLinks+1 + Nx*nFib);
                XFactor((jFib-1)*Nx+(1:Nx),Nx*nFib+(jFib-1)*N+(1:N))=IntegrationMatrix{jFib};
                XMat = XFactor*XMat;
                XFactor = eye(nFib*N+NLinks+1 + Nx*nFib);
                XFactor((jFib-1)*Nx+(1:Nx),(jFib-1)*Nx+(1:Nx))=eye(Nx)-repmat(barymat(FixPt,sX,bX),Nx,1);
                XFactor((jFib-1)*Nx+(1:Nx),(Upstream-1)*Nx+(1:Nx)) = repmat(barymat(MotherPt,sX,bX),Nx,1);
                if (ConnRow(5)>0)
                    XFactor((jFib-1)*Nx+(1:Nx),Nx*nFib+N*nFib+LinkNum)=ell;
                end
                XMat = XFactor*XMat;
            end
        end
    end
    if (nargout>1)
        XMat = XMat([1:Nx*nFib end],Nx*nFib+1:end);
        AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
        SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
        SubAvg = [SubAvg ones(Nx*nFib,1)];
        XMat = SubAvg*XMat;
    end
    
    % Remove the average
    AvgPt = zeros(1,3);
    for iFib=1:nFib
        AvgPt = AvgPt+1/(L*nFib)*wX*X((iFib-1)*Nx+(1:Nx),:);
    end
    X = X-AvgPt + DOFs(end,:);

    if (makePlot)
        figure;
        for iFib=1:nFib
            plot3(X((iFib-1)*Nx+(1:Nx),1),X((iFib-1)*Nx+(1:Nx),2),X((iFib-1)*Nx+(1:Nx),3))
            hold on
        end
        
        for iConn=1:size(Connections,1)
            iFib = Connections(iConn,1);
            iS = Connections(iConn,2);
            jFib = Connections(iConn,3);
            jS = Connections(iConn,4);
            pts = [barymat(iS,sX,bX)*X((iFib-1)*Nx+(1:Nx),:); barymat(jS,sX,bX)*X((jFib-1)*Nx+(1:Nx),:)];
            plot3(pts(:,1),pts(:,2),pts(:,3),':ko')
        end
        PlotAspect
    end
end