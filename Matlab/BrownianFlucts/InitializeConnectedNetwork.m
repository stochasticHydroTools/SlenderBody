% Initialize assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
function [DOFs,MasterConnections,SlaveConnections, ConstrainedPosNodes,...
  TangentVectorNodes,IntegrationMatrix,DiffMatrix,RegGridMatrix] = ...
  InitializeConnectedNetwork(Connections,nFib,N,L,ell)
    Nx = N+1;
    % Check list
    for p=1:size(Connections,1)
        if (Connections(p,5)==0 && Connections(p,4)~=0)
            warning('Cannot have a branch away from s=0 on daughter, at connection number %d',p)
            Connections(p,4)=0;
        end
    end
    % Check that there are not multiple CLs or branches at the same place
    AllConns = [Connections(:,1:2); Connections(:,3:4)];
    if (size(unique(AllConns, 'rows'), 1) < size(AllConns, 1))
        error('Cannot have two different constraints in the same place')
    end
    NLinks = nnz(Connections(:,5));
    BrInd = find(Connections(:,5)==0);
    nBr = length(BrInd);
    
    % Make connected graph
    G = graph(Connections(:,1),Connections(:,3));
    % Find minimal spanning tree (including edge nodes)
    BranchInds = [Connections(BrInd,1) Connections(BrInd,3)];
    % 2. Force inclusion of specific edges (e.g., edge between 1 and 3)
    % Find the index of the edge you want to force
    forcedEdgeIdx=[];
    for iBr=1:nBr
        forcedEdgeIdx = [forcedEdgeIdx; findedge(G, Connections(BrInd(iBr),1), Connections(BrInd(iBr),3))];
    end
    % 3. Compute the MST
    G.Edges.Weight(forcedEdgeIdx) = -inf; % Set weight to -infinity to force inclusion
    T = minspantree(G);

    % Go through the MST and order the connections. 
    % Then make a separate list of the follower connections. 
    % Identify "end" nodes (leaf nodes where the branch terminates)
    nodes_degree = degree(T);
    nodes_end = find(nodes_degree == 1);
    
    % Find the center node
    T.Edges.Weight(1:end)=1;
    % Calculate all-pairs shortest path distances in the tree
    dists = distances(T);
    % Find the eccentricity of each node (max distance to any other node)
    eccentricities = max(dists);
    tbreaker = mean(dists);
    % Find the node(s) with the minimum eccentricity (The Center)
    minEcc = min(eccentricities);
    node_start = find(eccentricities == minEcc);
    if (length(node_start)>1)
        meanDists=tbreaker(node_start);
        [~,indmin2]=min(meanDists);
        node_start=node_start(indmin2);
    end
    
    % Extract each branch path
    paths = {};
    for i = 1:numel(nodes_end)
        target = nodes_end(i);
        if target ~= node_start
            % Find the shortest path (branch) to that end node
            paths{end+1} = shortestpath(T, node_start, target);
        end
    end
    nPaths=length(paths);

    % Sort the connections by walking along the path
    ShortPathInds=[];
    MasterConnections=[];
    for iPath=1:nPaths
        FilsInPath = paths{iPath};
        for j=2:length(FilsInPath)
            jFib = FilsInPath(j);
            Upstream = FilsInPath(j-1);
            ConnInd = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            if (length(ConnInd)>1 && sum(Connections(ConnInd,5))>0)
                % Find the one that has the branch and use that one
                ConnInd = ConnInd(Connections(ConnInd,5)==0);
            else
                ConnInd = ConnInd(1);
            end
            ShortPathInds=[ShortPathInds; ConnInd];
            ConnRow = Connections(ConnInd,:); 
            % Sort so that the upstream fiber comes first
            if (ConnRow(3)==Upstream)
                ConnRow = [ConnRow(3:4) ConnRow(1:2) ConnRow(5)];
            end
            MasterConnections =[MasterConnections; ConnRow];
        end
    end
    % In the slave connections list, the first one is the master and the
    % second one is the slave. This just makes sure that every filament has
    % roughly the same number of slaves by reordering if necessary
    SlaveConnections = Connections;
    SlaveConnections(ShortPathInds,:)=[];
    nSlave = zeros(nFib,1);
    for p =1:size(SlaveConnections,1)
        SlaveRow = SlaveConnections(p,:);
        if (nSlave(SlaveRow(1))<nSlave(SlaveRow(3)))
            SlaveRow = [SlaveRow(3:4) SlaveRow(1:2) SlaveRow(5)];
        end
        nSlave(SlaveRow(3))=nSlave(SlaveRow(3))+1;
        SlaveConnections(p,:)=SlaveRow;
    end
    
    % Default Chebyshev points for position and tangents 
    % (will be overwritten)
    [s,~,~]=chebpts(N,[0 L],1);
    [sX,~,bX]=chebpts(Nx,[0 L],2);
    nTauReplaced = zeros(nFib,1);
    nXReplaced = zeros(nFib,1);
    TangentVectorNodes = cell(nFib,1);
    ConstrainedPosNodes = cell(nFib,1);
    for iFib=1:nFib
        TangentVectorNodes{iFib}=s;
        ConstrainedPosNodes{iFib}=sX;
    end
    % Go through the connections and start replacing the default nodes
    for iC=1:size(MasterConnections,1)
        iFib = MasterConnections(iC,1);
        iS = MasterConnections(iC,2);
        jFib = MasterConnections(iC,3);
        jS = MasterConnections(iC,4);
        % Replace the positional nodes
        [ConstrainedPosNodes,nXReplaced] = ReplaceNode(ConstrainedPosNodes,nXReplaced,iFib,iS);
        [ConstrainedPosNodes,nXReplaced] = ReplaceNode(ConstrainedPosNodes,nXReplaced,jFib,jS);
        MasterConnections(iC,2)=Nx+1-nXReplaced(iFib);
        MasterConnections(iC,4)=Nx+1-nXReplaced(jFib);
        % If a branch, replace the tangent vectors too
        if (MasterConnections(iC,5)==0)
            [TangentVectorNodes,nTauReplaced] = ReplaceNode(TangentVectorNodes,nTauReplaced,iFib,iS);
            [TangentVectorNodes,nTauReplaced] = ReplaceNode(TangentVectorNodes,nTauReplaced,jFib,jS);
        end
    end
    % Slave connections: the positional nodes get replaced, and the tangent
    % vectors get removed on the slave fiber only
    for iC=1:size(SlaveConnections,1)
        iFib = SlaveConnections(iC,1);
        iS = SlaveConnections(iC,2);
        jFib = SlaveConnections(iC,3);
        jS = SlaveConnections(iC,4);
        % Replace the positional nodes
        [ConstrainedPosNodes,nXReplaced] = ReplaceNode(ConstrainedPosNodes,nXReplaced,iFib,iS);
        [ConstrainedPosNodes,nXReplaced] = ReplaceNode(ConstrainedPosNodes,nXReplaced,jFib,jS);
        SlaveConnections(iC,2)=Nx+1-nXReplaced(iFib);
        SlaveConnections(iC,4)=Nx+1-nXReplaced(jFib);
        % For the slave filament only - remove the tangent vector
        TangentVectorNodes = RemoveNode(TangentVectorNodes,nTauReplaced,jFib,jS);
    end
    
    % Integration matrix: takes the tangent vector nodes and maps them to
    % the LEAD nodes
    % Diff matrix: takes the LEAD nodes and maps them to the tangent vector
    % nodes
    IntegrationMatrix = cell(nFib,1);
    DiffMatrix = cell(nFib,1);
    RegGridMatrix = cell(nFib,1);
    TauStart = ones(nFib+1,1);
    for iFib=1:nFib
        sTau = TangentVectorNodes{iFib};
        NTau = length(sTau);
        NLead = Nx - (N-NTau);
        LeadIndices = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==iFib,4));
        sLead = ConstrainedPosNodes{iFib}(LeadIndices);
        [sRegTau,~,bRegTau]=chebpts(NTau,[0 L],1);
        [sRegX,~,bRegX]=chebpts(NLead,[0 L],2);
        DX = diffmat(NLead,[0 L],'chebkind2');
        IntegrationMatrix{iFib}=barymat(sLead,sRegX,bRegX)*pinv(DX)*barymat(sRegX,sRegTau,bRegTau)...
            *barymat(TangentVectorNodes{iFib},sRegTau,bRegTau)^(-1);
        DiffMatrix{iFib}=barymat(sTau,sRegX,bRegX)*DX*barymat(sLead,sRegX,bRegX)^(-1);
        TauStart(iFib+1)=TauStart(iFib)+NTau;
        RegGridMatrix{iFib}=barymat(ConstrainedPosNodes{iFib},sX,bX)^(-1);
    end

    % Initialize filament DOFs (straight filaments)
    DOFs = zeros(TauStart(end)+NLinks,3);
    taus = zeros(nFib,3);
    Xstart = zeros(nFib,3);
    prevsn=-1;
    LinkNum=0;
    taus(1,:)=[0 1 0];
    DOFs(1:length(TangentVectorNodes{1}),:)=repmat(taus(1,:),length(TangentVectorNodes{1}),1);
    % Go through the list of master connections
    MasterConnections(:,6)=0;
    MasterConnections(MasterConnections(:,5)==1,6)=ell;
    for iC=1:size(MasterConnections,1)
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConstrainedPosNodes{UpFib}(ConnRow(2));
        DownFib = ConnRow(3);
        PtOnDown = ConstrainedPosNodes{DownFib}(ConnRow(4));
        % Figure out if it's a branch or cross link
        if (ConnRow(5)==0) % branch
            prevsn=-1*prevsn;
            taus(DownFib,:)=rotate(taus(UpFib,:),prevsn*70/180*pi*[0 0 1]);
            Xstart(DownFib,:)=Xstart(UpFib,:)+taus(UpFib,:)*PtOnUp;
        else % Cross link
            LinkNum=LinkNum+1;
            nRand = [randn(2,2) zeros(2,1)];
            taus(DownFib,:)=nRand(1,:)/norm(nRand(1,:));
            LinkVec =nRand(2,:)/norm(nRand(2,:));
            DOFs(TauStart(end)-1+LinkNum,:)=LinkVec;
            Xstart(DownFib,:)=Xstart(UpFib,:)+taus(UpFib,:)*PtOnUp + ...
                LinkVec*ell - taus(DownFib,:)*PtOnDown;
        end
        DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:)= ...
            repmat(taus(DownFib,:),length(TangentVectorNodes{DownFib}),1);
        % % Pairwise integration matrices (for the prconditioner)
        % DOFsToCustomNodes = [IntegrationMatrix{Upstream} zeros(Nx,N); ...
        %     ones(Nx,1).*barymat(MotherPt,sX,bX)*IntegrationMatrix{Upstream} ...
        %     (eye(Nx)-repmat(barymat(FixPt,sX,bX),Nx,1))*IntegrationMatrix{jFib}];
        % if (ConnRow(5)>0)
        %     DOFsToCustomNodes(Nx+1:end,end+1) = ell;
        % else
        %     masterpt = find(TangentVectorNodes{Upstream}==MotherPt);
        %     slavept = N+find(TangentVectorNodes{jFib}==FixPt);
        %     AssignMat = eye(N*2+1);
        %     for iBr=1:size(NodesByBranch,1)
        %         AssignMat(slavept,:)=0;
        %         AssignMat(slavept,masterpt)=1;
        %     end
        %     AssignMat(:,slavept)=[];
        %     AssignMat=stackMatrix(AssignMat);
        % end
        % % Only involves the first link
        % AvgMat = 1/(2*L)*repmat(wX,1,2);
        % SubAvg = eye(Nx*2)-repmat(ones(Nx,1),2,1).*AvgMat;
        % ChebMatZeroMean = SubAvg*DOFsToCustomNodes;
        % DOFsToChebNodes = [ChebMatZeroMean ones(2*Nx,1)];
        % PairwiseXMats{iPath,j-1,1} = stackMatrix(DOFsToChebNodes);
        % if (ConnRow(5)==0)
        %     PairwiseXMats{iPath,j-1,2} = AssignMat;
        % end
    end
    % Go through the slave nodes and set the constraints
    for iC=1:size(SlaveConnections,1)
        LinkNum=LinkNum+1;
        ConnRow = SlaveConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConstrainedPosNodes{UpFib}(ConnRow(2));
        DownFib = ConnRow(3);
        PtOnDown = ConstrainedPosNodes{DownFib}(ConnRow(4));
        tauCL = Xstart(DownFib,:)+PtOnDown*taus(DownFib,:) - ...
            (Xstart(UpFib,:)+PtOnUp*taus(UpFib,:));
        SlaveConnections(iC,6)=norm(tauCL);
        DOFs(TauStart(end)-1+LinkNum,:)=tauCL/norm(tauCL);
    end
    com=zeros(1,3);
    for iFib=1:nFib
        com=com+1/nFib*(Xstart(iFib,:)+L/2*taus(iFib,:));
        Xplt = [Xstart(iFib,:); Xstart(iFib,:)+L*taus(iFib,:)];
        plot3(Xplt(:,1),Xplt(:,2),Xplt(:,3),':')
        hold on
    end
    DOFs(end,:)=com;
end

function [ConstrainedPosNodes,nXReplaced] = ReplaceNode(ConstrainedPosNodes,nXReplaced,iFib,iS)
    iPosNodes=ConstrainedPosNodes{iFib};
    nXReplaced(iFib)=nXReplaced(iFib)+1;
    Nx=length(iPosNodes);
    [~,indmin]=min(abs(iPosNodes(1:Nx+1-nXReplaced(iFib))-iS));
    RegNodesI = iPosNodes(1:Nx+1-nXReplaced(iFib));
    RegNodesI(indmin)=[];
    iPosNodes(1:Nx-nXReplaced(iFib))=RegNodesI;
    iPosNodes(Nx+1-nXReplaced(iFib))=iS;
    ConstrainedPosNodes{iFib}=iPosNodes;
end

function ConstrainedPosNodes = RemoveNode(ConstrainedPosNodes,nReplaced,iFib,iS)
    iPosNodes=ConstrainedPosNodes{iFib};
    Nx=length(iPosNodes);
    [~,indmin]=min(abs(iPosNodes(1:Nx-nReplaced(iFib))-iS));
    iPosNodes(indmin)=[];
    ConstrainedPosNodes{iFib}=iPosNodes;
end