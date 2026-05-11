% Initialize assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 
function [paths,DOFs,TangentVectorNodes,IntegrationMatrix,DiffMatrix,...
    ConstrMat,NodesByBranch,PairwiseXMats] = ...
    InitializeConnectedNetwork(Connections,nFib,N,L,ell)
    Nx = N+1;
    % Check list
    for p=1:size(Connections,1)
        if (Connections(p,5)==0 && Connections(p,4)~=0)
            warning('Cannot have a branch away from s=0 on daughter, at connection number %d',p)
            Connections(p,4)=0;
        end
    end
    NLinks = nnz(Connections(:,5));
    nBr = length(Connections(:,5))-NLinks;
    DOFs = zeros(nFib*N+NLinks+1,3);
    ConstrMat = zeros(nBr,nFib*N+NLinks+1);
    
    % Make connected graph
    G = graph(Connections(:,1),Connections(:,3));
    if (G.hascycles)
        error('Directed graph cannot have cycles!')
    end
    % Identify "end" nodes (leaf nodes where the branch terminates)
    nodes_degree = degree(G);
    nodes_end = find(nodes_degree == 1);
    node_start = 1;
    
    % Extract each branch path
    paths = {};
    for i = 1:numel(nodes_end)
        target = nodes_end(i);
        if target ~= node_start
            % Find the shortest path (branch) to that end node
            paths{end+1} = shortestpath(G, node_start, target);
        end
    end
    
    % Default Chebyshev points for position and tangents (will be overwritten)
    [s,~,b]=chebpts(N,[0 L],1);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    DX = diffmat(Nx,[0 L],'chebkind2');
    nReplaced = zeros(nFib,1);
    TangentVectorNodes = cell(nFib,1);
    for iFib=1:nFib
        TangentVectorNodes{iFib}=s;
    end
    Branches = Connections(Connections(:,5)==0,:);
    NodesByBranch = [];
    for iBr=1:nBr
        iFib = Branches(iBr,1);
        iS = Branches(iBr,2);
        jFib = Branches(iBr,3);
        jS = Branches(iBr,4);
        iNodes=TangentVectorNodes{iFib};
        jNodes=TangentVectorNodes{jFib};
        nReplaced(iFib)=nReplaced(iFib)+1;
        nReplaced(jFib)=nReplaced(jFib)+1;
        [~,indmin]=min(abs(iNodes(1:N+1-nReplaced(iFib))-iS));
        tmp = iNodes(1:N+1-nReplaced(iFib));
        tmp(indmin)=[];
        iNodes(1:N-nReplaced(iFib))=tmp;
        iNodes(N+1-nReplaced(iFib))=iS;
        [~,indmin]=min(abs(jNodes(1:N+1-nReplaced(jFib))-jS));
        tmp = jNodes(1:N+1-nReplaced(jFib));
        tmp(indmin)=[];
        jNodes(1:N-nReplaced(jFib))=tmp;
        jNodes(N+1-nReplaced(jFib))=jS;
        TangentVectorNodes{iFib}=iNodes;
        TangentVectorNodes{jFib}=jNodes;
        ConstrMat(iBr,N*(iFib-1)+N+1-nReplaced(iFib))=1;
        ConstrMat(iBr,N*(jFib-1)+N+1-nReplaced(jFib))=-1;
        NodesByBranch = [NodesByBranch; N*(iFib-1)+N+1-nReplaced(iFib) ...
            N*(jFib-1)+N+1-nReplaced(jFib)];
    end
    
    IntegrationMatrix = cell(nFib,1);
    DiffMatrix = cell(nFib,1);
    for iFib=1:nFib
        IntegrationMatrix{iFib}=pinv(DX)*barymat(sX,s,b)...
            *barymat(TangentVectorNodes{iFib},s,b)^(-1);
        DiffMatrix{iFib}=barymat(TangentVectorNodes{iFib},s,b)*...
            barymat(s,sX,bX)*DX;
    end
    % 
    % Chebyshev points of first filament
    taus = zeros(nFib,3);
    taus(1,:)=[0 1 0];
    DOFs(1:N,:)=repmat(taus(1,:),N,1);
    nPaths=length(paths);
    LinkNum=0;
    prevsn=-1;
    for iPath=1:nPaths
        FilsInPath = paths{iPath};
        for j=2:length(FilsInPath)
            jFib = FilsInPath(j);
            Upstream = FilsInPath(j-1);
            % Find the corresponding row in the connection matrix
            Row = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            ConnRow = Connections(Row,:);
            % Figure out if it's a branch or cross link
            if (ConnRow(5)==0) % branch
                prevsn=-1*prevsn;
                taus(jFib,:)=rotate(taus(Upstream,:),prevsn*70/180*pi*[0 0 1]);
            else % Cross link
                LinkNum=LinkNum+1;
                nRand = [randn(2,2) zeros(2,1)];
                taus(jFib,:)=nRand(1,:)/norm(nRand(1,:));
                LinkVec =nRand(2,:)/norm(nRand(2,:));
                DOFs(N*nFib+LinkNum,:)=LinkVec;
            end
            DOFs((jFib-1)*N+(1:N),:)= repmat(taus(jFib,:),N,1);
            if (ConnRow(1)==jFib)
                FixPt = ConnRow(2);
                MotherPt = ConnRow(4);
            else
                MotherPt = ConnRow(2);
                FixPt = ConnRow(4);
            end
            % Pairwise integration matrices (for the prconditioner)
            DOFsToCustomNodes = [IntegrationMatrix{Upstream} zeros(Nx,N); ...
                ones(Nx,1).*barymat(MotherPt,sX,bX)*IntegrationMatrix{Upstream} ...
                (eye(Nx)-repmat(barymat(FixPt,sX,bX),Nx,1))*IntegrationMatrix{jFib}];
            if (ConnRow(5)>0)
                DOFsToCustomNodes(Nx+1:end,end+1) = ell;
            end
            % Only involves the first link
            AvgMat = 1/(2*L)*repmat(wX,1,2);
            SubAvg = eye(Nx*2)-repmat(ones(Nx,1),2,1).*AvgMat;
            ChebMatZeroMean = SubAvg*DOFsToCustomNodes;
            DOFsToChebNodes = [ChebMatZeroMean ones(2*Nx,1)];
            PairwiseXMats{iPath,j-1} = stackMatrix(DOFsToChebNodes);
        end
    end
end