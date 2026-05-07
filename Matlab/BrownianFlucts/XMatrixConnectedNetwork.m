% Compute the matrix X assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 

% Split this into two functions
% function [G,DOFs,TangentVectorNodes,IntegrationMatrix] =
% InitializeConnectedNetwork(Connections,nFib,N,L,ell)
% function [X,XMat]=XMatrixConnectedNetwork(G,DOFs,IntegrationMatrix)
nFib = 10;
N = 15;
Nx = N+1;
L = 1;
ell = 0.1;
Connections = [1 0.5 2 0 0; 2 0.5 3 0 0; 3 0.5 4 0 0; 4 0.5 5 0 0; ...
    1 0.9 6 0 0; 6 0.5 7 0 0; 7 0.9 8 0.1 1; 8 0.5 9 0 0; 9 0.5 10 0.1 1];
% Check list
for p=1:size(Connections,1)
    if (Connections(p,4)==0 && Connections(p,5)~=0)
        warning('Cannot have a branch away from s=0 on daughter, at connection number %d',p)
        Connections(p,5)=0;
    end
end
NLinks = nnz(Connections(:,5));
DOFs = zeros(nFib*N+NLinks+1,3);
X = zeros(Nx*nFib,3);

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
[s,w,b]=chebpts(N,[0 L],1);
[sX,wX,bX]=chebpts(Nx,[0 L],2);
DX = diffmat(Nx,[0 L],'chebkind2');
TangentVectorNodes = cell(nFib,1);
IntegrationMatrix = cell(nFib,1);
for iFib=1:nFib
    tNodes=s;
    % Find all the connections involving iFib
    Branches = Connections(Connections(:,5)==0,:);
    BranchPts = [Branches(Branches(:,1)==iFib,2); Branches(Branches(:,3)==iFib,4)];
    for iPt=1:length(BranchPts)
        [~,indmin]=min(abs(tNodes(1:N+1-iPt)-BranchPts(iPt)));
        tNodes(indmin)=[];
        tNodes=[tNodes;BranchPts(iPt)];
    end
    TangentVectorNodes{iFib}=tNodes;
    ChebToConstr_Br = barymat(tNodes,s,b);
    IntegrationMatrix{iFib}=pinv(DX)*barymat(sX,s,b)*barymat(tNodes,s,b)^(-1);
end
% 
% Chebyshev points of first filament
taus = zeros(nFib,3);
taus(1,:)=[0 1 0];
DOFs(1:N,:)=repmat(taus(1,:),N,1);
X(1:Nx,:) = IntegrationMatrix{1}*DOFs(1:N,:);
XMat = eye(nFib*N+NLinks+1 + Nx*nFib); % The positions in the top and DOFs in the bottom
XMat(1:Nx,nFib*Nx+(1:N))=IntegrationMatrix{1};
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
        % Figure out if it's a branch or cross link
        if (ConnRow(5)==0) % branch
            taus(jFib,:)=RotateSeventy(taus(Upstream,:));
        else % Cross link
            LinkNum=LinkNum+1;
            nRand = randn(2,3);
            taus(jFib,:)=nRand(1,:)/norm(nRand(1,:));
            LinkVec =nRand(2,:)/norm(nRand(2,:));
            DOFs(N*nFib+LinkNum,:)=LinkVec;
        end
        DOFs((jFib-1)*N+(1:N),:)= repmat(taus(jFib,:),N,1);
        X((jFib-1)*Nx+(1:Nx),:) = IntegrationMatrix{jFib}*DOFs((jFib-1)*N+(1:N),:);
        XFactor = eye(nFib*N+NLinks+1 + Nx*nFib);
        XFactor((jFib-1)*Nx+(1:Nx),Nx*nFib+(jFib-1)*N+(1:N))=IntegrationMatrix{jFib};
        XMat = XFactor*XMat;
        X((jFib-1)*Nx+(1:Nx),:) = X((jFib-1)*Nx+(1:Nx),:)-barymat(FixPt,sX,bX)*X((jFib-1)*Nx+(1:Nx),:)+ ...
            barymat(MotherPt,sX,bX)*X((Upstream-1)*Nx+(1:Nx),:);
        XFactor = eye(nFib*N+NLinks+1 + Nx*nFib);
        XFactor((jFib-1)*Nx+(1:Nx),(jFib-1)*Nx+(1:Nx))=eye(Nx)-repmat(barymat(FixPt,sX,bX),Nx,1);
        XFactor((jFib-1)*Nx+(1:Nx),(Upstream-1)*Nx+(1:Nx)) = repmat(barymat(MotherPt,sX,bX),Nx,1);
        if (ConnRow(5)>0)
            X((jFib-1)*Nx+(1:Nx),:)=X((jFib-1)*Nx+(1:Nx),:)+LinkVec*ell;
            XFactor((jFib-1)*Nx+(1:Nx),Nx*nFib+N*nFib+LinkNum)=ell;
        end
        XMat = XFactor*XMat;
    end
end
XMat = XMat(1:Nx*nFib,Nx*nFib+1:end);
AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
XMat = SubAvg*XMat;
XMat(:,end)=1;

% Remove the average
AvgPt = zeros(1,3);
for iFib=1:nFib
    AvgPt = AvgPt+1/(L*nFib)*wX*X((iFib-1)*Nx+(1:Nx),:);
end
X = X-AvgPt + DOFs(end,:);

X2 = XMat*DOFs;
max(abs(X-X2))

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