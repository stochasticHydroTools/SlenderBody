% Compute the matrix X assuming a network which is NON-CYCLIC
% Input: # fibers, list of connections between filaments (fiber1, s1, fiber2, s2,
% type). Type=0 for branch, 1 for cross link. 

% Split this into two functions
function DOFs = XInvConnectedNetwork(Connections,nFib,N,L,ell,...
    paths,X,DiffMatrix)
    
    Nx = N+1;
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    NLinks = nnz(Connections(:,5));
    DOFs = zeros(N*nFib+NLinks+1,3);

    % Compute the center
    for iFib=1:nFib
        Xthis = X((iFib-1)*Nx+(1:Nx),:);
        DOFs(end,:)=DOFs(end,:)+1/(L*nFib)*wX*Xthis;
        DOFs((iFib-1)*N+(1:N),:) = DiffMatrix{iFib}*Xthis;
    end

    % Compute the link vectors 
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
                LinkVec=(barymat(FixPt,sX,bX)*X((jFib-1)*Nx+(1:Nx),:) ...
                    - barymat(MotherPt,sX,bX)*X((Upstream-1)*Nx+(1:Nx),:))/ell;
                DOFs(N*nFib+LinkNum,:)=LinkVec;
            end
        end
    end
end