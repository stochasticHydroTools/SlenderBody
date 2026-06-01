function [MatsByPair,DOFs,TauStart,TauStartNoSlave] = PrecomputePairwisePC(Xinput,XInvFcn,MobFcn,...
    BranchIndices,MasterConnections,SlaveConnections,IntegrationMatrix, ...
    RegGridMatrix,BendForceMat,impcodt,L,nFib)
    Nxx = length(Xinput);
    Nx = Nxx/(3*nFib);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    
    X3 = reshape(Xinput,3,[])';
    DOFs = XInvFcn(X3);  

    TauStart = ones(nFib,1);
    TauStartNoSlave = ones(nFib,1);
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(IntegrationMatrix{iFib},2);
        TauStartNoSlave(iFib+1)=TauStartNoSlave(iFib)+...
            size(IntegrationMatrix{iFib},2)-sum(MasterConnections(:,3)==iFib & MasterConnections(:,5)==0);
    end
    
    DOFIndexToTau = -ones(TauStart(end)-1,2);
    DOFIndexToTau(:,1)=1:TauStart(end)-1;
    if (~isempty(BranchIndices))
        [~,orderb]=sort(BranchIndices(:,2),'descend');
        for jBr=1:size(BranchIndices,1)
            BranchInds = BranchIndices(orderb(jBr),:);
            DOFIndexToTau(BranchInds(1),2)=BranchInds(2);
            DOFIndexToTau(BranchInds(2),:)=[];
        end
    end


    LinkNum=0;
    nMasterLinks = sum(MasterConnections(:,5)==1);
    % First solve pretending that the master connections are the only ones
    for iC=1:size(MasterConnections,1)
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        DownFib = ConnRow(3);
        DOFsDown = TauStart(DownFib):TauStart(DownFib+1)-1;
        DOFsUp = TauStart(UpFib):TauStart(UpFib+1)-1;
        % Find the branch index if it exists
        if (~isempty(BranchIndices))
        BrIndex = find(BranchIndices(:,1) < TauStart(UpFib+1) & ...
            BranchIndices(:,1) >= TauStart(UpFib) & ...
            BranchIndices(:,2) < TauStart(DownFib+1) & ...
            BranchIndices(:,2) >= TauStart(DownFib));
        else
            BrIndex=[];
        end

        % Compute the matrices which map the lead nodes to a regular grid
        % of Nx Chebyshev points
        
        SlaveIndsUp = find(SlaveConnections(:,3)==UpFib);
        SlaveLinkInds = nMasterLinks+SlaveIndsUp;
        [SlavePtsUp,sorder]=sort(SlaveConnections(SlaveIndsUp,4),'ascend');
        OrderedSlaveLinks_up=SlaveConnections(SlaveIndsUp(sorder),:);
        SlaveLinkInds_up=SlaveLinkInds(sorder);

        SlaveIndsDown = find(SlaveConnections(:,3)==DownFib);
        SlaveLinkInds = nMasterLinks+SlaveIndsDown;
        [SlavePtsDwn,sorder]=sort(SlaveConnections(SlaveIndsDown,4),'ascend');
        OrderedSlaveLinks_dwn=SlaveConnections(SlaveIndsDown(sorder),:);
        SlaveLinkInds_dwn=SlaveLinkInds(sorder);

        Tau3 = [DOFs(TauStart(UpFib):TauStart(UpFib+1)-1,:); DOFs(TauStart(end)-1+SlaveLinkInds_up,:);...
            DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:); DOFs(TauStart(end)-1+SlaveLinkInds_dwn,:)];

        Nup = Nx - length(SlavePtsUp);
        Ndown = Nx - length(SlavePtsDwn);
        % Approximation: fixed center of mass of slave links
        DOFsToIrregNodes=[IntegrationMatrix{UpFib} zeros(Nup,2*(Nx-1)-(Nup-1)); 
            ones(Ndown,1).*IntegrationMatrix{UpFib}(ConnRow(2),:) ...
            (IntegrationMatrix{DownFib}-ones(Ndown,1).*IntegrationMatrix{DownFib}(ConnRow(4),:)) zeros(Ndown,2*(Nx-1)-(Nup-1)-(Ndown-1)); ......
            zeros(Nx-Nup,Nup-1+Ndown-1) diag(OrderedSlaveLinks_up(:,6))/2 zeros(Nx-Nup,(Nx-1)-(Ndown-1)); ...
            zeros(Nx-Ndown,Ndown-1+Nup-1) zeros(Nx-Ndown,(Nx-1)-(Nup-1)) diag(OrderedSlaveLinks_dwn(:,6))/2];
        
        ToChebNodes = [RegGridMatrix{UpFib}(:,1:Nup) zeros(Nx,Ndown) RegGridMatrix{UpFib}(:,Nup+1:end) zeros(Nx,Nx-Ndown); ...
            zeros(Nx,Nup) RegGridMatrix{DownFib}(:,1:Ndown) zeros(Nx,Nx-Nup) RegGridMatrix{DownFib}(:,Ndown+1:end)];
        DOFsToChebNodes=ToChebNodes*DOFsToIrregNodes;

        % Remove the slave DOF from the second fiber
        DOFsDownNoSlave= DOFsDown;
        if (ConnRow(5)>0)
            DOFsToChebNodes(Nx+1:end,end+1) = ConnRow(6);
            AssignMat = 1;
            LinkNum=LinkNum+1;
            Tau3 = [Tau3; DOFs(TauStart(end,:)-1+LinkNum,:)];
        else
            masterpt = BranchIndices(BrIndex,1)-TauStart(UpFib)+1;
            slavept = BranchIndices(BrIndex,2)-TauStart(DownFib)+1;
            slaveindex=length(DOFsUp)+slavept;
            AssignMat = eye(2*Nx-1);
            AssignMat(slaveindex,:)=0;
            AssignMat(slaveindex,masterpt)=1;
            AssignMat(:,slaveindex)=[];
            AssignMat=stackMatrix(AssignMat);
            DOFsDownNoSlave(slavept) = [];
        end
        % The matrix XMat gives you the positions of the pair
        % on the Nx grids, from the tangent vectors and center of mass
        AvgMat = 1/(2*L)*[wX wX];
        SubAvg = eye(2*Nx)-ones(2*Nx,1).*AvgMat;
        ChebMatZeroMean = SubAvg*DOFsToChebNodes;
        DOFsToChebNodes = [ChebMatZeroMean ones(2*Nx,1)];
        XMat = stackMatrix(DOFsToChebNodes);   

        % K matrix
        TauVelocity = zeros(3*size(Tau3,1)+3);
        % The matrix for all the taus (incl links) to evolve
        for iR =1:size(Tau3,1)
            inds = (iR-1)*3+1:iR*3;
            CMat = CPMatrix(Tau3(iR,:));
            TauVelocity(inds,inds) =  -CMat;
        end
        % The COM
        TauVelocity(end-2:end,end-2:end)=eye(3);
        K = XMat*TauVelocity*AssignMat;

        DOFIndicesToFind = [DOFsUp DOFsDownNoSlave];
        stAlphInds = 0*repmat(DOFIndicesToFind,1,3);
        for pTau=1:length(DOFIndicesToFind)
            ThisDOFInd=find(DOFIndexToTau(:,1)==DOFIndicesToFind(pTau));
            if (isempty(ThisDOFInd))
                ThisDOFInd=find(DOFIndexToTau(:,2)==DOFIndicesToFind(pTau));
            end
            stAlphInds(3*pTau-2:3*pTau)=3*ThisDOFInd-2:3*ThisDOFInd;
        end
        % Add slaves
        slaveUps = [];
        for iG=SlaveLinkInds_up'
            slaveUps=[slaveUps 3*TauStartNoSlave(end)-3+(3*iG-2:3*iG)];
        end
        slaveDwns = [];
        for iG=SlaveLinkInds_dwn'
            slaveDwns=[slaveDwns 3*TauStartNoSlave(end)-3+(3*iG-2:3*iG)];
        end
        stAlphInds = [stAlphInds slaveUps slaveDwns];
        % Add the taus for the CL on the RHS
        if (ConnRow(5)>0)
            stAlphInds=[stAlphInds ...
                3*TauStartNoSlave(end)-2+(3*LinkNum-3:3*LinkNum-1)];
        end

        % Form and solve linear system for this pair
        M = blkdiag(MobFcn(Xinput(3*Nx*(UpFib-1)+(1:3*Nx))),...
            MobFcn(Xinput(3*Nx*(DownFib-1)+(1:3*Nx))));
        KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
        NMat = pinv(K'*(M\KWithImp));
        MatsByPair{iC,1}=M;
        MatsByPair{iC,3}=KWithImp;
        MatsByPair{iC,2}=K;
        MatsByPair{iC,4}=NMat;
        MatsByPair{iC,5}=stAlphInds;
    end
end