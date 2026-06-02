function [PCMats,AllRHSInds,AllUpClamped,Nxx] = PrecomputePairwisePC(Xinput,XInvFcn,MobFcn,...
    BranchIndices,MasterConnections,SlaveConnections,IntegrationMatrix, ...
    RegGridMatrix,BendForceMat,impcodt,L,nFib,clampedTau)
    Nxx = length(Xinput);
    Nx = Nxx/(3*nFib);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    
    X3 = reshape(Xinput,3,[])';
    DOFs = XInvFcn(X3);

    nBranch = sum(MasterConnections(:,5)==0)+1*(clampedTau>0);

    TauStart = ones(nFib,1);
    for iFib=1:nFib
        TauStart(iFib+1)=TauStart(iFib)+size(IntegrationMatrix{iFib},2);
    end
    nFreeFibTaus = TauStart(end)-nBranch;
    
    DOFIndexToTau = -ones(TauStart(end)-1,2);
    DOFIndexToTau(:,1)=1:TauStart(end)-1;
    if (~isempty(BranchIndices))
        for jBr=1:size(BranchIndices,1)
            DOFIndexToTau(BranchIndices(jBr,1),2)=BranchIndices(jBr,2);
        end
    end
    delInds=[];
    if (clampedTau>0)
        delInds=[delInds;clampedTau];
    end
    if (~isempty(BranchIndices))
        delInds = [delInds;BranchIndices(:,2)];
    end
    DOFIndexToTau(delInds,:)=[];

    LinkNum=0;
    AllUpClamped=zeros(length(MasterConnections(:,1)),1);
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
        UpClamped = TauStart(UpFib) <= clampedTau && TauStart(UpFib+1) > clampedTau;
        AllUpClamped(iC)=UpClamped;

        % Compute the matrices which map the lead nodes to a regular grid
        % of Nx Chebyshev points
        SlaveLinkInds_up=find(SlaveConnections(:,3)==UpFib);
        SlaveLinkInds_dwn=find(SlaveConnections(:,3)==DownFib);
      
        Tau3 = [DOFs(TauStart(UpFib):TauStart(UpFib+1)-1,:); ...
            DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:);...
            DOFs(TauStart(end)-1+nMasterLinks+SlaveLinkInds_up,:);...
            DOFs(TauStart(end)-1+nMasterLinks+SlaveLinkInds_dwn,:)];

        slaveDown = SlaveConnections(SlaveLinkInds_dwn,4);
        slaveUp = SlaveConnections(SlaveLinkInds_up,4);
        LeadIndDown = setdiff(1:Nx,slaveDown);
        LeadIndUp = setdiff(1:Nx,slaveUp);
        Nup = length(LeadIndUp);
        Ndown = length(LeadIndDown);
        
        DOFsToIrregNodes=[IntegrationMatrix{UpFib} zeros(Nup,2*(Nx-1)-(Nup-1)); 
            ones(Ndown,1).*IntegrationMatrix{UpFib}(ConnRow(2),:) ...
            (IntegrationMatrix{DownFib}-ones(Ndown,1).*IntegrationMatrix{DownFib}(ConnRow(4),:)) zeros(Ndown,2*(Nx-1)-(Nup-1)-(Ndown-1)); ......
            zeros(Nx-Nup,Nup-1+Ndown-1) diag(SlaveConnections(SlaveLinkInds_up,6)) zeros(Nx-Nup,(Nx-1)-(Ndown-1)); ...
            zeros(Nx-Ndown,Ndown-1+Nup-1) zeros(Nx-Ndown,(Nx-1)-(Nup-1)) diag(SlaveConnections(SlaveLinkInds_dwn,6))];
        if (ConnRow(5)>0)
            % Will need to redo this
            DOFsToIrregNodes(Nup+1:Nup+Ndown,end+1) = ConnRow(6);
            AssignMat = 1;
            LinkNum=LinkNum+1;
            Tau3 = [Tau3; DOFs(TauStart(end,:)-1+LinkNum,:)];
        end
        % Add the connection if the slave link is connected to the other
        % fiber in the pair
        for iS=1:length(SlaveLinkInds_up)
            masterfib = SlaveConnections(SlaveLinkInds_up(iS),1);
            if (masterfib==DownFib)
                masterpt = SlaveConnections(SlaveLinkInds_up(iS),2);
                masterptIndex = find(LeadIndDown==masterpt);
                ElMatrix = eye(size(DOFsToIrregNodes,1));
                % Adding the Nup+masterpt row to Nup+Ndown+iS row
                ElMatrix(Nup+Ndown+iS,Nup+masterptIndex)=1;
                DOFsToIrregNodes=ElMatrix*DOFsToIrregNodes;
            end
        end
        for iS=1:length(SlaveLinkInds_dwn)
            masterfib = SlaveConnections(SlaveLinkInds_dwn(iS),1);
            if (masterfib==UpFib)
                masterpt = SlaveConnections(SlaveLinkInds_dwn(iS),2);
                masterptIndex = find(LeadIndUp==masterpt);
                ElMatrix = eye(size(DOFsToIrregNodes,1));
                % Adding the masterpt row to Nx+Ndown+iS row
                ElMatrix(Nx+Ndown+iS,masterptIndex)=1;
                DOFsToIrregNodes=ElMatrix*DOFsToIrregNodes;
            end
        end

        % Remove the slave DOF from the second fiber
        DOFsDownNoSlave= DOFsDown;
        if (ConnRow(5)==0)
            masterpt = BranchIndices(BrIndex,1)-TauStart(UpFib)+1;
            slavept = BranchIndices(BrIndex,2)-TauStart(DownFib)+1;
            slaveindex=length(DOFsUp)+slavept;
            AssignMat = eye(2*Nx-1);
            AssignMat(slaveindex,:)=0;
            AssignMat(slaveindex,masterpt)=1;
            % Adjust if up fiber is clalmped
            if (UpClamped)
                clampIndex = clampedTau-TauStart(UpFib)+1;
                AssignMat(clampIndex,:)=0;
                AssignMat(:,end)=[];
                AssignMat(end,:)=[];
            else
                clampIndex=[];
            end
            AssignMat(:,[slaveindex;clampIndex])=[];
            AssignMat=stackMatrix(AssignMat);
            DOFsDownNoSlave(slavept) = [];
        elseif (UpClamped) % and a cross link
            AssignMat = eye(2*Nx-1);
            clampIndex = clampedTau-TauStart(UpFib)+1;
            AssignMat(clampIndex,:)=0;
            AssignMat(:,clampIndex)=[];
            AssignMat=stackMatrix(AssignMat);
        end
        % Compute mean and subtract
        ToChebNodes = [RegGridMatrix{UpFib}(:,LeadIndUp) zeros(Nx,Ndown) RegGridMatrix{UpFib}(:,slaveUp) zeros(Nx,Nx-Ndown); ...
            zeros(Nx,Nup) RegGridMatrix{DownFib}(:,LeadIndDown) zeros(Nx,Nx-Nup) RegGridMatrix{DownFib}(:,slaveDown)];
        DOFsToChebNodes=ToChebNodes*DOFsToIrregNodes;
        if (UpClamped)
            XMat = stackMatrix(DOFsToChebNodes);
        else
            DOFsToChebNodes(end+1,end+1)=1;
            AvgMat = 1/(2*L)*[wX wX].*ones(2*Nx,1);
            SubAddAvg = [eye(2*Nx)-AvgMat ones(2*Nx,1)];
            XMat = stackMatrix(SubAddAvg*DOFsToChebNodes);   
        end

        % K matrix
        TauVelocity = zeros(3*size(Tau3,1)+3*(~UpClamped));
        % The matrix for all the taus (incl links) to evolve
        for iR =1:size(Tau3,1)
            inds = (iR-1)*3+1:iR*3;
            CMat = CPMatrix(Tau3(iR,:));
            TauVelocity(inds,inds) =  -CMat;
        end
        % The COM
        if (~UpClamped)
            TauVelocity(end-2:end,end-2:end)=eye(3);
        end
        K = XMat*TauVelocity*AssignMat;

        DOFIndicesToFind = [DOFsUp DOFsDownNoSlave];
        DOFIndicesToFind(DOFIndicesToFind==clampedTau)=[];
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
        for iG=(nMasterLinks+SlaveLinkInds_up')
            slaveUps=[slaveUps 3*nFreeFibTaus-3+(3*iG-2:3*iG)];
        end
        slaveDwns = [];
        for iG=(nMasterLinks+SlaveLinkInds_dwn')
            slaveDwns=[slaveDwns 3*nFreeFibTaus-3+(3*iG-2:3*iG)];
        end
        stAlphInds = [stAlphInds slaveUps slaveDwns];
        % Add the taus for the CL on the RHS
        if (ConnRow(5)>0)
            stAlphInds=[stAlphInds ...
                3*nFreeFibTaus-2+(3*LinkNum-3:3*LinkNum-1)];
        end
        UInds = [3*Nx*(UpFib-1)+(1:3*Nx) 3*Nx*(DownFib-1)+(1:3*Nx)];

        % Form and solve linear system for this pair
        M = blkdiag(MobFcn(Xinput(3*Nx*(UpFib-1)+(1:3*Nx))),...
            MobFcn(Xinput(3*Nx*(DownFib-1)+(1:3*Nx))));
        KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
        NMat = pinv(K'*(M\KWithImp));

        AllRHSInds{iC,1}=UInds;
        AllRHSInds{iC,2}=stAlphInds;
        PCMats{iC,1}=M;
        PCMats{iC,2}=K;
        PCMats{iC,3}=KWithImp;
        PCMats{iC,4}=NMat;
        
    end
end