function PrecompPCs = PrecomputePairwisePC(RHS,Xinput,XInvFcn,MobFcn,...
    BranchIndices,MasterConnections,SlaveConnections,IntegrationMatrix, ...
    RegGridMatrix,ConstrainedPosNodes,BendForceMat,impcodt,L,nFib)
    Nxx = length(Xinput);
    Nx = Nxx/(3*nFib);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    
    X3 = reshape(Xinput,3,[])';
    DOFs = XInvFcn(X3);

    Lam_All = zeros(Nxx,1);
    U_All = zeros(Nxx,1);
    nAlphas = 3*(size(DOFs,1)-sum(MasterConnections(:,5)==0));
    alphaU_All = zeros(nAlphas,1);
    
    U = RHS(1:Nxx);
    V = RHS(Nxx+1:end);

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
    % First solve pretending that the master connections are the only ones
    for iC=1:size(MasterConnections,1)
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        PtOnUp = ConstrainedPosNodes{UpFib}(ConnRow(2));
        DownFib = ConnRow(3);
        PtOnDown = ConstrainedPosNodes{DownFib}(ConnRow(4));
        DOFsDown = TauStart(DownFib):TauStart(DownFib+1)-1;
        DOFsUp = TauStart(UpFib):TauStart(UpFib+1)-1;
        Tau3 = [DOFs(TauStart(UpFib):TauStart(UpFib+1)-1,:); ...
            DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:)];
        % Find the branch index if it exists
        BrIndex = find(BranchIndices(:,1) < TauStart(UpFib+1) & ...
            BranchIndices(:,1) >= TauStart(UpFib) & ...
            BranchIndices(:,2) < TauStart(DownFib+1) & ...
            BranchIndices(:,2) >= TauStart(DownFib));

        % Compute the matrices which map the lead nodes to a regular grid
        % of Nx Chebyshev points
        LeadIndDown = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==DownFib,4));
        LeadIndUp = setdiff(1:Nx,SlaveConnections(SlaveConnections(:,3)==UpFib,4));
        Nup = length(LeadIndUp);
        Ndown = length(LeadIndDown);
        [sUp,wUp,bUp] = chebpts(Nup,[0 L],2);
        ResampleUp = barymat(sX,sUp,bUp)*barymat(ConstrainedPosNodes{UpFib}(LeadIndUp),sUp,bUp)^(-1);
        IntMatUp = ResampleUp*IntegrationMatrix{UpFib};
        [sDwn,wDwn,bDwn] = chebpts(Ndown,[0 L],2);
        ResampleDwn = barymat(sX,sDwn,bDwn)*barymat(ConstrainedPosNodes{DownFib}(LeadIndDown),sDwn,bDwn)^(-1);
        IntMatDwn = ResampleDwn*IntegrationMatrix{DownFib};

        % Compute the X matrix for this pair (maps the lead DOFs to the
        % positions by Chebyshev integration)
        DOFsToChebNodes = [IntMatUp zeros(Nx,length(DOFsDown)); ...
            ones(Nx,1).*barymat(PtOnUp,sX,bX)*IntMatUp ...
            (eye(Nx)-repmat(barymat(PtOnDown,sX,bX),Nx,1))*IntMatDwn];

        % Remove the slave DOF from the second fiber
        DOFsDownNoSlave= DOFsDown;
        if (ConnRow(5)>0)
            DOFsToChebNodes(Nup+1:end,end+1) = ConnRow(6);
            AssignMat = 1;
            LinkNum=LinkNum+1;
            Tau3 = [Tau3; DOFs(TauStart(end,:)-1+LinkNum,:)];
        else
            masterpt = BranchIndices(BrIndex,1)-TauStart(UpFib)+1;
            slavept = BranchIndices(BrIndex,2)-TauStart(DownFib)+1;
            slaveindex=length(DOFsUp)+slavept;
            AssignMat = eye(length(DOFsUp)+length(DOFsDown)+1);
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
        
        DOFIndicesToFind = [DOFsUp DOFsDownNoSlave];
        stAlphInds = 0*repmat(DOFIndicesToFind,1,3);
        for pTau=1:length(DOFIndicesToFind)
            ThisDOFInd=find(DOFIndexToTau(:,1)==DOFIndicesToFind(pTau));
            if (isempty(ThisDOFInd))
                ThisDOFInd=find(DOFIndexToTau(:,2)==DOFIndicesToFind(pTau));
            end
            stAlphInds(3*pTau-2:3*pTau)=3*ThisDOFInd-2:3*ThisDOFInd;
        end
        % Add the taus for the CL on the RHS
        if (ConnRow(5)>0)
            stAlphInds=[stAlphInds ...
                3*TauStartNoSlave(end)-2+(3*LinkNum-3:3*LinkNum-1)];
        end
        VThis = [V(stAlphInds); V(end-2:end)];
        UInds = [3*Nx*(UpFib-1)+(1:3*Nx) 3*Nx*(DownFib-1)+(1:3*Nx)];
        UThis = U(UInds);

        % Form and solve linear system for this pair
        K = KWithLink(XMat,Tau3,AssignMat);
        M = blkdiag(MobFcn(Xinput(3*Nx*(UpFib-1)+(1:3*Nx))),...
            MobFcn(Xinput(3*Nx*(DownFib-1)+(1:3*Nx))));
        KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
        NMat = pinv(K'*(M\KWithImp));
        KprimeMinvU = K' * (M \ UThis);
        alphaU = NMat*(KprimeMinvU+VThis);
        Lam = M \ (KWithImp*alphaU - UThis);

        Lam_All(UInds)=Lam_All(UInds)+Lam;
        U_All(UInds)=U_All(UInds)+K*alphaU;
        alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
        alphaU_All(end-2:end)=alphaU_All(end-2:end)+alphaU(end-2:end);
    end


    % Slave connections
    for iC=1:size(SlaveConnections,1)
        sFib = SlaveConnections(iC,3);
        sPt = SlaveConnections(iC,4);
        mFib = SlaveConnections(iC,1);
        mPt = SlaveConnections(iC,2);
        LinkNum=LinkNum+1;
        Tau3 = DOFs(TauStart(end,:)-1+LinkNum,:);
        CMat = CPMatrix(Tau3);
        stAlphInds=3*TauStartNoSlave(end)-2+(3*LinkNum-3:3*LinkNum-1);
        mInds = 3*Nx*(mFib-1)+(1:3*Nx);
        sInds = 3*Nx*(sFib-1)+(1:3*Nx);
        Umaster0 = stackMatrix(RegGridMatrix{mFib}) \ U_All(mInds);
        Umaster0=Umaster0(3*mPt-2:3*mPt);
        Uslave0 = stackMatrix(RegGridMatrix{sFib}) \ U_All(sInds);
        Uslave0=Uslave0(3*sPt-2:3*sPt);
        AssignMat = zeros(Nx,1);
        AssignMat(sPt)=1;
        Ext = stackMatrix(RegGridMatrix{sFib}*AssignMat);
        MobSlave = MobFcn(Xinput(sInds));
        KSl = -Ext*CMat*SlaveConnections(iC,6);
        RHS = Ext*(Uslave0-Umaster0);
        % Solve for the change in U and Lambda on the slave
        LamAlph = pinv([-MobSlave KSl; KSl' zeros(3)])*[RHS; V(stAlphInds)];
        alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+LamAlph(end-2:end);
        Lam_All(sInds)=Lam_All(sInds)+LamAlph(1:end-3);
    end
    PrecompPCs=[Lam_All;alphaU_All];
end
    
        
    


function KTogether = KWithLink(XMat,Tau3,AssignMat)
    TauVelocity = zeros(3*size(Tau3,1)+3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
    end
    % The COM
    TauVelocity(end-2:end,end-2:end)=eye(3);
    KTogether = XMat*TauVelocity*AssignMat;
end