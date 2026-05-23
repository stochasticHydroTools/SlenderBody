function PrecompPCs = PrecomputePairwisePC(RHS,Xinput,XInvFcn,MobFcn,...
    MasterConnections,SlaveConnections,IntegrationMatrix, ...
    ConstrainedPosNodes, TangentVectorNodes,BendForceMat,impcodt,L,nFib)
    Nxx = length(Xinput);
    Nx = Nxx/(3*nFib);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    
    X3 = reshape(Xinput,3,[])';
    DOFs = XInvFcn(X3);

    Lam_All = zeros(Nxx,1);
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
        % Compute the X matrix for those points
        DOFsToChebNodes = [IntMatUp zeros(Nx,length(DOFsDown)); ...
            ones(Nx,1).*barymat(PtOnUp,sX,bX)*IntMatUp ...
            (eye(Nx)-repmat(barymat(PtOnDown,sX,bX),Nx,1))*IntMatDwn];
        if (ConnRow(5)>0)
            DOFsToChebNodes(Nup+1:end,end+1) = ConnRow(6);
            AssignMat = 1;
        else
            masterpt = find(TangentVectorNodes{UpFib}==PtOnUp);
            slavept = length(DOFsUp)+find(TangentVectorNodes{DownFib}==PtOnDown);
            AssignMat = eye(length(DOFsUp)+length(DOFsDown)+1);
            AssignMat(slavept,:)=0;
            AssignMat(slavept,masterpt)=1;
            AssignMat(:,slavept)=[];
            AssignMat=stackMatrix(AssignMat);
        end
        % The matrix XMat gives you the positions of the pair
        % on the Nx grids, from the tangent vectors and center of mass
        AvgMat = 1/(2*L)*[wX wX];
        SubAvg = eye(2*Nx)-ones(2*Nx,1).*AvgMat;
        ChebMatZeroMean = SubAvg*DOFsToChebNodes;
        DOFsToChebNodes = [ChebMatZeroMean ones(2*Nx,1)];
        XMat = stackMatrix(DOFsToChebNodes);

        % Form K for those links
        upInds = 3*Nx*(UpFib-1)+(1:3*Nx);
        downInds = 3*Nx*(DownFib-1)+(1:3*Nx);
        UInds = [upInds'; downInds'];
        UThis = U(UInds);
        stAlphInds = [3*TauStartNoSlave(UpFib)-2:3*(TauStartNoSlave(UpFib+1)-1) ...
            3*TauStartNoSlave(DownFib)-2:3*(TauStartNoSlave(DownFib+1)-1)];
        % Remove slave taus
        Tau3 = [DOFs(TauStart(UpFib):TauStart(UpFib+1)-1,:); ...
            DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:)];
        MasterRow = find(MasterConnections(:,3)==UpFib & MasterConnections(:,5)==0);
        if (~isempty(MasterRow))
            % Find the real tangent vector node that is the master of this 
            % slave one
            ParentFib = MasterConnections(MasterRow,1);
            ParentPt = ConstrainedPosNodes{ParentFib}(MasterConnections(MasterRow,2));
            ParentTau = find(TangentVectorNodes{ParentFib}==ParentPt);
            stAlphInds = [3*TauStartNoSlave(UpFib)-2:3*(TauStartNoSlave(UpFib+1)-1) ...
            3*TauStartNoSlave(ParentFib)+3*ParentTau+(-5:-3) 3*TauStartNoSlave(DownFib)-2:3*(TauStartNoSlave(DownFib+1)-1)];
        end
        if (ConnRow(5)>0)
            LinkNum=LinkNum+1;
            Tau3 = [Tau3; DOFs(TauStart(end,:)-1+LinkNum,:)];
            stAlphInds=[stAlphInds 3*TauStartNoSlave(end)-2+(3*LinkNum-3:3*LinkNum-1)];
        end
        VThis = [V(stAlphInds); V(end-2:end)];
        K = KWithLink(XMat,Tau3,AssignMat);
        M = blkdiag(MobFcn(Xinput(3*Nx*(UpFib-1)+(1:3*Nx))),...
            MobFcn(Xinput(3*Nx*(DownFib-1)+(1:3*Nx))));
        KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
        NMat = pinv(K'*(M\KWithImp));
        KprimeMinvU = K' * (M \ UThis);
        alphaU = NMat*(KprimeMinvU+VThis);
        Lam = M \ (KWithImp*alphaU - UThis);

        Lam_All(UInds)=Lam_All(UInds)+Lam;
        alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
        alphaU_All(end-2:end)=alphaU_All(end-2:end)+alphaU(end-2:end);
    end

    % Slave connections
    for iC=1:size(SlaveConnections,1)
        LinkNum=LinkNum+1;
        Tau3 = DOFs(TauStart(end,:)-1+LinkNum,:);
        stAlphInds=[3*TauStartNoSlave(end)-2+(3*LinkNum-2:3*LinkNum)];
        alphaU_All(stAlphInds)=V(stAlphInds);
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