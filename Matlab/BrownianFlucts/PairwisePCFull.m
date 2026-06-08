% Go along master tree only
function PrecompPCs = PairwisePCFull(RHS,Xinput,XInvFcn,MobFcn,...
    BranchIndices,MasterConnections,SlaveConnections,IntegrationMatrix, ...
    RegGridMatrix,ConstrainedPosNodes,BendForceMat,impcodt,L,nFib,clampedTau)
    Nxx = length(Xinput);
    Nx = Nxx/(3*nFib);
    [sX,wX,bX]=chebpts(Nx,[0 L],2);
    
    X3 = reshape(Xinput,3,[])';
    DOFs = XInvFcn(X3);

    Lam_All = zeros(Nxx,1);
    U_All = zeros(Nxx,1);
    nBranch = sum(MasterConnections(:,5)==0)+1*(clampedTau>0);
    nAlphas = 3*(size(DOFs,1)-nBranch);
    alphaU_All = zeros(nAlphas,1);
    
    U = RHS(1:Nxx);
    V = RHS(Nxx+1:end);

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
    nMasterLinks = sum(MasterConnections(:,5)==1);
    % Loop through minimal spanning tree
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
        
        % Define master and slave nodes on each filament
        % Find all slave nodes on the filaments
        SlaveUp = find(SlaveConnections(:,3)==UpFib,3);
        SlavesOnUp = SlaveConnections(SlaveUp,4)';
        LeadIndUp = setdiff(1:Nx,SlavesOnUp);
        SlaveDown = find(SlaveConnections(:,3)==DownFib,3);
        SlavesOnDown = SlaveConnections(SlaveDown,4)';
        LeadIndDown = setdiff(1:Nx,SlavesOnDown);
        Nup = length(LeadIndUp);
        Ndown = length(LeadIndDown);

        % Append any links which are slaves to these filaments
        MasterLinksPair = find(SlaveConnections(:,1)==UpFib | ...
            SlaveConnections(:,1)==DownFib);
        MasterLinksPair = setdiff(MasterLinksPair,[SlaveUp;SlaveDown]);
        AllSlaveLinks = [SlaveUp;SlaveDown;MasterLinksPair];
        nSecLinks = length(AllSlaveLinks);
        MasterNodes = SlaveConnections(AllSlaveLinks,2);
        MasterFibs = SlaveConnections(AllSlaveLinks,1);
        
        % Define DOFs (tangent vectors on both fibs and any additional
        % links that connect them to slave nodes)
        Tau3 = [DOFs(TauStart(UpFib):TauStart(UpFib+1)-1,:); ...
            DOFs(TauStart(DownFib):TauStart(DownFib+1)-1,:);...
            DOFs(TauStart(end)-1+nMasterLinks+AllSlaveLinks,:)];

        % This matrix maps the tangent vectors to the lead nodes, and
        % scales the slave link vectors 
        % Order of nodes
        % [Lead up; Lead down; Slaves ON up; Slaves ON down; 
        % Slaves OF up Slaves OF down]
        % (Slaves could be on other fibers)
        DOFsToMasterSlaveNodes=[IntegrationMatrix{UpFib} zeros(Nup,Ndown-1+nSecLinks); 
            ones(Ndown,1).*IntegrationMatrix{UpFib}(ConnRow(2),:) ...
            (IntegrationMatrix{DownFib}-ones(Ndown,1).*IntegrationMatrix{DownFib}(ConnRow(4),:)) ...
            zeros(Ndown,nSecLinks); ...
            zeros(nSecLinks,Nup+Ndown-2) diag(SlaveConnections(AllSlaveLinks,6))];
        if (ConnRow(5)>0)
            DOFsToMasterSlaveNodes(Nup+1:Nup+Ndown,end+1) = ConnRow(6);
            AssignMat = 1;
            LinkNum=LinkNum+1;
            Tau3 = [Tau3; DOFs(TauStart(end,:)-1+LinkNum,:)];
        end
        if (~UpClamped)
            DOFsToMasterSlaveNodes(1:Nup+Ndown,end+1)=1; % a constant
        end

        % If the slave connections are connected to this pair of filaments,
        % update the matrix accordingly. If not, do nothing
        for iS=1:length(AllSlaveLinks)
            ElMatrix = eye(size(DOFsToMasterSlaveNodes,1));
            if (MasterFibs(iS)==UpFib)
                % Slave on another filament; master on up filament
                depnode = find(LeadIndUp==MasterNodes(iS));
                ElMatrix(Nup+Ndown+iS,depnode)=1;
            elseif (MasterFibs(iS)==DownFib)
                % Slave on another filament; master on down filament
                depnode = Nup+find(LeadIndDown==MasterNodes(iS));
                ElMatrix(Nup+Ndown+iS,depnode)=1;
            end
            DOFsToMasterSlaveNodes=ElMatrix*DOFsToMasterSlaveNodes;
        end


        % For the purposes of computing mobility, include in the "lead
        % nodes" those which are slave to the other fiber
        % Permute the X matrix
        XMat = [DOFsToMasterSlaveNodes(1:Nup,:); ...
            DOFsToMasterSlaveNodes(Nup+Ndown+1:Ndown+Nx,:); ...
            DOFsToMasterSlaveNodes(Nup+1:Nup+Ndown,:); ...
            DOFsToMasterSlaveNodes(Ndown+Nx+1:2*Nx,:); ...
            DOFsToMasterSlaveNodes(2*Nx+1:end,:)];
        XMat=stackMatrix(XMat);

        % Remove the slave DOF from the second fiber
        DOFsDownNoSlave= DOFsDown;
        if (ConnRow(5)==0)
            masterpt = BranchIndices(BrIndex,1)-TauStart(UpFib)+1;
            slavept = BranchIndices(BrIndex,2)-TauStart(DownFib)+1;
            slaveindex=length(DOFsUp)+slavept;
            AssignMat = eye(size(Tau3,1)+1);
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
            AssignMat = eye(size(Tau3,1));
            clampIndex = clampedTau-TauStart(UpFib)+1;
            AssignMat(clampIndex,:)=0;
            AssignMat(:,clampIndex)=[];
            AssignMat=stackMatrix(AssignMat);
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
        for iG=(nMasterLinks+AllSlaveLinks')
            stAlphInds=[stAlphInds 3*nFreeFibTaus-3+(3*iG-2:3*iG)];
        end
        % Add the taus for the CL on the RHS
        if (ConnRow(5)>0)
            stAlphInds=[stAlphInds ...
                3*nFreeFibTaus-2+(3*LinkNum-3:3*LinkNum-1)];
        end
        if (UpClamped)
            VThis = V(stAlphInds);
        elseif (clampedTau>0)
            VThis = [V(stAlphInds); zeros(3,1)];
        else
            VThis = [V(stAlphInds); V(end-2:end)];
        end

        % Start at the lead nodes
        [ULeadUp,Mup,MFUp,GTGUp] = IrregNodeMobility(UpFib,RegGridMatrix,...
            U,Xinput,[LeadIndUp SlavesOnUp],Nx,MobFcn,BendForceMat);
        [ULeadDwn,Mdwn,MFdown,GTGDwn] = IrregNodeMobility(DownFib,RegGridMatrix,...
            U,Xinput,[LeadIndDown SlavesOnDown],Nx,MobFcn,BendForceMat);
        % Order mobility and RHS appropriately 
        UThis = [ULeadUp;ULeadDwn];
        M = blkdiag(Mup,Mdwn);
        MF = blkdiag(MFUp,MFdown);
        GTG = blkdiag(GTGUp,GTGDwn);

        % Add the slave velocities not on this pair 
        for iS=1:length(MasterLinksPair)
            SlaveFib = SlaveConnections(MasterLinksPair(iS),3);
            SlaveNode = SlaveConnections(MasterLinksPair(iS),4);
            [Uex,Mex,MFex,GTGex] = IrregNodeMobility(SlaveFib,RegGridMatrix,U,Xinput,...
                SlaveNode,Nx,MobFcn,BendForceMat);
            UThis = [UThis;Uex];
            M = blkdiag(M,Mex);
            MF = blkdiag(MF,MFex);
            GTG = blkdiag(GTG,GTGex);
        end

        % Form and solve linear system for this pair
        KWithImp = K-impcodt*MF*K;
        NMat = pinv(K'*GTG*(M\KWithImp));
        KprimeMinvU = K' *GTG* (M \ UThis);
        alphaU = NMat*(KprimeMinvU+VThis);
        Lam = M \ (KWithImp*alphaU - UThis);

        % Assign lambda carefully
        Lam_All = AssignToChebNodes(Lam_All,Lam,UpFib,DownFib,Nx,...
            [LeadIndUp SlavesOnUp],[LeadIndDown SlavesOnDown],...
            RegGridMatrix,MasterLinksPair,SlaveConnections);
        U_All = AssignToChebNodes(U_All,K*alphaU,UpFib,DownFib,Nx,...
            [LeadIndUp SlavesOnUp],[LeadIndDown SlavesOnDown],...
            RegGridMatrix,MasterLinksPair,SlaveConnections);

        if (UpClamped)
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU;
        elseif (clampedTau>0)
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
        else
            alphaU_All(stAlphInds)=alphaU_All(stAlphInds)+alphaU(1:end-3);
        end
    end
    % Compute average velocity
    if (clampedTau<=0)
        MeanU = zeros(1,3);
        for iFib=1:nFib
            MeanU = MeanU+1/(nFib*L)*wX*reshape(U_All(3*Nx*(iFib-1)+1:3*Nx*iFib),3,[])';
        end
        alphaU_All(end-2:end)=MeanU;
    end
    PrecompPCs=[Lam_All;alphaU_All];
end

function [ULeadUp,Mup,MFUp,GTG] = IrregNodeMobility(UpFib,RegGridMatrix,U,Xinput,...
    inds,Nx,MobFcn,BendForceMat)
    UpInds = 3*Nx*(UpFib-1)+(1:3*Nx);
    Uup = U(UpInds);
    RegGridUpInv = stackMatrix(RegGridMatrix{UpFib}^(-1));
    Inds3D = [3*inds-2; 3*inds-1; 3*inds];
    Inds3D = Inds3D(:);
    ULeadUp = RegGridUpInv(Inds3D,:)*Uup;
    Mup=MobFcn(Xinput(UpInds));
    G = stackMatrix(RegGridMatrix{UpFib}(:,inds));
    MFUp = RegGridUpInv(Inds3D,:)*Mup*BendForceMat*G;
    Mup = RegGridUpInv(Inds3D,:)*Mup*G;
    GTG=G'*G;
end

function Lam_All = AssignToChebNodes(Lam_All,Lam,UpFib,DownFib,Nx,LeadIndUp,LeadIndDown,...
            RegGridMatrix,MLinkInds,SlaveConnections)
    Nup = length(LeadIndUp);
    Ndown = length(LeadIndDown);
    LamUp = stackMatrix(RegGridMatrix{UpFib}(:,LeadIndUp))*Lam(1:3*Nup);
    Lam_All(3*Nx*(UpFib-1)+(1:3*Nx))=Lam_All(3*Nx*(UpFib-1)+(1:3*Nx))+LamUp;
    LamDown = stackMatrix(RegGridMatrix{DownFib}(:,LeadIndDown))*Lam(3*Nup+1:3*(Nup+Ndown));
    Lam_All(3*Nx*(DownFib-1)+(1:3*Nx))=Lam_All(3*Nx*(DownFib-1)+(1:3*Nx))+LamDown;
    for iS=1:length(MLinkInds)
        SlaveFib = SlaveConnections(MLinkInds(iS),3);
        SlaveNode = SlaveConnections(MLinkInds(iS),4);
        LamSlave = stackMatrix(RegGridMatrix{SlaveFib}(:,SlaveNode))*Lam(3*(Nup+Ndown)+((3*iS-2):3*iS));
        Lam_All(3*Nx*(SlaveFib-1)+(1:3*Nx))=Lam_All(3*Nx*(SlaveFib-1)+(1:3*Nx))+LamSlave;
    end
end