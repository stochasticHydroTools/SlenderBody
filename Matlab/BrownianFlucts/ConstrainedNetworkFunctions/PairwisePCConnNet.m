% Apply preconditioner
function x = PairwisePCConnNet(RHS,PCMats,AllstAlphInds,NodeOrderByPair,...
    MasterConnections,SlaveConnections,RegGridMatrix,RegGridMatrixInv,...
    Nx,L,nFib,TauStart,clampedTau)

    [~,wX,~]=chebpts(Nx,[0 L],2);
    Nxx = 3*Nx*nFib;
    Lam_All = zeros(Nxx,1);
    U_All = zeros(Nxx,1);
    nAlphas = length(RHS)-Nxx;
    alphaU_All = zeros(nAlphas,1);
    
    U = RHS(1:Nxx);
    V = RHS(Nxx+1:end);

    % Loop through minimal spanning tree
    for iC=1:size(MasterConnections,1)
        ConnRow = MasterConnections(iC,:);
        UpFib = ConnRow(1);
        DownFib = ConnRow(3);
        % Find the branch index if it exists
        UpClamped = TauStart(UpFib) <= clampedTau && TauStart(UpFib+1) > clampedTau;

        NodesUp = NodeOrderByPair{iC,1};
        NodesDown = NodeOrderByPair{iC,2};

        % Append any links which are slaves to these filaments
        MasterLinksPair = NodeOrderByPair{iC,3};

        stAlphInds=AllstAlphInds{iC};

        if (UpClamped)
            VThis = V(stAlphInds);
        elseif (clampedTau>0)
            VThis = [V(stAlphInds); zeros(3,1)];
        else
            VThis = [V(stAlphInds); V(end-2:end)];
        end

        % Start at the lead nodes
        ULeadUp = IrregNodeVel(UpFib,RegGridMatrixInv,U,NodesUp,Nx);
        ULeadDwn= IrregNodeVel(DownFib,RegGridMatrixInv,U,NodesDown,Nx);
        % Order mobility and RHS appropriately 
        UThis = [ULeadUp;ULeadDwn];

        % Add the slave velocities not on this pair 
        for iS=1:length(MasterLinksPair)
            SlaveFib = SlaveConnections(MasterLinksPair(iS),3);
            SlaveNode = SlaveConnections(MasterLinksPair(iS),4);
            Uex = IrregNodeVel(SlaveFib,RegGridMatrixInv,U,SlaveNode,Nx);
            UThis = [UThis;Uex];
        end

        % Form and solve linear system for this pair
        Minv = PCMats{iC,1};
        K = PCMats{iC,2};
        KWithImp = PCMats{iC,3};
        NMat = PCMats{iC,4};
        GTG = PCMats{iC,5};

        KprimeMinvU = K' *GTG* (Minv*UThis);
        alphaU = NMat*(KprimeMinvU+VThis);
        Lam = Minv*(KWithImp*alphaU - UThis);

        % Assign lambda carefully
        Lam_All = AssignToChebNodes(Lam_All,Lam,UpFib,DownFib,Nx,...
            NodesUp,NodesDown,RegGridMatrix,MasterLinksPair,SlaveConnections);
        U_All = AssignToChebNodes(U_All,K*alphaU,UpFib,DownFib,Nx,...
            NodesUp,NodesDown, RegGridMatrix,MasterLinksPair,SlaveConnections);

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
    x=[Lam_All;alphaU_All];
end

function ULeadUp = IrregNodeVel(UpFib,RegGridMatrixInv,U,inds,Nx)
    UpInds = 3*Nx*(UpFib-1)+(1:3*Nx);
    Uup = reshape(U(UpInds),3,[])';
    ULeadUp = reshape((RegGridMatrixInv{UpFib}(inds,:)*Uup)',[],1);
end

function Lam_All = AssignToChebNodes(Lam_All,Lam,UpFib,DownFib,Nx,LeadIndUp,LeadIndDown,...
            RegGridMatrix,MLinkInds,SlaveConnections)
    Nup = length(LeadIndUp);
    Ndown = length(LeadIndDown);
    LamUp = RegGridMatrix{UpFib}(:,LeadIndUp)*reshape(Lam(1:3*Nup),3,[])';
    Lam_All(3*Nx*(UpFib-1)+(1:3*Nx))=Lam_All(3*Nx*(UpFib-1)+(1:3*Nx))+reshape(LamUp',[],1);
    LamDown = RegGridMatrix{DownFib}(:,LeadIndDown)*reshape(Lam(3*Nup+1:3*(Nup+Ndown)),3,[])';
    Lam_All(3*Nx*(DownFib-1)+(1:3*Nx))=Lam_All(3*Nx*(DownFib-1)+(1:3*Nx))+reshape(LamDown',[],1);
    for iS=1:length(MLinkInds)
        SlaveFib = SlaveConnections(MLinkInds(iS),3);
        SlaveNode = SlaveConnections(MLinkInds(iS),4);
        LamSlave = RegGridMatrix{SlaveFib}(:,SlaveNode)*reshape(Lam(3*(Nup+Ndown)+((3*iS-2):3*iS)),3,[])';
        Lam_All(3*Nx*(SlaveFib-1)+(1:3*Nx))=Lam_All(3*Nx*(SlaveFib-1)+(1:3*Nx))+reshape(LamSlave',[],1);
    end
end