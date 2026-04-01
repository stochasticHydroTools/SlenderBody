% The assumption is that all of the fibers are connected - there is a
% unique center of mass tracking pt
nFib = 4;
Nx = 13;
L = 1;
rng(2);
% Specify the fibers connected, the points where they are connected,
% the type of connection (branch=0 or cross link=1), and the length of the link
% Branches
Branches = [1 0.7 2; 3 0.25 4];
CrossLinks = [1 0.5 3 0.1; 1 0.8 3 0.45; 2 0.77 3 1; 2 0.47 4 0.2; 3 0.75 4 0.6; 2 0.1 4 0.7];
tr=randn(1,3);
Taus = [1 0 0;RotateSeventy([1 0 0]); tr/norm(tr); RotateSeventy(tr/norm(tr))];
Nlinks = size(CrossLinks,1);
LinkHats = [0 1 0].*ones(Nlinks,1); % will be overwritten
LinkLengths = 0.1*ones(Nlinks,1);

EndClamped = zeros(1,nFib);
nSlaveNodes = zeros(1,nFib);
EndClamped(Branches(:,3))=1;
% Figure out how many DOFs are available
DOFCount = Nx*ones(1,nFib)-EndClamped;
% Loop randomly through the non-branched links and assign a master and slave
% depending on DOF count
NBranch = size(Branches,1);
if (length(unique(Branches(:,2)))~=NBranch)
    error('Cannot have a branched filament with multiple mothers!')
end
Order = randperm(Nlinks);
for iL=1:Nlinks
    iLink = Order(iL);
    iFib = CrossLinks(iLink,1);
    jFib = CrossLinks(iLink,3);
    if (DOFCount(jFib)<DOFCount(iFib))
        CrossLinks(iLink,:)=[CrossLinks(iLink,3:4) CrossLinks(iLink,1:2)];
    end
    SlaveFib = CrossLinks(iLink,3);
    DOFCount(SlaveFib)=DOFCount(SlaveFib)-1;
end

% Step 1: define the auxilary grid of positional nodes which includes the
% ones constrained by CLs
MasterNodes = cell(nFib,1);
[sX,wX,bX] = chebpts(Nx,[0 L], 2);
DX = diffmat(Nx,[0 L],'chebkind2');
for iFib=1:nFib
    MasterNodes{iFib}=sX;
end
% Remove the Chebyshev nodes closest to the links and branches
for iLink=1:Nlinks
    MasterFib = CrossLinks(iLink,1);
    MasterPt = CrossLinks(iLink,2);
    SlaveFib = CrossLinks(iLink,3);
    SlavePt = CrossLinks(iLink,4);
    [~,indmin]=min(abs(MasterNodes{MasterFib}-MasterPt));
    MasterNodes{MasterFib}(indmin)=[];
    [~,indmin]=min(abs(MasterNodes{SlaveFib}-SlavePt));
    MasterNodes{SlaveFib}(indmin)=[];
end
for iBr=1:NBranch
    Mother = Branches(iBr,1);
    MotherPt = Branches(iBr,2);
    Daughter = Branches(iBr,3);
    [~,indmin]=min(abs(MasterNodes{Mother}-MotherPt));
    MasterNodes{Mother}(indmin)=[];
    MasterNodes{Daughter}(1)=[];
end

% Add link pts
SlaveNodes = cell(nFib,1);
for iLink=1:Nlinks
    MasterFib = CrossLinks(iLink,1);
    MasterPt = CrossLinks(iLink,2);
    SlaveFib = CrossLinks(iLink,3);
    SlavePt = CrossLinks(iLink,4);
    MasterNodes{MasterFib}=[MasterNodes{MasterFib};MasterPt];
    SlaveNodes{SlaveFib}=[SlaveNodes{SlaveFib};SlavePt MasterFib length(MasterNodes{MasterFib}) iLink];
end
for iBr=1:NBranch
    Mother = Branches(iBr,1);
    MotherPt = Branches(iBr,2);
    Daughter = Branches(iBr,3);
    MasterNodes{Mother}=[MasterNodes{Mother};MotherPt];
    SlaveNodes{Daughter}=[0 Mother length(MasterNodes{Mother}) -1; SlaveNodes{Daughter}];
end

% Step 2: define the grid for the tangent vectors
TangentVectorNodes = cell(nFib,1);
[sTau,~] = chebpts(Nx-1,[0 L], 1);
[~,FixedFib]=min(nSlaveNodes); % Fix one of the fibers
for iFib=1:nFib
    TangentVectorNodes{iFib}=sTau;
    nSlaveNodes(iFib)=length(SlaveNodes{iFib}(:,1));
end
% First remove tangent vectors closest to the SLAVE nodes
for iFib=1:nFib
    SlavePts = SlaveNodes{iFib};
    stSlave=2;
    if (iFib==FixedFib)
        stSlave=1;
    end
    for iP=stSlave:nSlaveNodes(iFib)
        [~,indmin]=min(abs(TangentVectorNodes{iFib}-SlavePts(iP,1)));
        TangentVectorNodes{iFib}(indmin)=[];
    end
end
% Now deal with the branches 
for iBr=1:NBranch
    Mother = Branches(iBr,1);
    MotherPt = Branches(iBr,2);
    Daughter = Branches(iBr,3);
    [~,indmin]=min(abs(TangentVectorNodes{Mother}-MotherPt));
    TangentVectorNodes{Mother}(indmin)=MotherPt;
    [~,indmin]=min(abs(TangentVectorNodes{Daughter}));
    TangentVectorNodes{Daughter}(indmin)=0;
end
   
% The number of tangent vectors is 1 less than the number of master nodes - 
% now integrate the tangent vectors to get the position at the master
% nodes. Set the integration constant so that the first slave node is on
% top of its master (keeping the filament with the fewest number of slave
% nodes fixed). 

% Compute master nodes
MasterFirstSlaveFromTau = cell(nFib,1);
StartNodeByFib = zeros(nFib,1);
StartNodeByFib(1)=1;
nTaus = zeros(nFib,1);
for iFib=1:nFib
    sTauFib = TangentVectorNodes{iFib};
    nTaus(iFib) = length(sTauFib);
    [sint,~,bint]=chebpts(nTaus(iFib),[0 L],1);
    if (iFib==FixedFib)
        MasterFirstSlaveFromTau{iFib}=barymat(MasterNodes{iFib},sX,bX)*pinv(DX)*...
            barymat(sX,sint,bint)*barymat(sTauFib,sint,bint)^(-1);
        StartNodeByFib(iFib+1) = StartNodeByFib(iFib)+length(MasterNodes{iFib});
    else
        AllNodesThis = [MasterNodes{iFib};SlaveNodes{iFib}(1)];
        MasterFirstSlaveFromTau{iFib}=(barymat(AllNodesThis,sX,bX)-barymat(AllNodesThis(end,:),sX,bX))*pinv(DX)*...
            barymat(sX,sint,bint)*barymat(sTauFib,sint,bint)^(-1);
        StartNodeByFib(iFib+1) = StartNodeByFib(iFib)+length(MasterNodes{iFib})+1;
    end
end
MasterNodesMatrix = [blkdiag(MasterFirstSlaveFromTau{:}) zeros(sum(nTaus+1),Nlinks); ...
    zeros(Nlinks,sum(nTaus)) eye(Nlinks)];

% Compute a starting configuration based on tangent vectors and first slave
% nodes (fibers are initially straight). This then gives you the length of
% the links which are not free
FirstSlaves=[];
for iFib=1:nFib
    if (iFib~=FixedFib)
        FirstSlaves=[FirstSlaves; iFib SlaveNodes{iFib}(1,:)];
    end
end
FoundInitial=zeros(1,nFib);
X0s=zeros(nFib,3);
MotherFibs = FixedFib;
FoundInitial(FixedFib)=1;
while (any(~FoundInitial))
    % Set everyone who is slave to mother
    for iM=1:length(MotherFibs)
        MotherFib = MotherFibs(iM);
        IndsToSet=find(FirstSlaves(:,3)==MotherFib);
        SetSlave = eye(size(MasterNodesMatrix,1));
        for jF=1:length(IndsToSet)
            RowInd = IndsToSet(jF);
            jFib = FirstSlaves(RowInd,1);
            LinkIndex=FirstSlaves(RowInd,5);
            MotherPt = X0s(MotherFib,:)+Taus(MotherFib,:)*MasterNodes{MotherFib}(FirstSlaves(RowInd,4));
            if (LinkIndex<1) % a branch
                X0s(jFib,:) = MotherPt;
                % Set the corresponding entry in master nodes matrix
                SetSlave(StartNodeByFib(jFib):StartNodeByFib(jFib+1)-1,...
                    StartNodeByFib(MotherFib)-1+FirstSlaves(RowInd,4))=1;
            else % a cross link
                LinkPt = MotherPt + LinkLengths(LinkIndex)*LinkHats(LinkIndex,:);
                X0s(jFib,:) = LinkPt - Taus(jFib,:)*FirstSlaves(RowInd,2);
                % Set the corresponding entry in master nodes matrix
                SetSlave(StartNodeByFib(jFib):StartNodeByFib(jFib+1)-1,...
                    StartNodeByFib(MotherFib)-1+FirstSlaves(RowInd,4))=1;
                SetSlave(StartNodeByFib(jFib):StartNodeByFib(jFib+1)-1,...
                    StartNodeByFib(end)-1+LinkIndex)=LinkLengths(LinkIndex);
            end
            FoundInitial(jFib)=1;
        end
        MasterNodesMatrix = SetSlave*MasterNodesMatrix;
    end
    MotherFibs =  FirstSlaves(IndsToSet,1);
end

% From the initial configuration, recompute the link lengths and hats (has
% to be (slave - master))
for iLink=1:Nlinks
    SlaveFib = CrossLinks(iLink,3);
    SlavePt = X0s(SlaveFib,:)+Taus(SlaveFib,:)*CrossLinks(iLink,4);
    MasterFib = CrossLinks(iLink,1);
    MasterPt = X0s(MasterFib,:)+Taus(MasterFib,:)*CrossLinks(iLink,2);
    R = SlavePt-MasterPt;
    LinkLengths(iLink) = norm(R);
    LinkHats(iLink,:) = R/norm(R);
end

% Compute slave nodes
SlaveNodesFromMaster = cell(nFib,nFib);
SlaveNodesFromLinks = cell(nFib,1); 
for iFib=1:nFib
    TheseSlave = SlaveNodes{iFib};
    if (iFib~=FixedFib)
        TheseSlave = TheseSlave(2:end,:);
    end
    nSlave = length(TheseSlave(:,1));
    for jFib=1:nFib
        SlaveNodesFromMaster{iFib,jFib}=zeros(nSlave,length(MasterNodes{jFib})+(jFib~=FixedFib));
    end
    SlaveNodesFromLinks{iFib}=zeros(nSlave,Nlinks);
    for iSlave=1:nSlave
        jFib = TheseSlave(iSlave,2);
        MasterIndex = TheseSlave(iSlave,3);
        LinkIndex = TheseSlave(iSlave,4);
        SlaveNodesFromMaster{iFib,jFib}(iSlave,MasterIndex)=1;
        if (LinkIndex>0)
            SlaveNodesFromLinks{iFib}(iSlave,LinkIndex)=LinkLengths(LinkIndex);
        end
    end
end
% The DOFs are [tau1, tau2, ..., tauN, Link1, Link2, LinkM, Xbar]
SlaveNodesMatrix = [cell2mat(SlaveNodesFromMaster) cell2mat(SlaveNodesFromLinks)]*MasterNodesMatrix;
DOFsToCustomNodes = zeros(nFib*Nx,sum(nTaus)+Nlinks);
LastIndMaster=0;
LastIndSlave=0;
for iFib=1:nFib
    nSlave = nSlaveNodes(iFib)-(iFib~=FixedFib);
    nMaster = Nx-nSlave;
    % Master nodes first
    DOFsToCustomNodes((iFib-1)*Nx+(1:nMaster),:)=MasterNodesMatrix(LastIndMaster+(1:nMaster),:);
    DOFsToCustomNodes((iFib-1)*Nx+nMaster+(1:nSlave),:)=SlaveNodesMatrix(LastIndSlave+(1:nSlave),:);
    LastIndMaster=LastIndMaster+nMaster;
    LastIndSlave=LastIndSlave+nSlave;
end

DOFs = [];
for iFib=1:nFib
    DOFs = [DOFs; repmat(Taus(iFib,:),nTaus(iFib),1)];
end
DOFs = [DOFs; LinkHats];

Xcustom = DOFsToCustomNodes*DOFs;

MasterNodesNow=MasterNodesMatrix*DOFs;
SlaveNodesNow=SlaveNodesMatrix*DOFs;
X0s=X0s+MasterNodesNow(1,:);
for iFib=1:nFib
fpts=X0s(iFib,:)+(0:0.01:1)'*Taus(iFib,:);
plot3(fpts(:,1),fpts(:,2),fpts(:,3))
hold on
set(gca,'ColorOrderIndex',iFib)
scatter3(Xcustom((iFib-1)*Nx+(1:Nx),1),Xcustom((iFib-1)*Nx+(1:Nx),2),Xcustom((iFib-1)*Nx+(1:Nx),3),'filled')
end

NodesToChebMats = cell(nFib,1);
for iFib=1:nFib
    TotalNodes = [MasterNodes{iFib};SlaveNodes{iFib}(:,1)];
    ChebToNodes = barymat(TotalNodes,sX,bX);
    NodesToChebMats{iFib} = ChebToNodes^(-1);
end
AvgMat = 1/(nFib*L)*repmat(wX,1,nFib);
SubAvg = eye(Nx*nFib)-repmat(ones(Nx,1),nFib,1).*AvgMat;
ChebMatZeroMean = SubAvg*blkdiag(NodesToChebMats{:})*DOFsToCustomNodes;
DOFsToChebNodes = [ChebMatZeroMean ones(nFib*Nx,1)];
XchebZeroMean = DOFsToChebNodes*[DOFs; 0 0 0];
% for iFib=1:nFib
% set(gca,'ColorOrderIndex',iFib)
% scatter3(XchebZeroMean((iFib-1)*Nx+(1:Nx),1),XchebZeroMean((iFib-1)*Nx+(1:Nx),2),XchebZeroMean((iFib-1)*Nx+(1:Nx),3),'filled')
% end