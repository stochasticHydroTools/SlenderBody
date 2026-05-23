function x = PairwisePCConnNet(RHS,PrecompPCMats,paths,Connections,NodesByBranch,nFib,Nx)

    Nxx = 3*Nx*nFib;
    N = Nx - 1;
    BranchedFibs = Connections(Connections(:,5)==0,3);
    NFreeTaus = N*ones(nFib,1);
    NFreeTaus(BranchedFibs)=NFreeTaus(BranchedFibs)-1;
    TauIndices = [0;cumsum(NFreeTaus)];
    Ndd = length(RHS)-Nxx;
    
    Lam_All = zeros(Nxx,1);
    alphaU_All = zeros(Ndd,1);
    
    U = RHS(1:Nxx);
    V = RHS(Nxx+(1:Ndd));
   
    % Go connection by connection, the total solution is then a
    % superposition
    LinkNum=0;
    for iC=1:size(MasterConnections,1)
        ConnRow=MasterConnections(iC,:);

        
        upFib = ConnRow(1)
            Upstream = FilsInPath(j-1);
            upInds = ((Upstream-1)*3*Nx+1:Upstream*3*Nx)';
            upIndsTau = (3*TauIndices(Upstream)+1:3*TauIndices(Upstream+1))';
            jInds = ((jFib-1)*3*Nx+1:jFib*3*Nx)';
            jIndsTau= (3*TauIndices(jFib)+1:3*TauIndices(jFib+1))';
            UThis = U([upInds;jInds]);
            if (length(upIndsTau)<3*N)
                % Add the upstream nodes
                UpConn = find(ceil(NodesByBranch(:,2)/N)==Upstream);
                UpNode = NodesByBranch(UpConn,1);
                UpFib = ceil(UpNode/N);
                UpNode = UpNode-(UpFib-1)*N;
                Vindex = 3*(TauIndices(UpFib)+UpNode)+(-2:0)';
                upIndsTau = [upIndsTau;Vindex];
            else
            end
            VThis = V([upIndsTau; jIndsTau]);

            % Find the corresponding row in the connection matrix
            Row = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            ConnRow = Connections(Row,:);

            if (ConnRow(5)>0) % Cross link
                LinkNum=LinkNum+1;
                VThis=[VThis; V(3*TauIndices(end)+3*(LinkNum-1)+(1:3))];
            end
            VThis=[VThis;V(end-2:end)];

            % Solve system
            % Schur complement (slightly faster)
            NMat = PrecompPCMats{iPath,j-1,1};
            M = PrecompPCMats{iPath,j-1,2};
            K = PrecompPCMats{iPath,j-1,3};
            KWithImp = PrecompPCMats{iPath,j-1,4};
            KprimeMinvU = K' * (M \ UThis);
            alphaU = NMat*(KprimeMinvU+VThis);
            Lam = M \ (KWithImp*alphaU - UThis);

            Lam_All([upInds;jInds])=Lam_All([upInds;jInds])+Lam;
            if (ConnRow(5)>0)
                alphaU_All([upIndsTau;jIndsTau])=alphaU_All([upIndsTau;jIndsTau])+...
                    alphaU(1:end-6);
                alphaU_All(3*TauIndices(end)+3*(LinkNum-1)+(1:3))=...
                    alphaU(end-5:end-3);
            else
                alphaU_All([upIndsTau;jIndsTau])=alphaU_All([upIndsTau;jIndsTau])+...
                    alphaU(1:end-3);
            end
            alphaU_All(end-2:end)=alphaU_All(end-2:end)+alphaU(end-2:end);
        end
    end
    x=[Lam_All;alphaU_All];
end