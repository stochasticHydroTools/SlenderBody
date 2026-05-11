function x = PairwisePCConnNet(RHS,X1D,XInvFcn,MobFcn,NodesByBranch,...
    PairwiseXMats,paths,Connections,BendForceMat,impcodt,nFib,ell)

    Nxx = length(X1D);
    Nx = Nxx/(3*nFib);
    N = Nx - 1;
    Ncc = size(NodesByBranch,1)*3;
    Ndd = length(RHS)-Nxx-Ncc;
    
    Lam_All = zeros(Nxx,1);
    alphaU_All = zeros(Ndd,1);
    Gamma_All = zeros(Ncc,1);
    
    X3 = reshape(X1D,3,[])';
    DOFs = XInvFcn(X3);
    U = RHS(1:Nxx);
    V = RHS(Nxx+(1:Ndd));
    W = RHS(Nxx+Ndd+(1:Ncc));
   
    % Go connection by connection, the total solution is then a
    % superposition
    LinkNum=0;
    nPaths=length(paths);
    for iPath=1:nPaths
        FilsInPath = paths{iPath};
        for j=2:length(FilsInPath)
            jFib = FilsInPath(j);
            Upstream = FilsInPath(j-1);
            upInds = ((Upstream-1)*3*Nx+1:Upstream*3*Nx)';
            upIndsTau = ((Upstream-1)*3*N+1:Upstream*3*N)';
            jInds = ((jFib-1)*3*Nx+1:jFib*3*Nx)';
            jIndsTau= ((jFib-1)*3*N+1:jFib*3*N)';
            UThis = U([upInds;jInds]);
            VThis = V([upIndsTau;jIndsTau]);

            % Find the corresponding row in the connection matrix
            Row = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            ConnRow = Connections(Row,:);
            XMat=PairwiseXMats{iPath,j-1};

            TheseTaus = DOFs([(Upstream-1)*N+1:Upstream*N;...
                (jFib-1)*N+1:jFib*N]',:);
            if (ConnRow(5)>0) % Cross link
                LinkNum=LinkNum+1;
                TheseTaus = [TheseTaus; DOFs(nFib*N+LinkNum,:)];
                VThis=[VThis; V(3*nFib*N+3*(LinkNum-1)+(1:3))];
            end
            VThis=[VThis;V(end-2:end)];

            K =  KWithLink(XMat,TheseTaus);
            M = blkdiag(MobFcn(X1D(upInds)),MobFcn(X1D(jInds)));
            KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
        
            if ~isempty(NodesByBranch)
                BranchedFibs = ceil(NodesByBranch/N);
                BranchInd = find((BranchedFibs(:,1)==Upstream & BranchedFibs(:,2)==jFib) | ...
                    (BranchedFibs(:,2)==Upstream & BranchedFibs(:,1)==jFib));
            else
                BranchInd=[];
            end
            if (~isempty(BranchInd))
                IndOnI = NodesByBranch(BranchInd,1)-(Upstream-1)*N;
                IndOnJ = NodesByBranch(BranchInd,2)-(jFib-1)*N;
                if (BranchedFibs(BranchInd,2)==Upstream)
                    IndOnI = NodesByBranch(BranchInd,2)-(Upstream-1)*N;
                    IndOnJ = NodesByBranch(BranchInd,1)-(jFib-1)*N;
                end
                B = zeros(1,2*N+1);
                B(N+IndOnJ)=-1;
                B(IndOnI)=1;
                B = stackMatrix(B);
                BigSys = [-M KWithImp zeros(6*Nx,3); K' zeros(3*(2*N+1)) B'; ...
                    zeros(3,6*Nx) B zeros(3)];
                WThis = W(3*(BranchInd-1)+(1:3));
            else
                BigSys = [-M KWithImp; K' zeros(6*Nx)];
                WThis=[];
            end
            
            % Solve system
            % Schur complement (slightly faster)
            % NMat = pinv(K'*(M\KWithImp));
            % KprimeMinvU = K' * (M \ UThis);
            % Gam = (B*NMat*B') \ (B*NMat*KprimeMinvU - WThis+B*NMat*VThis);
            % alphaU = NMat*(KprimeMinvU-B'*Gam+VThis);
            % Lam = M \ (KWithImp*alphaU - UThis);
            % x1=[Lam;alphaU;Gam];

            x1=lsqminnorm(BigSys,[UThis;VThis;WThis]);
            Lam_All([upInds;jInds])=Lam_All([upInds;jInds])+x1(1:6*Nx);
            alphaU_All([upIndsTau;jIndsTau])=alphaU_All([upIndsTau;jIndsTau])+...
                x1(6*Nx+1:6*Nx+6*N);
            if (~isempty(BranchInd))
                Gamma_All(3*(BranchInd-1)+(1:3))=x1(end-2:end);
                U0 = x1(end-5:end-3);
            else
                alphaU_All(nFib*3*N+(LinkNum-1)*3+(1:3))=x1(6*Nx+6*N+(1:3));
                U0 = x1(end-2:end);
            end
            alphaU_All(end-2:end)=alphaU_All(end-2:end)+U0;
        end
    end
    x=[Lam_All;alphaU_All;Gamma_All];
end

function KTogether = KWithLink(XMat,Tau3)
    TauVelocity = zeros(3*size(Tau3,1)+3);
    % The matrix for all the taus (incl links) to evolve
    for iR =1:size(Tau3,1)
        inds = (iR-1)*3+1:iR*3;
        CMat = CPMatrix(Tau3(iR,:));
        TauVelocity(inds,inds) =  -CMat;
    end
    % The COM
    TauVelocity(end-2:end,end-2:end)=eye(3);
    KTogether = XMat*TauVelocity;
end