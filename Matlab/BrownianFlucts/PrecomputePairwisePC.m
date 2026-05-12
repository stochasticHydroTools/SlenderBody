function PrecompPCs = PrecomputePairwisePC(X1D,XInvFcn,MobFcn,...
    PairwiseXMats,paths,Connections,BendForceMat,impcodt,nFib)

    Nxx = length(X1D);
    Nx = Nxx/(3*nFib);
    N = Nx - 1;
    
    X3 = reshape(X1D,3,[])';
    DOFs = XInvFcn(X3);
   
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
            jInds = ((jFib-1)*3*Nx+1:jFib*3*Nx)';

            % Find the corresponding row in the connection matrix
            Row = find((Connections(:,1)==jFib & Connections(:,3)==Upstream) | ...
                (Connections(:,3)==jFib & Connections(:,1)==Upstream));
            ConnRow = Connections(Row,:);
            XMat=PairwiseXMats{iPath,j-1,1};

            TheseTaus = DOFs([(Upstream-1)*N+1:Upstream*N;...
                (jFib-1)*N+1:jFib*N]',:);
            if (ConnRow(5)>0) % Cross link
                LinkNum=LinkNum+1;
                TheseTaus = [TheseTaus; DOFs(nFib*N+LinkNum,:)];
                AssignMat=1;
            else
                AssignMat = PairwiseXMats{iPath,j-1,2};
            end

            K =  KWithLink(XMat,TheseTaus,AssignMat);
            M = blkdiag(MobFcn(X1D(upInds)),MobFcn(X1D(jInds)));
            KWithImp = K-impcodt*M*blkdiag(BendForceMat,BendForceMat)*K;
            
            % Compute the SVD (pseudo-inverse)
            NMat = pinv(K'*(M\KWithImp));
            PrecompPCs{iPath,j-1,1}=NMat;
            PrecompPCs{iPath,j-1,2}=M;
            PrecompPCs{iPath,j-1,3}=K;
            PrecompPCs{iPath,j-1,4}=KWithImp;
        end
    end
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