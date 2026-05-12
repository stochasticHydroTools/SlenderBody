% Apply the big matrix
% Input: vector of (Lambda,alphaU,Gamma)
% Output: [-M*Lambda+K*alphaU; K'*Lambda + B'*Gamma; B*alpha]
% (without explicitly forming matrices)
% First one is easy: apply the mobility fiber by fiber
% Compute DOFs
function Ax = ApplyBigCMatrix(x,X1D,XFcn,XInvFcn,XTrFcn,MobFcn,NodesByBranch,...
    BendForceMat,impcodt,nFib)

    Nxx = length(X1D);
    Nx = Nxx/(3*nFib);
    nBranch = size(NodesByBranch,1);
    Lam1D=x(1:Nxx);
    Lam = reshape(Lam1D,3,[])';
    alphaU = reshape(x(Nxx+1:end),3,[])';
    
    X3 = reshape(X1D,3,[])';
    DOFs = XInvFcn(X3);
    
    % First equation
    % Assign branch nodes
    for iBr=1:nBranch
        masternode = NodesByBranch(iBr,1);
        slavenode = NodesByBranch(iBr,2);
        alphaU = [alphaU(1:slavenode-1,:); alphaU(masternode,:); ...
            alphaU(slavenode:end,:)];
    end
    NTau = size(DOFs,1)-1;
    CTau = alphaU;
    for iTau=1:NTau
        CTau(iTau,:)=cross(alphaU(iTau,:),DOFs(iTau,:));
    end
    KAlpha=XFcn(CTau);
    KAlpha1D = reshape(KAlpha',[],1);

    MLam = zeros(Nxx,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = MobFcn(X1D(finds));
        MLam(finds)=MWsymTildeOne*(Lam1D(finds)+impcodt*BendForceMat*KAlpha1D(finds));
    end
    Eq1 = -MLam+KAlpha1D;

    % Second eqn K^T Lam = 0 
    % K^T Lam = C^T X^T Lam = -C X^T Lam = cross(DOFs(iTau,:),XTLam(iTau,:)
    XTLam = XTrFcn(Lam);
    
    KTLam = XTLam;
    for iTau=1:NTau
        KTLam(iTau,:)=cross(DOFs(iTau,:),XTLam(iTau,:));
    end
    % Assign the branch nodes
    KTLam(NodesByBranch(:,1),:)=KTLam(NodesByBranch(:,1),:)+...
        KTLam(NodesByBranch(:,2),:);
    KTLam(NodesByBranch(:,2),:)=[];
    Eq2 = reshape(KTLam',[],1);
    
    Ax=[Eq1;Eq2];
end



