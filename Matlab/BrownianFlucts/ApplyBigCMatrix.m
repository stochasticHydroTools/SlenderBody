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
    Ncc = size(NodesByBranch,1)*3;
    Ndd = length(x)-Nxx-Ncc;
    Lam1D=x(1:Nxx);
    Lam = reshape(Lam1D,3,[])';
    alphaU = reshape(x(Nxx+1:Nxx+Ndd),3,[])';
    Gamma = reshape(x(end-Ncc+1:end),3,[])';
    
    X3 = reshape(X1D,3,[])';
    DOFs = XInvFcn(X3);
    
    % Cross products
    NTau = size(DOFs,1)-1;
    CTau = alphaU;
    % Second eqn K^T Lam + B^T Gamma = 0 
    % K^T Lam = C^T X^T Lam = -C X^T Lam = cross(DOFs(iTau,:),XTLam(iTau,:)
    XTLam = XTrFcn(Lam);
    KTLam = XTLam;
    for iTau=1:NTau
        CTau(iTau,:)=cross(alphaU(iTau,:),DOFs(iTau,:));
        KTLam(iTau,:)=cross(DOFs(iTau,:),XTLam(iTau,:));
    end
    
    KAlpha=XFcn(CTau);
    KAlpha1D = reshape(KAlpha',[],1);
    
    MLam = zeros(Nxx,1);
    for iFib=1:nFib
        finds = 3*Nx*(iFib-1)+1:3*Nx*iFib;
        MWsymTildeOne = MobFcn(X1D(finds));
        MLam(finds)=MWsymTildeOne*(Lam1D(finds)+impcodt*BendForceMat*KAlpha1D(finds));
    end
    
    % Third equation (checking constraints on taus)
    nBr = size(NodesByBranch,1);
    BAlpha = zeros(nBr,3);
    BTGamma = zeros(NTau+1,3);
    for iBr=1:nBr
        iIndex = NodesByBranch(iBr,1);
        jIndex = NodesByBranch(iBr,2);
        BAlpha(iBr,:)=alphaU(iIndex,:)-alphaU(jIndex,:);
        BTGamma(iIndex,:)=Gamma(iBr,:);
        BTGamma(jIndex,:)=-Gamma(iBr,:);
    end

    Eq1 = -MLam + KAlpha1D;
    Eq2 = reshape((KTLam+BTGamma)',[],1);
    Eq3 = reshape(BAlpha',[],1);
    
    Ax=[Eq1;Eq2;Eq3];
end



