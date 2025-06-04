%% Preliminaries
Nx = N+1;
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*(N+1),3);
for iR=1:N+1
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
Xst = zeros(3*N*nFib,1);
Xt = zeros(3*Nx*nFib,1);
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*...
    stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
for iFib=1:nFib
    Xst((iFib-1)*3*N+1:iFib*3*N) = ...
        reshape(X_s((iFib-1)*N+1:iFib*N,:)',[],1);
    Xt((iFib-1)*3*Nx+1:iFib*3*Nx) = XonNp1Mat*...
        [Xst((iFib-1)*3*N+1:iFib*3*N);XMP(:,iFib)];
end
% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, b2Np2] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
D2Np2 = diffmat(2*Nx,[0 L],'chebkind2');
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
saveEvery=max(floor(5e-2/dt),1);
% Hydrodynamics
NupsampleHydro = 200;%ceil(2/a);
[sup,wup,~] = chebpts(NupsampleHydro, [0 L],2);
RupsampleHydro = stackMatrix(barymat(sup,sNp1,bNp1));
WUpHydro = stackMatrix(diag(wup));
BDCell = repmat({RupsampleHydro},nFib,1);
RupsampleHydro_BD = blkdiag(BDCell{:});
BDCell = repmat({WUpHydro},nFib,1);
WUpHydro_BD = blkdiag(BDCell{:});
BDCell = repmat({WTilde_Np1_Inverse},nFib,1);
WTInv_BD = blkdiag(BDCell{:});
AllbS_Np1 = precomputeStokesletInts(sNp1,L,a,N+1,1);
AllbD_Np1 = precomputeDoubletInts(sNp1,L,a,N+1,1);
NForSmall = 8; % # of pts for R < 2a integrals for exact RPY
eigThres = 1e-3;
% Plotting
Npl=1000;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);