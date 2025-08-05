%% Preliminaries
function Disc = InitializeDiscretization(Xs3,TrkPt,TrkLoc,Lin,Eb,rin,mu,clamp)
    [Disc.N,~]=size(Xs3);
    Disc.L = Lin;
    [Disc.s,Disc.w,Disc.b]=chebpts(Disc.N,[0 Disc.L],1);
    Disc.Nx = Disc.N+1;
    [Disc.sNp1,Disc.wNp1,Disc.bNp1]=...
        chebpts(Disc.Nx,[0 Disc.L],2);
    Disc.DNp1 = diffmat(Disc.Nx,[0 Disc.L],'chebkind2');
    Disc.RToNp1 = barymat(Disc.sNp1,Disc.s,Disc.b);
    Disc.RNp1ToN = barymat(Disc.s,Disc.sNp1,Disc.bNp1);
    Disc.IntDNp1 = pinv(Disc.DNp1);
    Disc.TrkLoc=TrkLoc;
    Disc.BMNp1 = stackMatrix(barymat(TrkLoc,Disc.sNp1,Disc.bNp1));
    % Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
    Disc.I=zeros(3*Disc.Nx,3);
    for iR=1:Disc.Nx
        Disc.I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
    end
    Disc.Xst= reshape(Xs3',[],1);
    Disc.TrkPt=TrkPt;    
    Disc.XonNp1Mat = [(eye(3*Disc.Nx)-repmat(Disc.BMNp1,Disc.Nx,1))*...
        stackMatrix(Disc.IntDNp1*Disc.RToNp1) Disc.I];
    Disc.InvXonNp1Mat = [stackMatrix(Disc.RNp1ToN*Disc.DNp1); Disc.BMNp1];
    
    Disc.Xt = Disc.XonNp1Mat*[Disc.Xst;Disc.TrkPt];
    % Bending energy matrix (2N+2 grid)
    [Disc.s2Np2, Disc.w2Np2, Disc.b2Np2] = chebpts(2*Disc.Nx, [0 Disc.L], 2);
    Disc.W2Np2 = diag(Disc.w2Np2);
    Disc.R_Np1_To_2Np2 = barymat(Disc.s2Np2,Disc.sNp1,Disc.bNp1);
    Disc.D2Np2 = diffmat(2*Disc.Nx,[0 Disc.L],'chebkind2');
    Disc.WTilde_Np1 = stackMatrix((Disc.R_Np1_To_2Np2'*Disc.W2Np2*...
        Disc.R_Np1_To_2Np2));
    Disc.WTilde_Np1_Inverse = Disc.WTilde_Np1^(-1);
    Disc.BendingEnergyMatrix_Np1 = Eb*stackMatrix((Disc.R_Np1_To_2Np2*Disc.DNp1^2)'*...
        Disc.W2Np2*Disc.R_Np1_To_2Np2*Disc.DNp1^2);
    Disc.BendForceMat = -Disc.BendingEnergyMatrix_Np1;
    Disc.BendMatHalf_Np1 = real(Disc.BendingEnergyMatrix_Np1^(1/2));
    Disc.Npl=100;
    [Disc.spl,Disc.wpl,Disc.bpl]=chebpts(Disc.Npl,[0 Disc.L]);
    Disc.RplNp1 = barymat(Disc.spl,Disc.sNp1,Disc.bNp1);
    Disc.r=rin;
    Disc.eps = Disc.r/Disc.L;
    Disc.MobConst = -log(Disc.eps^2)/(8*pi*mu);
    Disc.clamp=clamp;
end