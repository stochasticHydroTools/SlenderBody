function Disc = PolymerizeClampedFib(Disc,Mem,ratePoly,deltaP)
    if (ratePoly < rand) 
        return
    end
    % Possible polymerization event - check if membrane accomodates
    TauLast = barymat(Disc.L,Disc.s,Disc.b)*reshape(Disc.Xst,3,[])';
    XLast = barymat(Disc.L,Disc.sNp1,Disc.bNp1)*reshape(Disc.Xt,3,[])';
    Xadded = TauLast/norm(TauLast)*deltaP+XLast;
    % Check the membrane position at this (x,y)
    hmempt = Interpolateh(Xadded(1:2),Mem,0);
    if (Xadded(3) > hmempt+1e-10)
        return
    end
    % Can do the polymerization step
    Disc.nPolEvents=Disc.nPolEvents+1;

    % Update the discretization
    Lfac = 1+deltaP/Disc.L;
    % Assuming you can reuse the same grid (later can regenerate grid)
    Disc.s = Lfac*Disc.s;
    Disc.w = Lfac*Disc.w;
    Disc.sNp1 = Lfac*Disc.sNp1;
    Disc.su = Lfac*Disc.su;
    Disc.wNp1 = Lfac*Disc.wNp1;
    Disc.DNp1 = Disc.DNp1/Lfac;
    Disc.IntDNp1 = Disc.IntDNp1*Lfac;
    Disc.XonNp1Mat = [(eye(3*Disc.Nx)-repmat(Disc.BMNp1,Disc.Nx,1))*...
        stackMatrix(Disc.IntDNp1*Disc.RToNp1) Disc.I];
    Disc.InvXonNp1Mat = [stackMatrix(Disc.RNp1ToN*Disc.DNp1); Disc.BMNp1];
    % Bending energy matrix (2N+2 grid)
    Disc.s2Np2 = Lfac*Disc.s2Np2;
    Disc.w2Np2 = Lfac*Disc.w2Np2;
    Disc.W2Np2 = diag(Disc.w2Np2);
    Disc.D2Np2 = Disc.D2Np2/Lfac;
    Disc.WTilde_Np1 = Disc.WTilde_Np1*Lfac;
    Disc.WTilde_Np1_Inverse = Disc.WTilde_Np1_Inverse/Lfac;
    Disc.BendingEnergyMatrix_Np1 = Disc.BendingEnergyMatrix_Np1/Lfac^3;
    Disc.BendForceMat = -Disc.BendingEnergyMatrix_Np1;
    Disc.BendMatHalf_Np1 = Disc.BendMatHalf_Np1/Lfac^(3/2);
    Disc.L = Disc.L*Lfac;
    Disc.eps = Disc.r/Disc.L;
    Disc.MobConst = -log(Disc.eps^2)/(8*pi*Disc.mu);
    PositionsToMatch = [Disc.sNp1/Lfac; Disc.L];
    Rnew = stackMatrix(barymat(PositionsToMatch,Disc.sNp1,Disc.bNp1));
    Disc.su = Lfac*Disc.su;
    Disc.wu = Lfac*Disc.wu;
    % Fill in what you know
    Tau0BC = Disc.Xst(1:3);
    ErrorNew=@(NewDOF) (Rnew*NewPos(NewDOF,Disc.XonNp1Mat,Tau0BC',Disc.TrkPt,...
        Disc.N) - [Disc.Xt;Xadded']);
    [azimuth,elevation,~] = cart2sph(Disc.Xst(4:3:end),...
        Disc.Xst(5:3:end),Disc.Xst(6:3:end));
    x0=[azimuth;elevation];
    % Unconstrained optimization 
    opts = optimset('display','off');
    NewDOFs = lsqnonlin(ErrorNew,x0,[-pi*ones(Disc.N-1,1); -pi/2*ones(Disc.N-1,1)],...
        [pi*ones(Disc.N-1,1); pi/2*ones(Disc.N-1,1)],opts);
    % Get new Tau's
    X = NewPos(NewDOFs,Disc.XonNp1Mat,Tau0BC',Disc.TrkPt,Disc.N);
    TausTrk = Disc.XonNp1Mat \ X;
    Disc.Xt = X;
    Disc.Xst = TausTrk(1:3*Disc.N);
end

function X = NewPos(DOFs,XonNp1Mat,Tau0,XTrk,N)
    azimuth = DOFs(1:N-1);
    elevation = DOFs(N:2*N-2);
    r = ones(N-1,1);
    [x,y,z] = sph2cart(azimuth,elevation,r);
    tau3 = [Tau0;[x y z]];
    X = XonNp1Mat*[reshape(tau3',[],1);XTrk];
end
