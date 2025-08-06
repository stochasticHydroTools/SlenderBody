function Disc=EvolveClampedFil(Disc,kbT,dt,impcoeff,RandomNumbers,Fext)
    MobConst=Disc.MobConst;
    Nx = Disc.Nx;
    DNp1 = Disc.DNp1;
    N = Nx - 1;
    WTilde_Np1_Inverse=Disc.WTilde_Np1_Inverse;
    XonNp1Mat = Disc.XonNp1Mat;
    Xs3 = reshape(Disc.Xst,3,[])';
    Xti = XonNp1Mat*[Disc.Xst;Disc.TrkPt];
    MWsym = LocalDragMob(Xti,DNp1,MobConst,WTilde_Np1_Inverse);
    MWsymHalf = real(MWsym^(1/2));
    % Obtain Brownian velocity
    XonNp1Mat = Disc.XonNp1Mat;
    I = Disc.I;
    K = KonNp1(Xs3,XonNp1Mat,I);
    if (Disc.clamp)
        sNp1 = Disc.sNp1;
        bNp1 = Disc.bNp1;
        s = Disc.s;
        b = Disc.b;
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
    end
    g = RandomNumbers(1:3*Nx);
    RandomVelBM = sqrt(2*kbT/dt)*MWsymHalf*g;
    % Advance to midpoint
    RNp1ToN = Disc.RNp1ToN;
    DNp1 = Disc.DNp1;
    OmegaTilde = cross(Xs3,RNp1ToN*DNp1*reshape(RandomVelBM,3,[])');
    if (Disc.clamp)
        OmegaTilde = ProjectClamp*[reshape(OmegaTilde',[],1);zeros(3,1)];
        OmegaTilde = reshape(OmegaTilde,3,[])';
    end
    Xstilde = rotateTau(Xs3,OmegaTilde(1:N,:),dt/2);
    Ktilde = KonNp1(Xstilde,XonNp1Mat,I);
    Xtilde = XonNp1Mat*[reshape(Xstilde',[],1);Disc.TrkPt];
    MWsymTilde = LocalDragMob(Xtilde,DNp1,MobConst,WTilde_Np1_Inverse);
    if (Disc.clamp)
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*Ktilde;...
            stackMatrix([barymat(0,s,b) 0])];
        ProjectClampTilde = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = Ktilde*ProjectClampTilde;
    else
        Kaug = Ktilde;
    end
    deltaRFD = 1e-5;
    WRFD = RandomNumbers(3*Nx+1:6*Nx);
    OmegaPlus = cross(Xs3,RNp1ToN*DNp1*reshape(WRFD,3,[])');
    if (Disc.clamp)
        OmegaPlus = ProjectClamp*[reshape(OmegaPlus',[],1);zeros(3,1)];
        OmegaPlus = reshape(OmegaPlus,3,[])';
    end
    TauPlus = rotateTau(Xs3,deltaRFD*OmegaPlus(1:N,:),1);
    XPlus = XonNp1Mat*[reshape(TauPlus',[],1);Disc.TrkPt];
    MWsymPlus = LocalDragMob(XPlus,DNp1,MobConst,WTilde_Np1_Inverse);
    M_RFD = kbT/deltaRFD*(MWsymPlus-MWsym)*WRFD;
    BERand = RandomNumbers(6*Nx+1:end);
    BendMatHalf_Np1 = Disc.BendMatHalf_Np1;
    RandomVelBE = sqrt(kbT)*MWsymTilde*BendMatHalf_Np1*BERand;
    RandomVel = RandomVelBM + M_RFD + RandomVelBE;
    B = Kaug-impcoeff*dt*MWsymTilde*Disc.BendForceMat*Kaug;
    U0 = zeros(3*Nx,1);
    %U0(1:3:end)=1;
    %Fext = zeros(3*Nx,1);
    %Fext(end-1) = 100*Eb;
    RHS = Kaug'*(Disc.BendForceMat*Xti+ Fext + ...
        MWsymTilde \ (RandomVel + U0));
    % Form psuedo-inverse manually
    maxRank = 2*N+3;
    if (Disc.clamp)
        maxRank = 2*N-2;
    end
    alphaU = ManualPinv(Kaug'*(MWsymTilde \ B),maxRank)*RHS;
    Omega = reshape(alphaU(1:3*N),3,N)';
    newXs = rotateTau(Xs3,Omega,dt);
    Disc.Xst = reshape(newXs',[],1);
    Disc.TrkPt = Disc.TrkPt+dt*alphaU(end-2:end);
    Disc.Xt = Disc.XonNp1Mat*[Disc.Xst;Disc.TrkPt];
end