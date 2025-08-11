function Mem = InitializeMembraneDisc(N,Lm,Kc,Kh,dt,mu,uRatio)
    Mem.Lm = Lm;
    % If N is even make it odd
    if (mod(N,2)==0)
        N=N+1;
    end
    Mem.M = N;
    nPairs = (N-1)/2;
    Mem.dx=Mem.Lm/Mem.M;
    Mem.x=(0:Mem.M-1)*Mem.dx;
    [Mem.xg,Mem.yg]=meshgrid(Mem.x,Mem.x);
    Mem.kvals = [0:nPairs -nPairs:-1]*2*pi/Mem.Lm;
    [Mem.kx,Mem.ky]=meshgrid(Mem.kvals);
    Mem.ksq=Mem.kx.^2+Mem.ky.^2;
    % KSqDiag = diag(ksq(:));
    % FMatBase = dftmtx(M);
    % FMat2 = kron(FMatBase,FMatBase);
    Mem.Kcmem = Kc;
    %EnergyMatrixMem = Kcmem*real((FMat2'*(KSqDiag'*KSqDiag)*FMat2)/M^4*Lm^2);
    % Need to sort this out - just do spheres?
    hmem = 0.1;
    Mem.Mmem = eye(Mem.M^2)/(6*pi*mu*hmem);
    Mem.Mhalfmem = eye(Mem.M^2)/sqrt(6*pi*mu*hmem);
    %ImpMatMem = eye(M^2)/dt + Mmem*EnergyMatrixMem;
    %InvImpMatMem = ImpMatMem^(-1); % fix this later to Fourier
    Mem.h = zeros(Mem.M^2,1);
    Mem.Kh = Kh;
    Mem.FourierEnergyMat = Mem.Kcmem*Mem.ksq.^2*Mem.Lm^2/Mem.M^4;
    Mem.ImpfacFourier = (1/dt+Mem.Mmem(1,1)*Mem.FourierEnergyMat);

    % Upsampling
    Mem.uRatio = uRatio;
    Mem.Nu = uRatio*N;
    % If Nu is even make it odd
    if (mod(uRatio*N,2)==0)
        Mem.Nu = Mem.Nu+1;
        Mem.uRatio = Mem.Nu/Mem.M;
    end
    nPairs = (Mem.Nu-1)/2;
    Mem.wtu = (Mem.Lm/Mem.Nu)^2;
    Mem.xu=(0:Mem.Nu-1)*Mem.dx/Mem.uRatio;
    [Mem.xgu,Mem.ygu]=meshgrid(Mem.xu,Mem.xu);
    Mem.kvalsUp = [0:nPairs -nPairs:-1]*2*pi/Mem.Lm;
    [Mem.kxUp,Mem.kyUp]=meshgrid(Mem.kvalsUp);
    Mem.ksqUp=Mem.kxUp.^2+Mem.kyUp.^2;
    % Matrix format
    FMatBase = dftmtx(N);
    FMat2 = kron(FMatBase,FMatBase);
    FMatUp = dftmtx(Mem.Nu);
    FMatUp2 = 1/N^2*kron(FMatUp,FMatUp);
    PaddingMatrix = zeros(Mem.Nu^2,N^2);
    % "Paired modes"
    for iR=0:(N-1)/2
        for iC=0:(N-1)/2
            PaddingMatrix(iC*Mem.Nu+iR+1,iC*N+iR+1)=1;
            iCUp = Mem.Nu - iC;
            iCDwn = N-iC;
            PaddingMatrix((iCUp-1)*Mem.Nu+iR+1,(iCDwn-1)*N+iR+1)=1;
            iRUp = Mem.Nu  - iR;
            iRDwn = N-iR;
            PaddingMatrix(iC*Mem.Nu+iRUp,iC*N+iRDwn)=1;
            iCUp = Mem.Nu  - iC;
            iCDwn = N-iC;
            iRUp = Mem.Nu  - iR;
            iRDwn = N-iR;
            PaddingMatrix((iCUp-1)*Mem.Nu+iRUp,(iCDwn-1)*N+iRDwn)=1;
        end
    end
    Mem.UpsamplingMatrix=real(FMatUp2'*PaddingMatrix*FMat2);
end