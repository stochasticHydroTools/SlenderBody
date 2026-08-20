function MCMCClamped(Nx,gtype)
% Generate initial chain
addpath(genpath('../../'))
clampL=0;
clamp0=0;
L = 1;
kbT = 4.1e-3; % pN * um
lp = 1*L;
Eb = lp*kbT;
nSamp = 5e8;
nSaveSamples = 0.5*nSamp;
nTrial = 10;
%Nx = 8; % number tangent vectors
N = Nx - 1;
TauConst=5e-3*8/Nx;
lpstar = Eb/kbT*1/L;
%gtype = 1;
Tau0BC = [0;1;0];
TrkLoc=0;
nBins = 100;
AllEndToEndDists = zeros(nTrial,nBins);

%%% Base state %%%
try
    [s,~,b] = chebpts(N, [0 L], gtype);
catch
    [s,~,b] = chebpts(N, [0 L], 1);
end
% Add rows for the constraints 
sC=s;
if (gtype==1)
    % Replace first and last entry with L
    sC(1)=0;
    if (clampL)
        sC(end)=L;
    end
    ChebToConstr = barymat(sC,s,b);
    ConstrToCheb = ChebToConstr^(-1);
elseif (gtype=='u')
    sC=(0.5:N)'/(N)*L;
    ChebToConstr = barymat(sC,s,b);
    ConstrToCheb = ChebToConstr^(-1);
else
    ChebToConstr = eye(N);
    ConstrToCheb = eye(N);
end
Xs3=repmat(Tau0BC',N,1);
%Xs3=Xs3+sqrt(((L - sC).*sC)./(L*lp)).*randn(N,3);
%Xs3=Xs3./sqrt(sum(Xs3.*Xs3,2));

%%% Preliminary computations %%%
[sNp1,~,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(TrkLoc,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from X_s
XonNp1Mat = (eye(3*Nx)-repmat(BMNp1,Nx,1))*stackMatrix(IntDNp1*RToNp1*ConstrToCheb);

% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, ~] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);

% Propose a move around the state and evaluate its energy
X = XonNp1Mat*reshape(Xs3',[],1);
EPrev = 1/2*X'*BendingEnergyMatrix_Np1*X;
nAcc=0;
MeanTauSq = zeros(N,3,nTrial);
AllTanVecDots = zeros(nTrial,N);

for iTrial=1:nTrial
TanVecDots = zeros(N,1);
nSamplesDs = zeros(N,1);
disp(strcat('New trial = ',num2str(iTrial)))
tic
for iSamp=1:nSamp
    DTau = TauConst.*randn(N,3);
    % Project off zero
    if (clamp0)
        DTau(1,:)=0;
    end
    if (clampL)
        DTau(end,:)=0;
    end
    XsProp = rotateTauM(Xs3,DTau);
    % Eval energy
    % Upsample and integrate, then upsample again
    X = XonNp1Mat*reshape(XsProp',[],1);
    Energy = 1/2*X'*BendingEnergyMatrix_Np1*X;
    deltaE = (Energy-EPrev)/kbT;
    p_acc = exp(-deltaE);
    r=rand;
    if (r < p_acc)
        EPrev=Energy;
        Xs3 = XsProp;
        nAcc = nAcc+1;
    end
    if (iSamp > (nSamp-nSaveSamples))
        MeanTauSq(:,:,iTrial)=MeanTauSq(:,:,iTrial)+...
            Xs3.*Xs3;
        tau=Xs3;
        X3 = XonNp1Mat(1:3:end,1:3:end)*tau;
        EEDist=norm(X3(1,:)-X3(end,:));
        EndBinNum = min(ceil(EEDist/L*nBins),nBins); % [0,1000]
        AllEndToEndDists(iTrial,EndBinNum)=AllEndToEndDists(iTrial,EndBinNum)+1;
        % Tangent vector dot products
        for iLink=1:Nx-1
            for jLink=iLink:Nx-1
                index = jLink-iLink+1;
                nSamplesDs(index)=nSamplesDs(index)+1;
                TanVecDots(index)=TanVecDots(index)+dot(tau(iLink,:),tau(jLink,:));
            end
        end
    end
end
MeanTauSq(:,:,iTrial)=MeanTauSq(:,:,iTrial)/nSaveSamples;
TanVecDots = TanVecDots./nSamplesDs;
AllTanVecDots(iTrial,:) = TanVecDots;
toc
save(strcat('MCMC',num2str(gtype),'_Nx',num2str(Nx),'_Lp',num2str(lpstar),'.mat'))
end
end


function newXs = rotateTauM(Xsin,Omega)
    nOm = sqrt(sum(Omega.*Omega,2));
    % Have to truncate somewhere to avoid instabilities
    k = Omega./nOm;
    k(nOm < 1e-12,:) = 0;
    % Rodriguez formula on the N grid. 
    newXs = Xsin.*cos(nOm)+cross(k,Xsin).*sin(nOm)...
        +k.*sum(k.*Xsin,2).*(1-cos(nOm));
end
