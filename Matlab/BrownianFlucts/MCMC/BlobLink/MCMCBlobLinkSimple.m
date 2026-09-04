function MCMCBlobLinkSimple(ds,seed)
addpath(genpath('../../'));
rng(seed)
% Generate initial chain
a=1e-2;
%ds = 0.04;
Nlinks = 1/ds;
L = ds*Nlinks;
kbT = 4.1e-3; % pN * um
lp = L;
K_b = lp*kbT;
nSamp = 1e5;
nSaveSamples = 0.8*nSamp;
nTrial = 1;
lpstar = (K_b)/kbT*1/L;
xUni = (0:ds:L)';
PropC = ds/lp; % constant for proposal
Confine=0;

%%% Base state
tau = [ones(Nlinks,1) zeros(Nlinks,2)];
X0 = [0 0 8*a];
X = [0 0 0; cumsum(ds*tau)]+X0;

% The energy matrix
N = length(tau)+1;
EnergyMat = diff(eye(N+4),4);
EnergyMat(:,[1 2 N+3 N+4]) = [];
EnergyMat(1,1:3) = -1*[-1 2 -1];
EnergyMat(2,1:4) = -1*[2 -5 4 -1];
EnergyMat(end,end:-1:end-2) = -1*[-1 2 -1];
EnergyMat(end-1,end:-1:end-3) = -1*[2 -5 4 -1];
EnergyMat = K_b*kron(sparse(EnergyMat),eye(3))/ds^3;

EPrev = 0;
nAcc=0;
nBins = 1000;
AllEndToEndDists = zeros(nTrial,nBins);
Deltas = (0:Nlinks-1)*ds;
AllTanVecDots = zeros(nTrial,length(Deltas));

for iTrial=1:nTrial
disp(strcat('New trial = ',num2str(iTrial)))
tic
MeanSqCoeffs = zeros(3*N,1);
TanVecDots = zeros(Nlinks,1);
nSamplesDs = zeros(Nlinks,1);
MeanDev = 0;
CovMat = zeros(3*N);
minZ=X0(3);
for iSamp=1:nSamp
    OmTau = randn(Nlinks,3);
    TauProp = rotateTau(tau,OmTau,PropC);
    X0prop = X0 + PropC*randn(1,3);
    Xprop = reshape((X0prop+[0 0 0; cumsum(ds*TauProp)])',[],1);
    % Eval energy
    Energy = 1/2*Xprop'*EnergyMat*Xprop;
    if (Confine)
        zloc = Xprop(3:3:end);
        Energy = Energy+20/2*(zloc-8*a)'*(zloc-8*a);
    end
    p_acc = exp(-Energy/kbT)/exp(-EPrev/kbT);
    r=rand;
    if (r < p_acc)
        % if (min(X(:,3))<minZ)
        %     minZ=min(X(:,3));
        %     if (minZ<a)
        %         keyboard
        %     end
        % end
        EPrev=Energy;
        tau = TauProp;
        X0 = X0prop;
        X = reshape(Xprop,3,[])';
        nAcc = nAcc+1;
    end
    % Compute coefficients and add to array
    if (iSamp > nSamp-nSaveSamples)
        % Sample the fiber at 5 points
        eedist = norm(X(1,:)-X(end,:));
        EndBinNum = min(ceil(eedist/L*nBins),nBins); % [0,1000]
        AllEndToEndDists(iTrial,EndBinNum)=AllEndToEndDists(iTrial,EndBinNum)+1;
        % Tangent vector dot products
        if (mod(iSamp,1000)==0)
        for iLink=1:Nlinks
            for jLink=iLink:Nlinks
                index = jLink-iLink+1;
                nSamplesDs(index)=nSamplesDs(index)+1;
                TanVecDots(index)=TanVecDots(index)+dot(tau(iLink,:),tau(jLink,:));
            end
        end
        end
    end
end
TanVecDots = TanVecDots./nSamplesDs;
AllTanVecDots(iTrial,:) = TanVecDots;
%save(strcat('FreeUnifMCMCkbT_Lp',num2str(lpstar),'.mat'))
toc
end
save(strcat('BLMCMC_Lp',num2str(lpstar),'_',num2str(seed),'.mat'))
end
% diffc = (0:Nlinks-1)*ds;
% errorbar(diffc,mean(AllTanVecDots),2*std(AllTanVecDots),'-o','LineWidth',2.0)
% hold on
% plot(diffc,exp(-diffc/lp))
%exit;
% Get the PDF of end to end dists




