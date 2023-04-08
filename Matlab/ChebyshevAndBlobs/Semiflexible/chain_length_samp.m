clear
%tiledlayout(1,2, 'Padding', 'none', 'TileSpacing', 'compact');
for l_p=[0.5]
for plotTaus=[0]
%nexttile
%plotTaus=1;
rng(0);
ds = 0.005;
L = 1;
%l_p=1;
gamma = (1/ds) * l_p; %exponential factor for angle distribution
Nlinks = round(L/ds);
Nblob = Nlinks+1;
sBlob=(0:Nblob-1)'*ds;
sLink=(0.5:Nlinks)'*ds;
Nsamp = 1;

Nruns = 1;%40; % number of independent runs for samples (to get error bars)
nbins = 21; % bins for histogram
histedges=(0:nbins-1)*L/(nbins-1);
dshist = histedges(2)-histedges(1);
histmps = (histedges(1:end-1)+histedges(2:end))/2;
Hd = NaN(Nruns,nbins-1);
r = linspace(0,1,nbins);
r = 0.5*(r(2:end) + r(1:end-1));

%for bReg=[0.1 0.05 0.025]
clear normXCoeffs normTau L2_Error
MeanOptEr=0;
for i = 1:Nruns
X_L = NaN(1,Nsamp);
alphas = rand_alpha(gamma,(Nlinks-1)*Nsamp);
alphas = reshape(alphas,Nsamp,(Nlinks-1));
EndToEndDistances = zeros(Nsamp,1);
for j = 1:Nsamp
alphas_l = alphas(j,:);
taus = Random_Chain(Nlinks,alphas_l);
XBL=[0 0 0; cumsum(taus'*ds)];
TauBL = taus';
if (~plotTaus)
plot3(XBL(:,1),XBL(:,2),XBL(:,3),'-k')
hold on
end
%for plotTaus=[0 1]
NChebs=36*ones(1,7);
NChebs=[8 12 16 24];
% Solve optimization prblem to find how far
N=NChebs(1);
% if (j==1)
% QuadProgram=1; AdHoc=0; BVP=0;
% SpectralXFromBlob;
% MeanOptEr=MeanOptEr+opter/Nruns;
% end
%bRegs = [0.8 0.4 0.2 0.1 0.05 0.025 0.01];
bRegs = zeros(length(NChebs),1);
QuadProgram=1; AdHoc=0; BVP=0;
for iChebPts=1:length(NChebs)
N = NChebs(iChebPts);
bReg = bRegs(iChebPts);
SpectralXFromBlob;
% Compute spectrum
CoeffsToVals = cos(acos(2*sNp1/L-1).*(0:Nx-1));
XCoeffs = CoeffsToVals \ reshape(X,3,[])';
normXCoeffs{iChebPts}(j,:)=sqrt(sum(XCoeffs.*XCoeffs,2));
%normTau{iChebPts}(j,:)=sqrt(sum(tau.*tau,2));
%L2_Error{iChebPts}(j) = bvpper;
end
end
end
end
end
%nexttile;
% for iChebPts=1:length(NChebs)
% MeanL2ers{iChebPts}(i)=mean(L2_Error{iChebPts});
% MeanNormTau{iChebPts}(i,:)=mean(normTau{iChebPts});
% MeanXCoeffs{iChebPts}(i,:)=mean(normXCoeffs{iChebPts});
% end
% end
% for iChebPts=1:length(NChebs)
% MeanMeanL2ers(iChebPts)=mean(MeanL2ers{iChebPts}-MeanOptEr);
% StdMeanL2ers(iChebPts)=std(MeanL2ers{iChebPts}-MeanOptEr);
% MeanMeanNormTau{iChebPts}=mean(MeanNormTau{iChebPts});
% MeanMeanXCoeffs{iChebPts}=mean(MeanXCoeffs{iChebPts});
% end
% errorbar(bRegs/sqrt(L*l_p),MeanMeanL2ers,2*StdMeanL2ers/(sqrt(Nruns)),'-o','LineWidth',2.0)
% hold on
% %MeanMeanL2ers
% %mean(MeanMeanNormTau{1})-1
% %end