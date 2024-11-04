names = ["SpecMCMCFreeL1_ConstKbT_N12_Lp2.mat" ];
for iName = 1:length(names)
load(names(iName))
plotIndex=iName;
nbins = 100;
%AllEndToEndDists = AllEndToQuarterDists;
%L = L;
histedges=(0:nbins)*L/nbins;
NumPerBin = 1000/nbins;
dshist = histedges(2)-histedges(1);
histmps = (histedges(1:end-1)+histedges(2:end))/2;
ChebHistCounts = zeros(nTrial,length(histedges)-1);
nSaveSamples=sum(AllEndToEndDists(1,:));
for iTrial=1:nTrial
    if (NumPerBin > 1)
        ChebHistCounts(iTrial,:) = sum(reshape(AllEndToEndDists(iTrial,:),NumPerBin,[]))/...
            (nSaveSamples*dshist);
    else
        ChebHistCounts(iTrial,:) = AllEndToEndDists(iTrial,:)/(nSaveSamples*dshist);
    end
end
if (plotIndex==length(names))
h(length(h)+1)=plot(histmps/L,mean(L*ChebHistCounts),'k');
else
set(gca,'ColorOrderIndex',plotIndex)
h(plotIndex)=plot(histmps/L,mean(L*ChebHistCounts));
end
hold on
errorBarEvery=2;
if (plotIndex==length(names))
errorbar(histmps(plotIndex:errorBarEvery:end)/L,...
    mean(L*ChebHistCounts(:,plotIndex:errorBarEvery:end)),...
    2*std(L*ChebHistCounts(:,plotIndex:errorBarEvery:end))/sqrt(nTrial),'ko','LineWidth',2.0,...
    'MarkerSize',1);
else
set(gca,'ColorOrderIndex',plotIndex)
errorbar(histmps(plotIndex:errorBarEvery:end)/L,...
    mean(L*ChebHistCounts(:,plotIndex:errorBarEvery:end)),...
    2*std(L*ChebHistCounts(:,plotIndex:errorBarEvery:end))/sqrt(nTrial),'o','LineWidth',2.0,...
    'MarkerSize',1);
hold on
end
if (iName==length(names)-1 && L ==2)% Theory curve
dr=1e-5;
r = (0.5:1/dr)'*dr;
G = zeros(length(r),1);
for ell=1:3
    G=G+1./(lpstar*(1-r)).^(3/2).*exp(-(ell-1/2)^2./(lpstar*(1-r))).*...
        (4*((ell-1/2)./sqrt(lpstar*(1-r))).^2-2);
end
% Estimate integral of G, normalize to 1
G=G.*r.^2;
G = G/sum(G*dr);
h(length(h)+1)=plot(r,G);
end
end