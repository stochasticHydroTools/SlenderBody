N = 13;
F = 200;
step = F*N;
FileName = 'BundlingWithMotorsAll_1';
LoadFileName = strcat(FileName,'.txt'); % ALWAYS LOOK @ MOVIE!
Locs=load(strcat('Locs',LoadFileName));
Labels=load(strcat('FinalLabels_Sep',LoadFileName));
cd '/home/ondrejmaxian/Documents/SLENDER_FIBERS/SlenderBody/Python/plots/Motors/'
NBundlesPerstep_Sep = load(strcat('NumberOfBundles_Sep',LoadFileName));


[numTs, F] = size(Labels);
newlabels=zeros(numTs,F);
for iT=1:numTs
lab=1;
for iFib=1:F
if (newlabels(iT,iFib)==0)
iFiblab = Labels(iT,iFib);
if (sum(Labels(iT,:)==iFiblab)==1)
newlabels(iT,iFib)=-1;
else
newlabels(iT,Labels(iT,:)==iFiblab)=lab;
lab=lab+1;
end
end
end
% Rearrange to sort bundles by size
nPerLab=zeros(lab,1);
for iL=1:lab
nPerLab(iL)=sum(newlabels(iT,:)==iL);
end
temp = newlabels(iT,:);
[vals,newInds]=sort(nPerLab,'descend');
for iL=1:lab
temp(newlabels(iT,:)==newInds(iL))=iL;
end
newlabels(iT,:)=temp;
end
writematrix(newlabels,strcat('NewLabels_',FileName,'.txt'),'Delimiter','tab');
for iT=1:numTs
step=F*N;
X = Locs((iT-1)*step+1:iT*step,:);
writematrix(X,strcat(FileName,'_',num2str(iT),'.txt'),'Delimiter','tab');
end