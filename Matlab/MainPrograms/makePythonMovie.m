close all;
clear vels fibmaxes meanstrains meancurves
f=figure;
movieframes=getframe(f);
nFib=700;
N=16;
Ld=4;
step=N*nFib;
rl=0.5;
Lf = 2.0;
for nCL=[8400]
% Locs=load(strcat('RelaxingOm0.1LocsF',num2str(nFib),'C',num2str(nCL),'.txt'));
% strains=load(strcat('RelaxingLinkStrainsOm0.1F',num2str(nFib),'C',num2str(nCL),'.txt'));
% curves=load(strcat('RelaxingFibCurvesOm0.1F',num2str(nFib),'C',num2str(nCL),'.txt'));
s0=chebpts(N,[0 Lf],1);
Locs=LocalLocs;
links = makeCLinks([],Locs(1:step,:),nFib,N,rl,Lf,nCL,Ld,0);
%dt = 2000/(length(Locs)/step-1);
dt = 5e-2;
allvels = (Locs(1:end-step,:)-Locs(step+1:end,:))/dt;
[s,w]=chebpts(N,[0 Lf],1);
XtB4=zeros(step*3,1);
OmHz = 1;
for iT=0:80
    clf;
    t=dt*iT;
    gn=0.1*sin(OmHz*2*pi*t);
    L = [1 -gn 0; 0 1 0; 0 0 1];
    thisv=allvels((1:step)+step*iT,:);
    absvs = sum(thisv.*thisv,2);
    for iFib=1:nFib
        vFibs(iFib)=sqrt(w*absvs((iFib-1)*N+1:iFib*N));
    end
    [maxvels(iT+1),ind]=max(vFibs);
    meanvels(iT+1) = mean(vFibs);
    fibmaxes(iT+1)=ind;
    Xt = reshape(LocalLocs((1:step)+step*iT,:)',3*step,1);
    thk=1.0;
    makePlot;
%     Xt = reshape(HydroLocs((1:step)+step*iT,:)',3*step,1);
%     thk=1.0;
%     makePlot;
    movieframes(length(movieframes)+1)=getframe(f);
    % Primed coords
%     gn=0;
%     Xt = reshape(L * Locs((1:step)+step*iT,:)',3*step,1);
%     primedvels(iT+1)=max(abs(XtB4-Xt)/5e-3);
%     XtB4=Xt;
end
% makePlot;
% for iT=0:length(strains)/nCL-1
%     meanstrains(iT+1) = mean(abs(strains((1:nCL)+nCL*iT)));
%     maxstrains(iT+1) =  max(abs((strains((1:nCL)+nCL*iT))));
%     meancurves(iT+1) = mean(curves((1:nFib)+nFib*iT));
%     maxcurves(iT+1) = max(curves((1:nFib)+nFib*iT));
% end
% movieframes=movieframes(2:end);
% eval(strcat('LocsF',num2str(nFib),'C',num2str(nCL),'=Locs;'))
% eval(strcat('maxvels',num2str(nFib),'C',num2str(nCL),'=maxvels;'))
% eval(strcat('meanstrains',num2str(nFib),'C',num2str(nCL),'=meanstrains;'))
% eval(strcat('meancurves',num2str(nFib),'C',num2str(nCL),'=meancurves;'))
% eval(strcat('meanvels',num2str(nFib),'C',num2str(nCL),'=meanvels;'))
end
