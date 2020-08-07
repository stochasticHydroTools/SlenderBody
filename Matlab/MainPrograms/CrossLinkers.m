% % Making cross linkers
% rng(0);
% Lf = 2;
% sigma = 0.05*Lf;
% K = 1;
% rl=0.5;
% N = 32;
% Nup = 1000;
% [s,~,b]=chebpts(N,[0 Lf],1);
% X1=[s zeros(N,2)];
%     X2=[s 0.2*ones(N,1) zeros(N,1)];
% X=[X1; X2];
% nFib=2;
% nCL = 2;
% % Make links
% links = makeCLinks(X,nFib,N,rl,Lf,nCL);
% % Compute force from those links
% % Plot the uniform points and cross links
% n=N;
% th=flipud(((2*(0:n-1)+1)*pi/(2*n))');
% Lmat = (cos((0:n-1).*th));
% su = 0:Lf/(N-1):Lf;
% thuni = acos(2*su/Lf-1)';
% Umat = (cos((0:n-1).*thuni));
% % Sampling at N uniform points
% for iFib=1:nFib
%     Xuni((iFib-1)*N+1:iFib*N,:)=Umat*(Lmat \ X((iFib-1)*N+1:iFib*N,:));
% end
% linkinds = links;
% linkinds(:,2)=floor(links(:,2)*(N-1)/Lf+1e-5);
% linkinds(:,4)=floor(links(:,4)*(N-1)/Lf+1e-5);
% linkstarts = Xuni((linkinds(:,1)-1)*N+linkinds(:,2),:);
% linkends = Xuni((linkinds(:,3)-1)*N+linkinds(:,4),:);
% hold on
% for iFib=1:nFib
%     finds=(iFib-1)*N+1:iFib*N;
%     plot3(Xuni(finds,1),Xuni(finds,2),Xuni(finds,3),'-o')
% end
% for iCL=1:nCL
%     plot3([linkstarts(iCL,1) linkends(iCL,1)], [linkstarts(iCL,2) linkends(iCL,2)],...
%           [linkstarts(iCL,3) linkends(iCL,3)],'--k')
% end

% Some tests on the force functions for cross-linking
rng(0);
Lf = 2;
K = 1;
rl = 0.25;
Nless = 24;
Nup = 1000;
[s,wdwn,b]=chebpts(Nless,[0 Lf],1);
[sup,wup,bup]=chebpts(Nup,[0 Lf],1);
B=barymat(sup,s,b);
Bdwn = barymat(s,sup,bup);
N=Nless;
for iN=1:2
    if (iN==2) 
        N=Nup;
        fless = Clf;
    end
    [s,w,b]=chebpts(N,[0 Lf],1);
    X1=[s zeros(N,1) zeros(N,1)];
    X1=[cos(s) sin(s) zeros(N,1)];
    X2=[s ones(N,1) zeros(N,1)];
    % Suppose the cross linker is at the center
    s1star=14/15;
    s2star=16/15;
    % Calculate the force densities
    links = [1 s1star 2 s2star 0 0 0];
    Clf = getCLforce(links,[X1;X2],N,s,w,Lf, K, rl,0,0);
end
% Compute L^2 error in the force
erf=B*fless(1:Nless,:)-Clf(1:Nup,:);
erf=sqrt(wup*sum(erf.*erf,2))/max(sqrt(sum(Clf.*Clf,2)))
fdwn = [Bdwn*Clf(1:Nup,:); Bdwn*Clf(Nup+1:2*Nup,:)];

