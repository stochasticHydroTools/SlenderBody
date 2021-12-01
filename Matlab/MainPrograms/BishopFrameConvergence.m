N=100;
SBTRelaxingFib;
refas=Rpl*bishA;
Ns=4:4:80;
for iN=1:length(Ns)
N=Ns(iN);
SBTRelaxingFib;
Allas{iN}=Rpl*bishA;
Allbs{iN}=Rpl*bishB;
end
ersA=zeros(length(Ns)-1,1);
ersB=zeros(length(Ns)-1,1);
erParallel = ersA;
ersPerp = ersA;
for iE=1:length(Ns)
vecersA=Allas{iE}-refas;
vecersParallel = sum(vecersA.*refas,2);
erParallel(iE) = sqrt(wpl*sum(vecersParallel.*vecersParallel,2));
vecersPerp = vecersA-(vecersA.*refas).*refas;
ersPerp(iE) = sqrt(wpl*sum(vecersPerp.*vecersPerp,2));
ersA(iE)=sqrt(wpl*sum(vecersA.*vecersA,2));
end
semilogy(Ns,ersA,'-o')
hold on
semilogy(Ns,erParallel,'-.s')
semilogy(Ns,ersPerp,'--d')


