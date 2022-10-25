% Find the optimal theta for each N
%clear;
aas=10;
linestys=["-.o","-s","--d",":o","-o","-o","-o","-o","-o","-o"];
rotrot=0;
rottrans=1;
transtrans=0;
%aas = 40;
for Cindex=[1 2 6 10]
aa=aas;%(Cindex);
if (rottrans)
    if (aa==40)
        Ns1 = 120:80:520;
        Nthets = 12:4:32;
    elseif (aa==20)
        Ns1 = 400:100:800;
        Nthets = 16:4:32;
    elseif (aa==80)
        Ns1 = 40:40:200;
        Nthets= 16:4:32;
    elseif (aa==10)
        Ns1 = 800:200:1600;
        Nthets = 16:4:32;
    end
elseif (rotrot)
    if (aa==80)
        Ns1 = round(25:12.5:100);%50:30:140;
        Nthets=8:4:32;
    elseif (aa==40)
        Ns1 = 50:25:200;
        Nthets = 8:4:32;
    elseif (aa==20)
        Ns1 = 100:50:400;
        Nthets = 8:4:32;
    elseif (aa==10)
        Ns1 = 200:100:800;
        Nthets = 8:4:32;
    end
elseif (transtrans)
    if (aa==80)
        Ns1 = 80:40:240;
        Nthets=12:4:28;
    elseif (aa==40)
        Ns1 = 160:80:560;
        Nthets = 12:4:32;
    elseif (aa==20)
        Ns1 = [480:160:1280];
        Nthets = [12:4:32];
    end
end
NtAll=Ns1.*Nthets;
for jjN=1:length(Ns1)
    Nch = Ns1(jjN);
    iiT= Nthets(jjN);
    if (rottrans)
        load(strcat('FloreGeo_a',num2str(aa),'Nt',num2str(iiT),'.mat'))
    elseif (rotrot)
        load(strcat('RotResults_a',num2str(aa),'Nt',num2str(iiT),'.mat'))
    elseif (transtrans)
        load(strcat('TransResults_a',num2str(aa),'Nt',num2str(iiT),'.mat'))
    end
    [~,ind]=find(Ns==Nch);
    ind=ind(1);
    fRef{jjN}=FToCompare{ind};
    if (transtrans || rottrans)
        if (rottrans)
            AllUErs(jjN,Cindex)=UEr(ind,Cindex);
        else
            AllUErs(jjN,Cindex)=UEr(ind);
        end
    end
    if (rotrot || rottrans)
        if (rottrans)
            AllOmErs(jjN,Cindex)=OmEr(ind,Cindex);
        else
            AllOmErs(jjN,Cindex)=OmEr(ind);
        end
    end
    nRef{jjN}=NToCompare{ind};
end
clear fer ner;
for j=1:length(Ns1)-1
    fer(j,1)=sqrt(wc*sum((fRef{j}-fRef{j+1}).*(fRef{j}-fRef{j+1}),2));
    ner(j,1)=sqrt(wc*sum((nRef{j}-nRef{j+1}).*(nRef{j}-nRef{j+1}),2));
end
fNorm = sqrt(wc*sum(fRef{length(Ns1)}.*fRef{length(Ns1)},2));
nNorm = sqrt(wc*(nRef{length(Ns1)}.*nRef{length(Ns1)}));
set(gca,'ColorOrderIndex',Cindex)
if (~rotrot)
figure(1)
loglog(NtAll(1:end-1)*aa*1e-3/2,fer/fNorm,linestys(Cindex))
hold on
xlabel('$\epsilon N_t$')
ylabel('$f$ error','interpreter','latex')
figure(2)
loglog(NtAll(1:end)*aa*1e-3/2,AllUErs(1:length(NtAll),Cindex),linestys(Cindex))
hold on
xlabel('$\epsilon N_t$')
ylabel('$||U_\textrm{SB}-U||_{L^2}$')
end
if (~transtrans)
figure(3)
set(gca,'ColorOrderIndex',Cindex)
loglog(NtAll(1:end-1)*aa*1e-3/2,ner/nNorm,linestys(Cindex))
hold on
xlabel('$\epsilon N_t$')
ylabel('$n^\parallel$ error','interpreter','latex')
end
if (~transtrans)
figure(4)
set(gca,'ColorOrderIndex',Cindex)
loglog(NtAll(1:end)*aa*1e-3/2,AllOmErs(1:length(NtAll),Cindex),linestys(Cindex))
hold on
xlabel('$\epsilon N_t$')
ylabel('$a^2||\Psi^\parallel_\textrm{SB}-\Psi^\parallel||_{L^2}$')
end
%semilogy(Nthets,ner(2,:),'-.s')

%plot((8:4:24),10./(8:4:24).^2,':k')
%plot((16:4:24),10./(16:4:24).^2,':k')
%legend('$f, N=80$','$n^\parallel, N=80$','$f, N=150$','$n^\parallel, N=150$','2nd order')
end