function stress=getStress(Xt,forces,w,N,nFib,links,nCL,rl,K,g,Ld,s,L)
    % AVERAGE STRESS
    stressInt=zeros(3);
    stressCL = zeros(3);
    interf=0;
    CLf=0;
    intertorq=0;
    CLtorq=0;
    % Forces from fibers
    for iFib=1:nFib
        for iPt=1:N
            inds=(iFib-1)*3*N+3*iPt-2:(iFib-1)*3*N+3*iPt;
            f=forces(inds);
            X=Xt(inds)';
            stressInt=stressInt-w(iPt)*(f*X);
            interf=interf+w(iPt)*f;
            intertorq = intertorq + w(iPt)*cross(f,X);
        end
    end
    % Forces from cross linkers
    [numLinks,~]=size(links);
    for iL=1:numLinks
        fib1 = links(iL,1);
        s1star = links(iL,2);
        fib2 = links(iL,3);
        s2star = links(iL,4);
        shift = links(iL,5:7);
        % Calculate the force density on fiber 1
        inds1 = (fib1-1)*N+1:fib1*N;
        inds2 = (fib2-1)*N+1:fib2*N;
        X=reshape(Xt,3,N*nFib)';
        X1=X(inds1,:);
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        for iPt=1:N
            ds = X1(iPt,:)-X2;
            ig = ds-rl*(ds)./sqrt(sum(ds.*ds,2));
            ig = ig.*deltah(s-s2star,N,L);
            f1=-K*w*ig*deltah(s(iPt)-s1star,N,L);
            stressCL=stressCL-w(iPt)*(f1'*X1(iPt,:));
            CLf=CLf+w(iPt)*f1';
            CLtorq = CLtorq + w(iPt)*cross(f1,X1(iPt,:));
        end
         % Calculate the force density on fiber 2
        for iPt=1:N
            ds = X2(iPt,:)-X1;
            ig = ds-rl*(ds)./sqrt(sum(ds.*ds,2));
            ig = ig.*deltah(s-s1star,N,L);
            f2=-K*w*ig*deltah(s(iPt)-s2star,N,L);
            stressCL=stressCL-w(iPt)*(f2'*X2(iPt,:));
            CLf=CLf+w(iPt)*f2';
            CLtorq = CLtorq + w(iPt)*cross(f2,X2(iPt,:));
        end
    end
    stress = stressCL + stressInt;
    %stress = stress-1/3*trace(stress);
end