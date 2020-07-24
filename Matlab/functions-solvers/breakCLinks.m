function links = breakCLinks(X,links,nFib,N,rl,Lf,Ld,g,dt)
    if (N < 24)
        sigma = 0.1*Lf;
    elseif (N < 32)
        sigma = 0.07*Lf;
    else
        sigma = 0.05*Lf;
    end
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmat = (cos((0:N-1).*th));
    hu = (Lf-4*sigma)/(N-1);
    % Uniform points on [2*sigma,Lf-2*sigma]
    su = (0:N-1)*hu+2*sigma;
    thuni = acos(2*su/Lf-1)';
    Umat = (cos((0:N-1).*thuni));
    % Sampling at those N uniform points
    for iFib=1:nFib
        Xuni((iFib-1)*N+1:iFib*N,:)=Umat*(Lmat \ X((iFib-1)*N+1:iFib*N,:));
    end
    [nLinks,~]=size(links);
    delinds=[];
    for iL=1:nLinks
        linkinfo=links(iL,:);
        shift = linkinfo(5:7);
        iPt = find(abs(su-linkinfo(2)) < 1e-10)+(linkinfo(1)-1)*N;
        jPt = find(abs(su-linkinfo(4)) < 1e-10)+(linkinfo(3)-1)*N;
        ds = Xuni(iPt,:)-Xuni(jPt,:)+[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        if (norm(ds) > 0.6*rl)
            delinds=[delinds iL];
        end
    end
    links(delinds,:)=[];
end
        
        
        
        
        