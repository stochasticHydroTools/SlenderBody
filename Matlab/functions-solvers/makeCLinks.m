% Function to make the cross links. X = fiber positions (N*nFib x 3 array), 
% nFib = numbers of fibs, N = # of pts per fib, rl = rest length of the
% CLs, rl = rest length of the cross linkers, Lf = length of fibers, nCL =
% number of available CLs, Ld = periodic domain length
% The length of the cross links is obviously going to be less than half the
% periodic domain length - this means that connections can only happen with
% the NEAREST NEIGHBOR
function links = makeCLinks(links,X,nFib,N,rl,Lf,nCL,Ld,g,dt)
    global Periodic
    [nLinks,~]=size(links);
    if (nCL==nLinks)
        return;
    end
    %rcut = rl*1.2;%0.3001;
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmat = (cos((0:N-1).*th));
    % Uniform points on [0, Lf]
    Nuni = 16;
    hu = Lf/(Nuni-1);
    su = (0:Nuni-1)*hu;
    thuni = acos(2*su/Lf-1)';
    Umat = (cos((0:N-1).*thuni));
    % Sampling at those N uniform points
    for iFib=1:nFib
        Xuni((iFib-1)*Nuni+1:iFib*Nuni,:)=Umat*(Lmat \ X((iFib-1)*N+1:iFib*N,:));
    end
    % Figure out which points are already occupied
    added=zeros(1,Nuni*nFib);
    for iL=1:nLinks
        linkinfo=links(iL,:);
        iPt = find(abs(su-linkinfo(2)) < 1e-10)+(linkinfo(1)-1)*Nuni;
        jPt = find(abs(su-linkinfo(4)) < 1e-10)+(linkinfo(3)-1)*Nuni;
        added(iPt)=1;
        added(jPt)=1;
    end
    % Option to load link information from python 
%     pylinks = load('FreeF25C25.txt');
%     iPts = pylinks(2:end,1)+1;
%     jPts = pylinks(2:end,2)+1;
    for i = 1:length(iPts) % loop over POINTS
%         for j = 1:length(jPts)
           iPt = iPts(i);
           jPt = jPts(i);
%     disp('Allowing multiple links!')
%     iPts = randperm(Nuni*nFib);
%     jPts = randperm(Nuni*nFib);
%     for iPt = iPts
%         for jPt = jPts
            % Find the nearest image
            fib1 = floor((iPt-1)/Nuni)+1;
            fib2 = floor((jPt-1)/Nuni)+1;
            alreadyLinked = added(iPt) | added(jPt);
            if (fib1 ~= fib2 && iPt < jPt)% && ~alreadyLinked) % can't be on the same fiber, don't double count
                rvec=Xuni(iPt,:)-Xuni(jPt,:);
                shift = [0 0 0];
                if (Periodic)
                    [rvec, shift] = calcShifted(rvec,g,Ld,Ld,Ld);
                end
                if (norm(rvec) < 1.2*rl && norm(rvec) > 0.8*rl)
                   % Put a CL between the two points
                   % with some probability
                   p1 = mod(iPt-1,Nuni)+1;
                   p2 = mod(jPt-1,Nuni)+1;
                   added(iPt)=1;
                   added(jPt)=1;
                   links = [links; fib1 su(p1) fib2 su(p2) -shift];
                   [nIn, ~]=size(links);
                   if (nIn==nCL) % stop if we've reached the right number
                       return;
                   end
                end
            end
        %end
    end
end



