function [links,rCLs] = updateDynamicCLs(links,rCLs,Runi,X,nFib,Ld,KCL,rCL,kbT,kon,koff,dt)
    %% Assumption: link can only form or break once per time step!
    [Nu,N]=size(Runi);
    [nLinks,~]=size(links);
    if (Ld > 0)
        error('This method only works in free space')
    end
    % Form list of potential bindings
    % Uniform point list
    UniPtList = zeros(Nu*nFib,3);
    for iFib=1:nFib
        inds = (iFib-1)*N+1:iFib*N;
        indsu = (iFib-1)*Nu+1:iFib*Nu;
        UniPtList(indsu,:) = Runi*X(inds,:);
    end
    deltaL = 2*sqrt(kbT/KCL);
    cutoff = rCL+deltaL;
    Idx = rangesearch(UniPtList,UniPtList,cutoff);
    NewLinks=[];
    NewRs=[];
    for iPt=1:nFib*Nu
        iFib = floor((iPt-1)/Nu)+1;
        inds1 = (iFib-1)*N+1:iFib*N;
        iPtMod = iPt-(iFib-1)*Nu;
        for jPt=Idx{iPt}
            if (jPt > iPt) % only count once
                jFib = floor((jPt-1)/Nu)+1;
                try
                alreadyExists = find(links(:,1)==iPt & links(:,2)==jPt);
                catch
                alreadyExists = [];
                end
                if (iFib~=jFib && isempty(alreadyExists))
                    inds2 = (jFib-1)*N+1:jFib*N;
                    jPtMod = jPt-(jFib-1)*Nu;
                    rvec = UniPtList(iPt,:)-UniPtList(jPt,:);
                    r = norm(rvec);
                    if (r > rCL-deltaL)
                        NewLinks = [NewLinks; iPt jPt 0 0 0];
                        NewRs = [NewRs;r];
                    end
                end
            end
        end
    end
    % Assign times for new links to form and times for old links to break
    NewRates = kon*ones(length(NewRs),1);
    nNew = length(NewRates);
    NewTimes = -log(1.0-rand(nNew,1))./NewRates;
    OffRates = koff*ones(nLinks,1);
    OffTimes = -log(1.0-rand(nLinks,1))./OffRates;
    Events = [links OffTimes (1:nLinks)'; NewLinks NewTimes (nLinks+1:nLinks+nNew)'];
    Events = Events(Events(:,6) < dt,:);
    [~,inds]=sort(Events(:,6));
    Events = Events(inds,:);
    [nEvents,~]=size(Events);
    linksToBreak = [];
    linksToForm = [];
    for iEv=1:nEvents
        event = Events(iEv,:);
        if (event(7) <=nLinks) % Breaking a link
            linksToBreak=[linksToBreak; event(7)];
        else % Forming a link
           linksToForm=[linksToForm; event(7)-nLinks];
        end
    end
    links(linksToBreak,:)=[];
    rCLs(linksToBreak,:)=[];
    links=[links; NewLinks(linksToForm,1:5)];
    rCLs=[rCLs; rCL*ones(length(linksToForm),1)];
end
