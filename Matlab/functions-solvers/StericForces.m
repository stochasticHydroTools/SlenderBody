% Steric forces
function [StericForce,NewPairsAndIntervals] = StericForces(X,Runi,a,kbT,su,nFib,g,Ld)
    [Nu,N]=size(Runi);
    delta = a;
    cutoff = 4*delta;
    F0 = 4*kbT/(delta*a^2)*sqrt(2/pi);
    ds_u = su(2)-su(1);
    StericForce = zeros(nFib*N,3);
    % Uniform point list
    UniPtList = zeros(Nu*nFib,3);
    for iFib=1:nFib
        inds = (iFib-1)*N+1:iFib*N;
        indsu = (iFib-1)*Nu+1:iFib*Nu;
        UniPtList(indsu,:) = Runi*X(inds,:);
    end
    % Neighbor search
    if (Ld > 0)
        NList = load('Neighbors.txt');
        warning('Loading python neighbors!')
        AllPairs=[];
        for iPair=1:length(NList)
            iPt = NList(iPair,1)+1;
            iFib = floor((iPt-1)/Nu)+1;
            inds1 = (iFib-1)*N+1:iFib*N;
            iPtMod = iPt-(iFib-1)*Nu;
            wtiPt = ds_u; 
            if (iPtMod==1 || iPtMod==Nu)
                wtiPt = ds_u/2;
            end
            for jPt=NList(iPair,2)+1
                jFib = floor((jPt-1)/Nu)+1;
                inds2 = (jFib-1)*N+1:jFib*N;
                jPtMod = jPt-(jFib-1)*Nu;
                wtjPt = ds_u; 
                if (jPtMod==1 || jPtMod==Nu)
                    wtjPt = ds_u/2;
                end
                isAway=1;
                if (iFib==jFib && abs(su(jPtMod)-su(iPtMod)) < 1.1*cutoff)
                    isAway=0;
                end
                if (isAway)
                    rvec = UniPtList(iPt,:)-UniPtList(jPt,:);
                    rvecSh = calcShifted(rvec,g,Ld,Ld,Ld);
                    r = norm(rvecSh);
                    if (r < cutoff)
                        ForceMag=F0*exp(-r^2/(2*delta^2));
                        Force = ForceMag*rvecSh/r*wtiPt*wtjPt;
                        R1 = Runi(iPtMod,:);
                        StericForce(inds1,:)=StericForce(inds1,:)+Force.*R1';
                        R2 = Runi(jPtMod,:);
                        StericForce(inds2,:)=StericForce(inds2,:)-Force.*R2';
                        AllPairs=[AllPairs; iFib (iPtMod-1)*ds_u jFib (jPtMod-1)*ds_u];
                    else
                        keyboard
                    end
                end
            end
        end
        % Compile intervals from the list of pairs
        % Take unions where there are multiple different intervals
        iFibjFib=[AllPairs(:,1) AllPairs(:,3)];
        [~,inds]=unique(iFibjFib,'rows');
        NewPairsAndIntervals = [AllPairs(inds,1) AllPairs(inds,2) AllPairs(inds,2) ...
            AllPairs(inds,3) AllPairs(inds,4) AllPairs(inds,4)];
        RepeatInds = setdiff(1:length(AllPairs(:,1)),inds);
        for iInd=RepeatInds
            % Find the row that we already have a domain for
            RowIndex = find(iFibjFib(iInd,1)==NewPairsAndIntervals(:,1) & ...
                iFibjFib(iInd,2)==NewPairsAndIntervals(:,4));
            % Take a union
            NewPairsAndIntervals(RowIndex,2) = ...
                min(NewPairsAndIntervals(RowIndex,2),AllPairs(iInd,2));
            NewPairsAndIntervals(RowIndex,3) = ...
                max(NewPairsAndIntervals(RowIndex,3),AllPairs(iInd,2));
            NewPairsAndIntervals(RowIndex,5) = ...
                min(NewPairsAndIntervals(RowIndex,5),AllPairs(iInd,4));
            NewPairsAndIntervals(RowIndex,6) = ...
                max(NewPairsAndIntervals(RowIndex,6),AllPairs(iInd,4));
        end
    else
        Idx = rangesearch(UniPtList,UniPtList,cutoff);
        for iPt=1:nFib*Nu
            iFib = floor((iPt-1)/Nu)+1;
            inds1 = (iFib-1)*N+1:iFib*N;
            iPtMod = iPt-(iFib-1)*Nu;
            wtiPt = ds_u; 
            if (iPtMod==1 || iPtMod==Nu)
                wtiPt = ds_u/2;
            end
            for jPt=Idx{iPt}
                if (jPt > iPt) % only count once
                    jFib = floor((jPt-1)/Nu)+1;
                    inds2 = (jFib-1)*N+1:jFib*N;
                    jPtMod = jPt-(jFib-1)*Nu;
                    wtjPt = ds_u; 
                    if (jPtMod==1 || jPtMod==Nu)
                        wtjPt = ds_u/2;
                    end
                    isAway=1;
                    if (iFib==jFib && abs(su(jPtMod)-su(iPtMod)) < 1.1*cutoff)
                        isAway=0;
                    end
                    if (isAway)
                        rvec = UniPtList(iPt,:)-UniPtList(jPt,:);
                        r = norm(rvec);
                        if (r < cutoff)
                            ForceMag=F0*exp(-r^2/(2*delta^2));
                            Force = ForceMag*rvec/r*wtiPt*wtjPt;
                            R1 = Runi(iPtMod,:);
                            StericForce(inds1,:)=StericForce(inds1,:)+Force.*R1';
                            R2 = Runi(jPtMod,:);
                            StericForce(inds2,:)=StericForce(inds2,:)-Force.*R2';
                            %AllPairs=[AllPairs; iFib (iPtMod-1)*ds_u jFib (jPtMod-1)*ds_u];
                        else
                            keyboard
                        end
                    end
                end
            end
        end
    end
    