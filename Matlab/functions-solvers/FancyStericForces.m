% Steric forces using segment implementation
function [StericForce,NewPairsAndIntervals] = FancyStericForces(X,L,N,Nseg,s,b,a,kbT,nFib,...
                                                                    g,Ld,NPtsPerStd)
    %% Parameters
    delta = a;
    cutoff = 4*delta;
    F0 = 4*kbT/(delta*a^2)*sqrt(2/pi);
    StericForce = zeros(nFib*N,3);
    D = diffmat(N,[0 L]);
    nForced=0;nUnderCurve=0;

    %% Rsample at uniform points (midpoints of segments)
    % This is a quadratic loop in the Matlab code
    Nuniseg=Nseg+1;
    Lseg = L/Nseg;
    if (Lseg <= cutoff)
        error('Segments have to be longer than cutoff!')
    end
    sEPSeg = (0:Nseg)'*Lseg;
    sMPSeg = (1/2:Nseg)'*Lseg;
    REPseg = barymat(sEPSeg,s,b);
    RMPseg = barymat(sMPSeg,s,b);
    EPsegs = zeros(nFib*Nuniseg,3);
    MPsegs = zeros(nFib*Nseg,3);
    for iFib=1:nFib
        EPsegs((iFib-1)*Nuniseg+1:iFib*Nuniseg,:) = REPseg*X((iFib-1)*N+1:iFib*N,:);
        MPsegs((iFib-1)*Nseg+1:iFib*Nseg,:) = RMPseg*X((iFib-1)*N+1:iFib*N,:);
    end
    cutoffLarge = cutoff+Lseg;
    % Neighbor search
    AllPairs=[];
    PeriodicShifts =[];
    for iPt=1:Nseg*nFib
        iFib = floor((iPt-1)/Nseg)+1;
        for jPt=iPt+1:Nseg*nFib
            jFib = floor((iPt-1)/Nseg)+1;
            if (iFib~=jFib || abs(jPt-iPt) > 1)
                rvec = MPsegs(iPt,:)-MPsegs(jPt,:);
                rvecSh = calcShifted(rvec,g,Ld,Ld,Ld);
                r = norm(rvecSh);
                if (r < cutoffLarge)
                   AllPairs=[AllPairs; iPt jPt];
                   PeriodicShifts=[PeriodicShifts; rvec-rvecSh];
                end
            end
        end
    end
    
    %% Use Newton to identify which segments are truly interacting 
    % and then use quadratic approximation to find domain to integrate over
    [nPairs,~] =size(AllPairs);
    AllPairsAndIntervals=[];
    tic
    for iPair=1:nPairs
        iSeg=AllPairs(iPair,1);
        iFib = floor((iSeg-1)/Nseg)+1;
        iSegMod = iSeg-(iFib-1)*Nseg;
        jSeg=AllPairs(iPair,2);
        jFib = floor((jSeg-1)/Nseg)+1;
        jSegMod = jSeg-(jFib-1)*Nseg;
        Seg1Start = EPsegs((iFib-1)*Nuniseg+iSegMod,:);
        Seg1End = EPsegs((iFib-1)*Nuniseg+iSegMod+1,:);
        Seg2Start = EPsegs((jFib-1)*Nuniseg+jSegMod,:)+PeriodicShifts(iPair,:);
        Seg2End = EPsegs((jFib-1)*Nuniseg+jSegMod+1,:)+PeriodicShifts(iPair,:);
        % Closest point based on segments
        [distance,sSeg1,sSeg2] = DistBetween2Segment(Seg1End,Seg1Start,Seg2End,Seg2Start);
        inds1=(iFib-1)*N+1:iFib*N;
        inds2=(jFib-1)*N+1:jFib*N;
        X1 = X(inds1,:);
        X2 = X(inds2,:)+PeriodicShifts(iPair,:);
        Seg1MP = (Seg1Start+Seg1End)/2;
        Seg2MP = (Seg2Start+Seg2End)/2;
        Curv1MP = barymat(0.5*Lseg+(iSegMod-1)*Lseg,s,b)*X1;
        Curv2MP = barymat(0.5*Lseg+(jSegMod-1)*Lseg,s,b)*X2;
        c1 = norm(Curv1MP-Seg1MP);
        c2 = norm(Curv2MP-Seg2MP);
        tol = delta/10;
        if (distance < cutoff+c1+c2)
            nUnderCurve=nUnderCurve+1;
            s1star = sSeg1*Lseg+(iSegMod-1)*Lseg;
            s2star = sSeg2*Lseg+(jSegMod-1)*Lseg;
            [distance,s1star,s2star]=DistanceTwoFibCurves(X1,X2,s1star,s2star,L,s,b,D,tol/10);
            if (distance < cutoff)
                pt1 = barymat(s1star,s,b)*X1;
                pt2 = barymat(s2star,s,b)*X2;
                disp = pt1-pt2;
                tau1 = barymat(s1star,s,b)*D*X1;
                tau2 = barymat(s2star,s,b)*D*X2;
                curv1 = barymat(s1star,s,b)*D^2*X1;
                curv2 = barymat(s2star,s,b)*D^2*X2;
                % a s1^2 + bs2^2 + 2 e s1 s2 + 2 c s1 + 2 d s2 + f
                aa = dot(tau1,tau1)+dot(disp,curv1);
                bb = dot(tau2,tau2)-dot(disp,curv2);
                cc = dot(disp,tau1); % zero unless @ EP
                dd = -dot(disp,tau2); % zero unless @ EP
                ee = -dot(tau1,tau2);
                ff = dot(disp,disp)-cutoff^2;
                s1disc = sqrt((8*dd*ee-8*cc*bb)^2-4*(4*ee^2-4*aa*bb)*(4*dd^2-4*bb*ff));
                if (abs(imag(s1disc)) > 0)
                    s1disc=0;
                end
                s1plus = ((8*cc*bb-8*dd*ee)+s1disc)/(8*ee^2-8*aa*bb);
                s1minus= ((8*cc*bb-8*dd*ee)-s1disc)/(8*ee^2-8*aa*bb);
                Deltas1 = max(abs(s1plus),abs(s1minus));
                s2disc = sqrt((8*cc*ee-8*aa*dd)^2-4*(4*ee^2-4*aa*bb)*(4*cc^2-4*aa*ff));
                if (abs(imag(s2disc)) > 0)
                    s2disc=0;
                end
                s2plus = ((8*dd*aa-8*cc*ee)+s2disc)/(8*ee^2-8*aa*bb);
                s2minus = ((8*dd*aa-8*cc*ee)-s2disc)/(8*ee^2-8*aa*bb);
                %s2plus = 1/(2*bb)*(-2*dd-2*ee*s1plus);
                %s2minus = 1/(2*bb)*(-2*dd-2*ee*s1minus);
                Deltas2 = max(abs(s2minus),abs(s2plus));
                s1Interval = [max(s1star-Deltas1,0) min(s1star+Deltas1,L)];
                s2Interval = [max(s2star-Deltas2,0) min(s2star+Deltas2,L)];
                AllPairsAndIntervals = [AllPairsAndIntervals;...
                    iFib s1Interval jFib s2Interval PeriodicShifts(iPair,:)];
%                 if (iFib==28 && jFib==146)
%                 Rpl=barymat((0:0.001:L)',s,b);
%                 plot3(Rpl*X1(:,1),Rpl*X1(:,2),Rpl*X1(:,3),'LineWidth',1)
%                 hold on
%                 plot3(Rpl*X2(:,1),Rpl*X2(:,2),Rpl*X2(:,3),'LineWidth',1)
%                 Rpl=barymat((s1Interval(1):0.001:s1Interval(2))',s,b);
%                 plot3(Rpl*X1(:,1),Rpl*X1(:,2),Rpl*X1(:,3),'LineWidth',5)
%                 Rpl=barymat((s2Interval(1):0.001:s2Interval(2))',s,b);
%                 plot3(Rpl*X2(:,1),Rpl*X2(:,2),Rpl*X2(:,3),'LineWidth',5)
%                 plot3(pt1(1),pt1(2),pt1(3),'o')
%                 plot3(pt1(1),pt1(2),pt1(3),'o')
%                 keyboard
%                 end
            end
        end
    end
    toc
    
    %% Remove duplicates, and merge intervals for quadrature
    IntArray = floor(AllPairsAndIntervals/tol);
    [~,inds] = unique(IntArray,'rows');
    AllPairsAndIntervals= AllPairsAndIntervals(inds,:);
    % Take unions where there are multiple different intervals
    iFibjFib=[AllPairsAndIntervals(:,1) AllPairsAndIntervals(:,4)];
    [nTotPairs,~]=size(iFibjFib);
    [~,inds] = unique(iFibjFib,'rows');
    NewPairsAndIntervals = AllPairsAndIntervals(inds,:);
    RepeatPairs = iFibjFib(setdiff(1:nTotPairs,inds),:);
    RepeatPairs = unique(RepeatPairs,'rows');
    [nRepeats,~]=size(RepeatPairs);
    for iPair=1:nRepeats
        iFib = RepeatPairs(iPair,1);
        jFib = RepeatPairs(iPair,2);
        % Remove from the unique version (to add back later)
        NewPairsAndIntervals(iFib==NewPairsAndIntervals(:,1) & ...
            jFib==NewPairsAndIntervals(:,4),:)=[];
        % Sort intervals by where they start in x direction (AUTOMATIC)
        IntervalsToSort = AllPairsAndIntervals(iFib==iFibjFib(:,1) ...
            & jFib==iFibjFib(:,2),:);
        [~,inds]=sort(IntervalsToSort(:,2),'ascend');
%         if (max(abs(inds-(1:length(IntervalsToSort(:,2)))')) > 0)
%             keyboard
%         end
        IntervalsToSort=IntervalsToSort(inds,:);
        JoinedIntervals=IntervalsToSort(1,:);
        nJoined=1;
        for iInt=2:length(inds)
            s1start = IntervalsToSort(iInt,2);
            s1end = IntervalsToSort(iInt,3);
            s2start = IntervalsToSort(iInt,5);
            s2end = IntervalsToSort(iInt,6);
            DisjointFromAll=1;
            for jInt=1:nJoined
                Rs1start = JoinedIntervals(jInt,2);
                Rs1end = JoinedIntervals(jInt,3);
                Rs2start = JoinedIntervals(jInt,5);
                Rs2end = JoinedIntervals(jInt,6);
                % Determine if intervals are disjoint
                % We are checking overlap between 
                % [s1start,s1end] x [Rs1start, Rs1end] AND 
                % [s2start, s2end] x [Rs2start, Rs2end]
                if ((Rs1start > s1end || Rs1end < s1start) && ...
                    (Rs2start > s2end || Rs2end < s2start)) 
                else
                    % This merges the two rectangles into one large
                    % rectangle. Because the intervals are always processed
                    % in increasing order of s1, there will be no double
                    % counting
                    DisjointFromAll=0;
                    JoinedIntervals(jInt,2) = min(s1start,Rs1start);
                    JoinedIntervals(jInt,3) = max(s1end,Rs1end);
                    JoinedIntervals(jInt,5) = min(s2start,Rs2start);
                    JoinedIntervals(jInt,6) = max(s2end,Rs2end);
                end
            end
            if (DisjointFromAll)
                JoinedIntervals = [JoinedIntervals; IntervalsToSort(iInt,:)];
                nJoined=nJoined+1;
            end
            % Check that joined intervals are actually disjoint
            for kInt=1:nJoined
                s1start = JoinedIntervals(kInt,2);
                s1end = JoinedIntervals(kInt,3);
                s2start = JoinedIntervals(kInt,5);
                s2end = JoinedIntervals(kInt,6);
                for jInt=iInt+1:nJoined
                    Rs1start = JoinedIntervals(jInt,2);
                    Rs1end = JoinedIntervals(jInt,3);
                    Rs2start = JoinedIntervals(jInt,5);
                    Rs2end = JoinedIntervals(jInt,6);
                    % Determine if intervals are disjoint
                    % We are checking overlap between 
                    % [s1start,s1end] x [Rs1start, Rs1end] AND 
                    % [s2start, s2end] x [Rs2start, Rs2end]
                    if ((Rs1start > s1end || Rs1end < s1start) && ...
                        (Rs2start > s2end || Rs2end < s2start)) 
                    else
                        keyboard
                    end
                end
            end
        end % End join intervals
        NewPairsAndIntervals=[NewPairsAndIntervals;JoinedIntervals];
    end

    %% Quadrature on the intervals
    [nPairs,~]=size(NewPairsAndIntervals);
    for iPair=1:nPairs
        iFib=NewPairsAndIntervals(iPair,1);
        jFib=NewPairsAndIntervals(iPair,4);
        inds1=(iFib-1)*N+1:iFib*N;
        inds2=(jFib-1)*N+1:jFib*N;
        X1 = X(inds1,:);
        X2 = X(inds2,:)+NewPairsAndIntervals(iPair,7:9);
        s1Bounds = NewPairsAndIntervals(iPair,2:3);
        s2Bounds = NewPairsAndIntervals(iPair,5:6);
        NumS1Pts = floor(NPtsPerStd*(s1Bounds(2)-s1Bounds(1))/delta)+1;
        [Seg1QPts,Wts1] = legpts(NumS1Pts,s1Bounds);
        NumS2Pts = floor(NPtsPerStd*(s2Bounds(2)-s2Bounds(1))/delta)+1;
        [Seg2QPts,Wts2] = legpts(NumS2Pts,s2Bounds);
        % Gauss-Legendre grid
        nForced=nForced+1;
        for iQP1=1:length(Seg1QPts)
            siqp = Seg1QPts(iQP1);
            MatRow1 = barymat(siqp,s,b);
            X1qp = MatRow1*X1;
            for iQP2=1:length(Seg2QPts)
                sjqp = Seg2QPts(iQP2);
                MatRow2 = barymat(sjqp,s,b);
                X2qp = MatRow2*X2;
                rvec = X1qp-X2qp;
                r = norm(rvec);
                if (r < cutoff)
                    ForceMag=F0*exp(-r^2/(2*delta^2));
                    Force = ForceMag*rvec/r*Wts1(iQP1)*Wts2(iQP2);
                    StericForce(inds1,:)=StericForce(inds1,:)+Force.*MatRow1';
                    StericForce(inds2,:)=StericForce(inds2,:)-Force.*MatRow2';
                end
            end
        end
    end
end