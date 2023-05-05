% Find roots
function [distance,sFib1,sFib2] = DistanceFiberParts(X1,X2,seg1,s1,seg2,s2,Lseg,sNp1,bNp1,DNp1)
    rts = [s1;s2];
    Fcn = 100*ones(2,1);
    Jac = zeros(2);
    tol = 1e-12;
    nIts=0;
    maxIts=10;
    while (norm(Fcn) > tol && nIts < maxIts)
        nIts=nIts+1;
        Ev1 = barymat(rts(1),sNp1,bNp1);
        Ev2 = barymat(rts(2),sNp1,bNp1);
        X1p = Ev1*X1;
        DX1p = Ev1*DNp1*X1;
        D2X1p = Ev1*DNp1^2*X1;
        X2p = Ev2*X2;
        DX2p = Ev2*DNp1*X2;
        D2X2p = Ev2*DNp1^2*X2;
        Fcn(1) = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
        Fcn(2) = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
        Jac(1,1) = 2*(dot(DX1p,DX1p)+dot(D2X1p,X1p)-dot(D2X1p,X2p));
        Jac(2,1) = -2*dot(DX1p,DX2p);
        Jac(1,2) = Jac(2,1);
        Jac(2,2) = 2*(dot(DX2p,DX2p)+dot(D2X2p,X2p)-dot(D2X2p,X1p));
        rts = rts - Jac \ Fcn;
    end
    if (nIts==maxIts)
        warning('Newton did not converge')
    end
    sFib1=rts(1);
    sFib2=rts(2);
    distance = norm(X1p-X2p);
    seg1end = seg1*Lseg;
    seg1start = seg1end-Lseg;
    seg2end = seg2*Lseg;
    seg2start = seg2end-Lseg;
    if (sFib1 > seg1start && sFib1 < seg1end && sFib2 > seg2start && sFib2 < seg2end)
        return
    end
    % Form list of eight possible other minimums
    % Check s1 = seg1start (1D Newton)
    exteriorDists=[];
    exteriorS1=[];
    exteriorS2=[];
    for s1boundary=[seg1start seg1end]
        Fcn=100;
        s2rt = s2;
        Ev1 = barymat(s1boundary,sNp1,bNp1);
        X1p=Ev1*X1;
        nIts=0;
        while (abs(Fcn) > tol && nIts < maxIts)
            nIts=nIts+1;
            Ev2 = barymat(s2rt,sNp1,bNp1);
            X2p = Ev2*X2;
            DX2p = Ev2*DNp1*X2;
            D2X2p = Ev2*DNp1^2*X2;
            Fcn = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
            DFcn = 2*(dot(DX2p,DX2p)+dot(D2X2p,X2p)-dot(D2X2p,X1p));
            s2rt = s2rt - Fcn/DFcn;
        end
        if (nIts==maxIts)
            warning('Newton did not converge')
        end
        if (s2rt > seg2start && s2rt < seg2end)
            % interior maximum
            exteriorDists = [exteriorDists norm(X1p-X2p)];
            exteriorS1 = [exteriorS1 s1boundary];
            exteriorS2 = [exteriorS2 s2rt];
        end
        for s2boundary=[seg2start seg2end]
            Ev2 = barymat(s2boundary,sNp1,bNp1);
            X2p=Ev2*X2;
            exteriorDists = [exteriorDists norm(X1p-X2p)];
            exteriorS1 = [exteriorS1 s1boundary];
            exteriorS2 = [exteriorS2 s2boundary];
        end
    end
    for s2boundary=[seg2start seg2end]
        Fcn=100;
        s1rt = s1;
        Ev2 = barymat(s2boundary,sNp1,bNp1);
        X2p=Ev2*X2;
        nIts=0;
        while (abs(Fcn) > tol && nIts < maxIts)
            nIts=nIts+1;
            Ev1 = barymat(s1rt,sNp1,bNp1);
            X1p = Ev1*X1;
            DX1p = Ev1*DNp1*X1;
            D2X1p = Ev1*DNp1^2*X1;
            Fcn = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
            DFcn = 2*(dot(DX1p,DX1p)+dot(D2X1p,X1p)-dot(D2X1p,X2p));
            s1rt = s1rt - Fcn/DFcn;
        end
        if (nIts==maxIts)
            warning('Newton did not converge')
        end
        if (s1rt > seg1start && s1rt < seg1end)
            % interior maximum
            exteriorDists = [exteriorDists norm(X1p-X2p)];
            exteriorS1 = [exteriorS1 s1rt];
            exteriorS2 = [exteriorS2 s2boundary];
        end
    end
    [distance,index]=min(exteriorDists);
    sFib1 = exteriorS1(index);
    sFib2 = exteriorS2(index);
end

