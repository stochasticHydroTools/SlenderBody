function [xg,newtits,newtfail,matfail] = ...
    AdvanceProjection(x,dt,kbT,EMat,Mobility,CMat,cfcn,...
    MaxIts,tol,nC,implicit,opts)
    nX = length(x);
    M = Mobility(x);
    Mhalf = chol(M)';
    nW = 1;
    delta = 1e-5; % RFD
    
    divM = zeros(nX,1);
    for iP=1:nW
        w1 = randn(nX,1);
        xr = x + delta*w1;
        Mu = Mobility(xr);
        divM = divM + 1/(nW*delta)*(Mu-M)*w1;
    end
    divMtru = divM;

    % Take unconstrained step 
    W = randn(nX,1);
    ExPart = dt*kbT*divMtru+sqrt(2*dt*kbT)*Mhalf*W;
    if (implicit)
        xtilde = (eye(nX)+dt*M*EMat) \ (x+ExPart);
    else
        xtildeEx = x - dt*M*EMat*x + ExPart;
    end

    % Half step
    xHalf = x + sqrt(kbT*dt/2)*Mhalf*W;
    Mhalf = Mobility(xHalf);
    Chalf = CMat(xHalf);

    % Nonlinear system for the projection
    % x - xtilde + Mhalf*Chalf'*lambda = 0 
    % c(x) = 0
    % Newton solve
    [xg,newtits,newtfail] = NewtonSolveProjection(xtilde,xtilde,CMat,cfcn,...
        Mhalf,Chalf,MaxIts,tol,nC);
    matfail=0;
    if (newtfail)
        % Matlab default
        NLFcn = @(x) NonLinSys(x,xtilde,Mhalf,Chalf,CMat,cfcn);
        [xnlsolve,~,exitflag] = fsolve(NLFcn,[x;zeros(nC,1)],opts);
        xg = xnlsolve(1:nX);
        if (exitflag<=0)
            matfail=1;
        end
    end
end

% Nonlinear system for the projection
% x - xtilde + Mhalf*Chalf'*lambda = 0 
% c(x) = 0
function [xg,it,fail] = NewtonSolveProjection(x0,xtilde,...
    GradFcn,cfcn,Mhalf,Chalf,MaxIts,tol,nC)
    xg = x0;
    nX = length(x0);
    lam = zeros(nC,1);
    Allresids = zeros(MaxIts,1);
    fail=1;
    for it=1:MaxIts
        % Compute the gradient and Hessian at x
        C = GradFcn(xg);
        J = [eye(nX) -Mhalf*Chalf'; C zeros(nC)];
        ceqc=cfcn(xg);
        resid = [(xg-xtilde) - Mhalf *Chalf'*lam;ceqc ];
        er=norm(resid);
        Allresids(it)=er;
        if (er > tol)
            newsol = [xg;lam] - J \ resid;
            xg = newsol(1:nX);
            lam = newsol(nX+1:end);
        else
            fail=0;
            break;
        end
    end
end

function [val,J] = NonLinSys(xin,xtilde,Mhalf,Chalf,CMat,cfcn)
    nX = length(xtilde);
    x = xin(1:nX);
    lam = xin(nX+1:end);
    val = [(x-xtilde) - Mhalf *Chalf'*lam; cfcn(x)];
    C = CMat(x);
    J = [eye(length(x)) -Mhalf*Chalf'; C zeros(length(lam))];
end