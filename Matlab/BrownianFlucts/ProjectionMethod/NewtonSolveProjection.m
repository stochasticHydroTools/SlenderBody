% Nonlinear system for the projection
% x - xtilde + Mhalf*Chalf'*lambda = 0 
% c(x) = 0
function [xg,it,Allresids] = NewtonSolveProjection(x0,xtilde,...
    GradFcn,cfcn,Mhalf,Chalf,MaxIts,tol,nC)
    xg = x0;
    nX = length(x0);
    lam = zeros(nC,1);
    Allresids = zeros(MaxIts,1);
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
            break
        end
    end
end