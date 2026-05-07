function [Xnew,nIts] = ...
    SolveOptimProblem(Xin,XonNp1Mat,WTilde_Np1,InvXonNp1Mat,ConsMat,Constr,Xt)
    % Now that we checked everything, ready to implement Newton
    TauIn = InvXonNp1Mat*Xin;
    Nx = length(Xin)/3;
    N = Nx - 1;
    nConstr=length(Constr)+N;

    x = TauIn;
    HessEachConstr = ConstraintHessian(N,ConsMat);
    lambda = zeros(nConstr,1);
    maxIts = 20;
    tol = 1e-8;
    GradNorms = zeros(maxIts,1);
    % Compute function, constraints, Hessian and Jacobian at x 
    for nIts=1:maxIts
        JacC = ConstraintJacobian(N,x,XonNp1Mat,ConsMat);
        DfDx = XonNp1Mat'*WTilde_Np1*XonNp1Mat*(x-TauIn) + JacC'*lambda;
        [~,ceq] = EqConstr(x,N,XonNp1Mat,ConsMat,Constr);
        Grad = -[DfDx; ceq];
        GradNorms(nIts)=norm(Grad);
        if (norm(Grad)<tol)
            break
        end
        % Compute Hessian at x
        HessLam = sum(reshape(lambda,1,1,nConstr).*HessEachConstr,3);
        Hxx = XonNp1Mat'*WTilde_Np1*XonNp1Mat + HessLam;
        HMat = [Hxx JacC'; JacC zeros(nConstr)];
        deltaVars = pinv(HMat)*Grad;
        x = x + deltaVars(1:3*Nx);
        lambda = lambda+deltaVars(3*Nx+1:end);
    end

    if (nIts==maxIts)
        % Switch to built-in solve if Newton is slow
        opts=optimoptions(@lsqnonlin,'OptimalityTolerance',1e-10,...
            'SpecifyObjectiveGradient',true,'Display','off');
        fun=@(x)NLMinimizer(x,TauIn,XonNp1Mat,WTilde_Np1);
        ceqfcn=@(x)EqConstr(x,N,XonNp1Mat,ConsMat,Constr);
        [x,resnorm,residual,exitflag,output,lambda,jacobian] = ...
            lsqnonlin(fun,TauIn,[],[],[],[],[],[],ceqfcn,opts);
    end
    Xnew=XonNp1Mat*x;
    %max(abs(Xnew1-Xnew))
end

function [val,J] = NLMinimizer(tau,tauIn,XonNp1Mat,WTilde)
    val = 1/2*(tau-tauIn)'*XonNp1Mat'*WTilde*XonNp1Mat*(tau-tauIn);
    if nargout > 1  
        J = XonNp1Mat'*WTilde*XonNp1Mat*(tau-tauIn);
        J = J';
    end
end

function [c,ceq] = EqConstr(tau,N,XonNp1Mat,ConsMat,Constr)
    ceq = [ConsMat*XonNp1Mat*tau - Constr; ...
        sum(reshape(tau(1:3*N).*tau(1:3*N),3,N))'-1];
    c=[];
end

function JacC = ConstraintJacobian(N,tau,XonNp1Mat,ConsMat)
    % Jacobian of constraint
    JacNormTau = zeros(N,3*(N+1));
    for iPt=1:N
        JacNormTau(iPt,3*(iPt-1)+1:3*iPt)=2*tau(3*(iPt-1)+1:3*iPt);
    end
    JacC = [ConsMat*XonNp1Mat; JacNormTau];
end

function HessEachConstr = ConstraintHessian(N,ConsMat)
    [nConstr,~]=size(ConsMat);
    Nx = N+1;
    % Jacobian of constraint
    HessEachConstr = zeros(3*Nx,3*Nx,nConstr+N);
    for jPt = 1:N
        jC = jPt+nConstr;
        for d = 1:3
            HessEachConstr(3*(jPt-1)+d,3*(jPt-1)+d,jC)=2;
        end
    end
end

% The Lagrangian is f(x) + c^T lambda, 
% % Gradient check
% nConstr = length(Constr) + N;
% tau0 = randn(3*Nx,1);
% lambda0 = randn(nConstr,1);
% deltaTau = randn(3*Nx,1);
% deltaLam = randn(nConstr,1);
% TauIn = [Xst;XTrk];
% Cons0 = [ConsMat*XonNp1Mat*tau0 - Constr; ...
%     sum(reshape(tau0(1:3*N).*tau0(1:3*N),3,N))'-1];
% f0 = 1/2*(tau0-TauIn)'*XonNp1Mat'*WTilde_Np1*XonNp1Mat*(tau0-TauIn) ...
%     + Cons0'*lambda0;
% JacC = ConstraintJacobian(N,tau0,XonNp1Mat,ConsMat);
% DfDx0 = XonNp1Mat'*WTilde_Np1*XonNp1Mat*(tau0-TauIn) + JacC'*lambda0;
% GradEr = deltaTau'*DfDx0;
% 
% % Gradient check
% for iP=1:10
% tau = tau0+10^(-iP)*deltaTau;
% Cons = [ConsMat*XonNp1Mat*tau - Constr; ...
%     sum(reshape(tau(1:3*N).*tau(1:3*N),3,N))'-1];
% f = 1/2*(tau-TauIn)'*XonNp1Mat'*WTilde_Np1*XonNp1Mat*(tau-TauIn) ...
%     + Cons'*lambda0;
% Er(iP) = (f - f0)/10^(-iP);
% end
% TheEr = abs(Er-GradEr);
% semilogy((1:10),TheEr);
% % 
% % Hessian check
% % Compute Hessian at x0
% HessEachConstr = ConstraintHessian(N,ConsMat);
% HessLam = sum(reshape(lambda0,1,1,nConstr).*HessEachConstr,3);
% Hxx = XonNp1Mat'*WTilde_Np1*XonNp1Mat + HessLam;
% Hdelta=Hxx*deltaTau;
% 
% for iP=1:10
% tau = tau0+10^(-iP)*deltaTau;
% JacC = ConstraintJacobian(N,tau,XonNp1Mat,ConsMat);
% DfDx = XonNp1Mat'*WTilde_Np1*XonNp1Mat*(tau-TauIn) + JacC'*lambda0;
% GradEr(iP) = norm((DfDx-DfDx0)/10^(-iP)-Hdelta);
% end
% hold on
% semilogy(GradEr)
