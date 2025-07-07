% The Lagrangian is f(x) + c^T lambda, 
% % Gradient check
nConstr = length(Constr);
x0 = randn(3*Nx,1);
lambda0 = randn(nConstr,1);
deltaX=randn(3*Nx,1);
deltaLam = randn(nConstr,1);
Xin=Xin;
Xp1_0 = ComputeX(x0,Xin,XonNp1Mat,InvXonNp1Mat);
Cons0 = ConsMat*Xp1_0 - Constr;
f0 = 1/2*(Xp1_0-Xin)'*WTilde_Np1*(Xp1_0-Xin) + Cons0'*lambda0;
DXLag0 = XDerivLag(x0,lambda0,Xin,XonNp1Mat,InvXonNp1Mat,...
    WTilde_Np1,ConsMat);
GradEr = deltaX'*DXLag0;

for iP=1:10
x = x0+10^(-iP)*deltaX;
Xp1 =ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
ConsEr = ConsMat*Xp1-Constr;
fNew = 1/2*(Xp1-Xin)'*WTilde_Np1*(Xp1-Xin) + ConsEr'*lambda0;
Er(iP) = (fNew - f0)/10^(-iP);
end
TheEr = abs(Er-GradEr);
semilogy((1:10),TheEr);
% 
% Hessian check
% Compute Hessian at x0
TotalHess = Hessian(x0,lambda0,Xin,InvXonNp1Mat,XonNp1Mat,...
    WTilde_Np1,ConsMat);
Hdelta=TotalHess*deltaX;

for iP=1:10
x = x0+10^(-iP)*deltaX;
DXLag = XDerivLag(x,lambda0,Xin,XonNp1Mat,InvXonNp1Mat,...
    WTilde_Np1,ConsMat);
GradEr(iP) = norm((DXLag-DXLag0)/10^(-iP)-Hdelta);
end
hold on
semilogy(GradEr)

% Now that we checked everything, ready to implement Newton
% Get built in Matlab answer
opts=optimoptions(@lsqnonlin,'OptimalityTolerance',1e-10,...
    'SpecifyObjectiveGradient',true,'Display','off');
tic
fun=@(x)NLMinimizer(x,Xin,XonNp1Mat,InvXonNp1Mat,WTilde_Np1);
ceq=@(x)EqConstr(x,Xin,XonNp1Mat,InvXonNp1Mat,ConsMat,Constr);
[x,resnorm,residual,exitflag,output,lambda,jacobian] = ...
    lsqnonlin(fun,zeros(3*Nx,1),[],[],[],[],[],[],ceq,opts);
toc
xb4=x;
Xb4 = ComputeX(xb4,Xin,XonNp1Mat,InvXonNp1Mat);

x = zeros(3*Nx,1)+0.01*randn(3*Nx,1);
Xp1 = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
nConstr = size(ConsMat,1);
lambda = zeros(nConstr,1)+0.01*randn(nConstr,1);
NormRHS = 1;
NewNorm = 1;
while (NewNorm > 1e-4)
    % Compute function, constraints, Hessian and Jacobian at x 
    JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    DfDx = XDerivLag(x,lambda,Xin,XonNp1Mat,InvXonNp1Mat,...
        WTilde_Np1,ConsMat);
    JacC = ConsMat*JGrad;
    % Compute Hessian at x
    TotalHess = Hessian(x,lambda,Xin,InvXonNp1Mat,XonNp1Mat,...
        WTilde_Np1,ConsMat);
    Xp1 = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    ceq = ConsMat*Xp1-Constr;
    Mat = [TotalHess JacC'; JacC zeros(nConstr)];%+0.1*eye(3*Nx+nConstr);
    %Mat = eye(3*Nx+nConstr);
    Grad = -[DfDx; ceq];
    NormRHS = norm(Grad);
    deltaVars = pinv(Mat)*Grad;
    StepSize = 2;
    NewNorm = inf;
    while (NewNorm>NormRHS)
        StepSize = StepSize/2;
        xtry = x + StepSize*deltaVars(1:3*Nx);
        lamtry = lambda+StepSize*deltaVars(3*Nx+1:end);
        Xp1 = ComputeX(xtry,Xin,XonNp1Mat,InvXonNp1Mat);
        ceq = ConsMat*Xp1-Constr;
        JGrad = GradX(xtry,Xin,XonNp1Mat,InvXonNp1Mat);
        JacC = ConsMat*JGrad;
        DfDx = XDerivLag(xtry,lamtry,Xin,XonNp1Mat,InvXonNp1Mat,...
            WTilde_Np1,ConsMat);
        NewGrad = -[DfDx; ceq];
        NewNorm = norm(NewGrad);
    end
    x=xtry;
    lambda = lamtry;
end


function Xnew = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat)
    XsXMP = InvXonNp1Mat*Xin;
    newXs = rotateTau(reshape(XsXMP(1:end-3),3,[])',...
        reshape(x(1:end-3),3,[])',1);
    Xsp1 = reshape(newXs',[],1);
    NewMP = XsXMP(end-2:end)+x(end-2:end);
    Xnew = XonNp1Mat*[Xsp1;NewMP];
end

function JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat)
    JGrad = zeros(length(Xin));
    XsXMP = InvXonNp1Mat*Xin;
    JGrad(1:end-3,1:end-3) = DrotateTau(reshape(XsXMP(1:end-3),3,[])',...
        reshape(x(1:end-3),3,[])');
    for d=0:2
        JGrad(end-d,end-d)=1;
    end
    JGrad = XonNp1Mat*JGrad;
end

function TotalGradX = XDerivLag(x,lambda,Xin,XonNp1Mat,InvXonNp1Mat,...
    WTilde_Np1,ConsMat)
    JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    ConsGrad = ConsMat*JGrad;
    Xp1 = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    TotalGradX = JGrad'*WTilde_Np1*(Xp1-Xin) + ConsGrad'*lambda;
end

function TotalHess = Hessian(x,lambda,Xin,InvXonNp1Mat,XonNp1Mat,...
  WTilde_Np1,ConsMat)% Compute Hessian at x
    Nx=length(Xin)/3;
    [nConstr,~]=size(ConsMat);
    JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    HessTerm1 = JGrad'*WTilde_Np1*JGrad;
    HessTerm2 = zeros(3*Nx);
    HessEachConstr = zeros(3*Nx,3*Nx,6);
    XsXMP = InvXonNp1Mat*Xin;
    HMaster=RotateHessian(reshape(XsXMP(1:end-3),3,[])',...
        reshape(x(1:end-3),3,[])');
    Xnew = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    Prefactor=(WTilde_Np1*(Xnew-Xin))'*XonNp1Mat;
    PrefacCons = ConsMat*XonNp1Mat;
    for iPt=1:Nx-1
        for d = 1:3
            pEntry = Prefactor(3*(iPt-1)+d);
            LocalHMat = HMaster(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt,d);
            HessTerm2(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt) = ...
                HessTerm2(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt)+LocalHMat*pEntry;
            for iConstr=1:size(ConsMat,1)
                HessEachConstr(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt,iConstr) = ...
                  HessEachConstr(3*(iPt-1)+1:3*iPt,3*(iPt-1)+1:3*iPt,iConstr)...
                    +LocalHMat*PrefacCons(iConstr,3*(iPt-1)+d);
            end
        end
    end
    Hess_X = HessTerm1+HessTerm2;
    Hess_Cons = sum(reshape(lambda,1,1,nConstr).*HessEachConstr,3);
    TotalHess = Hess_X+Hess_Cons;
end

function [val,J] = NLMinimizer(x,Xin,XonNp1Mat,InvXonNp1Mat,WTilde)
    Xp1 = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    val = 1/2*(Xp1-Xin)'*WTilde*(Xp1-Xin);
    if nargout > 1  
        JGrad = GradX(x,Xin,XonNp1Mat,InvXonNp1Mat);
        J = JGrad'*WTilde*(Xp1-Xin);
        J = J';
    end
end

function [c,ceq] = EqConstr(x,Xin,XonNp1Mat,InvXonNp1Mat,ConsMat,Constr)
    Xp1 = ComputeX(x,Xin,XonNp1Mat,InvXonNp1Mat);
    ceq = ConsMat*Xp1-Constr;
    c=[];
end
