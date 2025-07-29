function [x,nIts] = SolveNonLinEqns(x,dt,Xin,XonNp1Mat,InvXonNp1Mat,M,...
    BendForceMat,K,ConsMat,Constr,U0)
    [nConstr,NxThr]=size(ConsMat);
    Nx = NxThr/3;
    SolAll=1;
    nIts=0;
    StepSize=1;
    x0=x;
    % Compute basis of motions
    Lambda = x(1:3*Nx);
    OmegaV = x(3*Nx+1:6*Nx);
    MotionBasis =  calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat);    
    Gamma = x(6*Nx+1:6*Nx+nConstr);
    NewX = ComputeX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
    Eq1 = (NewX-Xin)/dt - M*(BendForceMat*NewX+Lambda) - U0;
    Eq2 = MotionBasis'*Lambda+MotionBasis'*ConsMat'*Gamma;
    Eq3 = ConsMat*NewX-Constr;
    SolAll=[Eq1;Eq2;Eq3];
    while (norm(SolAll)>1e-6 && nIts < 100 && StepSize > 1e-6)
        % Compute gradient
        JGrad = GradX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        dXdOm = JGrad*dt;  
        GradMotDotLamGam = gradMotionBasis(OmegaV,Xin,dt,JGrad,Lambda+ConsMat'*Gamma,...
            XonNp1Mat,InvXonNp1Mat);
        GradMat = [-M (eye(3*Nx)/dt-M*BendForceMat)*dXdOm zeros(3*Nx,nConstr); ...
            MotionBasis' GradMotDotLamGam MotionBasis'*ConsMat'; ...
            zeros(nConstr,3*Nx) ConsMat*dXdOm zeros(nConstr)];
        % Gradient check
        dx = randn(6*Nx+nConstr,1);
        ExpDiff = GradMat*dx;
        for dp=1:10
            newx = x+10^(-dp)*dx;
            Lambdac = newx(1:3*Nx);
            OmegaVc = newx(3*Nx+1:6*Nx);
            Gammac = newx(6*Nx+1:end);
            NewXc = ComputeX(OmegaVc*dt,Xin,XonNp1Mat,InvXonNp1Mat);
            NewMotionBasis = calcMotionBasis(OmegaVc,dt,Xin,XonNp1Mat,InvXonNp1Mat);
            Eq1 = (NewXc-Xin)/dt - M*(BendForceMat*NewXc+Lambdac) - U0;
            Eq2 = NewMotionBasis'*Lambdac+NewMotionBasis'*ConsMat'*Gammac;
            Eq3 = ConsMat*NewXc-Constr;
            SolAlld(:,dp)=([Eq1;Eq2;Eq3]-SolAll)/(10^(-dp));
        end
        % Newton solve
        NewDir = -pinv(GradMat)*SolAll;
        % Line search
        StepSize=1;
        Normnew = inf;
        while (Normnew>norm(SolAll))
            xtry = x+StepSize*NewDir;
            Lambda = xtry(1:3*Nx);
            OmegaV = xtry(3*Nx+1:6*Nx);
            Gamma = xtry(6*Nx+1:6*Nx+nConstr);
            MotionBasis = calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat); 
            NewX = ComputeX(OmegaV*dt,Xin,XonNp1Mat,InvXonNp1Mat);
            Eq1 = (NewX-Xin)/dt - M*(BendForceMat*NewX+Lambda) - U0;
            Eq2 = MotionBasis'*Lambda+MotionBasis'*ConsMat'*Gamma;
            Eq3 = ConsMat*NewX-Constr;
            Normnew=norm([Eq1;Eq2;Eq3]);
            StepSize=StepSize/2;
        end
        SolAll=[Eq1;Eq2;Eq3];
        nIts=nIts+1;
        x=xtry;
    end
    %max(abs(x(1:3*Nx)-x0(1:3*Nx)))
    %nIts
    %norm(SolAll)
    if (norm(SolAll) > 1e-6)
        keyboard
    end
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

function MotionBasis = calcMotionBasis(OmegaV,dt,Xin,XonNp1Mat,InvXonNp1Mat)
    Nx = length(Xin)/3;
    Tau = InvXonNp1Mat*Xin;
    Tau = reshape(Tau(1:3*(Nx-1)),3,[])';
    Omega = reshape(OmegaV(1:3*(Nx-1)),3,[])';
    OmegaPerp = cross(Tau,Omega);
    MotionBasis=zeros(3*Nx,2*Nx+1);
    for j=1:Nx-1
        alpha=zeros(3*Nx,1);
        alpha((j-1)*3+1:3*j) = Omega(j,:);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,j)=(NewX-Xin)/dt;
    end
    for j=1:Nx-1
        alpha=zeros(3*Nx,1);
        alpha((j-1)*3+1:3*j) = OmegaPerp(j,:);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,j+Nx-1)=(NewX-Xin)/dt;
    end
    for d=1:3
        alpha=zeros(3*Nx,1);
        alpha(3*(Nx-1)+d)=OmegaV(3*(Nx-1)+d);
        NewX = ComputeX(alpha*dt,Xin,XonNp1Mat,InvXonNp1Mat);
        MotionBasis(:,2*(Nx-1)+d)=(NewX-Xin)/dt;
    end
end

function GradMotDotLam = gradMotionBasis(OmegaV,Xin,dt,JGrad,Lambda,...
        XonNp1Mat,InvXonNp1Mat)
    Nx = length(Xin)/3;
    Tau = InvXonNp1Mat*Xin;
    Tau = reshape(Tau(1:3*(Nx-1)),3,[])';
    Omega = reshape(OmegaV(1:3*(Nx-1)),3,[])';
    OmegaPerp = cross(Tau,Omega);
    OmegaVPerp = OmegaV;
    OmegaVPerp(1:3*(Nx-1))=reshape(OmegaPerp',[],1);
    JGradPerp = GradX(OmegaVPerp*dt,Xin,XonNp1Mat,InvXonNp1Mat);
    GradMotDotLam = zeros(2*Nx+1,3*Nx);
    for p = 1:Nx-1
        GradMotDotLam(p,(p-1)*3+1:3*p)=JGrad(:,(p-1)*3+1:3*p)'*Lambda;
        GradMotDotLam(p+Nx-1,(p-1)*3+1:3*p)=...
            (JGradPerp(:,(p-1)*3+1:3*p)*CPMatrix(Tau(p,:)))'*Lambda;
    end
    for d=1:3
        GradMotDotLam(2*(Nx-1)+d,(Nx-1)*3+d)=JGrad(:,(Nx-1)*3+d)'*Lambda;
    end
end