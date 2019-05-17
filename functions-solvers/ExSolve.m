function ut = ExSolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt)   
%     bigMat = [-M K I];     % Dynamic eqn
%     bigMat = [bigMat; Kt zeros(2*N-2,2*N+1)]; % lambdas do no work
%     bigMat = [bigMat; -wIt zeros(3,2*N+1)]; % no net force
%     b = [M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt)); ...
%         zeros(2*N-2,1); wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))];
%     lamalph=lsqminnorm(bigMat,b,1e-6);
%     lambdas=lamalph(1:3*N);
%     alphas=lamalph(3*N+1:5*N-2);
%     Urigid=lamalph(end-2:end);
    % Schur complement solve
    B=[K I];
    C=[Kt; wIt];
    RHS = C*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))+...
         [zeros(2*N-2,1); -wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))];
    alphaU = lsqminnorm(C*M^(-1)*B,RHS);
    alphas=alphaU(1:2*N-2);
    Urigid=alphaU(2*N-1:2*N+1);
    ut = K*alphas+I*Urigid;
end