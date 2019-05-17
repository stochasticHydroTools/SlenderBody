% Backward Euler
function Xp1 = BESolve(N,M,K,Kt,I,wIt,FE,LRLs,URLs,Xt,dt)
%     bigMat = [-M K-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
%         I-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
%     bigMat = [bigMat; Kt zeros(2*N-2) zeros(2*N-2,3)];
%     bigMat = [bigMat; wIt dt*wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
%         wIt*dt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
%     b = [M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt)); ...
%         zeros(2*N-2,1); -wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))];
%     lamalph=lsqminnorm(bigMat,b,1e-6);
%     lambdas=lamalph(1:3*N);
%     alphas=lamalph(3*N+1:5*N-2);
%     Urigid=lamalph(end-2:end);
    % Schur complement solve
    B=[K-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
        I-dt*M*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
    C=[Kt; wIt];
    D=[zeros(2*N-2) zeros(2*N-2,3); ...
        dt*wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*K)) ...
        wIt*dt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*I))];
    RHS = C*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))+...
         [zeros(2*N-2,1); -wIt*FE*(URLs \ (LRLs \ [eye(3*N); zeros(12,3*N)]*Xt))];
    alphaU = lsqminnorm(C*M^(-1)*B+D,RHS);
    alphas=alphaU(1:2*N-2);
    Urigid=alphaU(2*N-1:2*N+1);
    ut = K*alphas+I*Urigid;
    Xp1 = Xt+dt*ut;
end