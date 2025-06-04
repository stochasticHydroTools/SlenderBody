function [X3new,LfacNew]=PolymerizeBarbed(Lprime,Lfac,L,X,sNp1,bNp1,DNp1,dt)    
    % Add the polymerization velocity and recompute center
    Lextra = dt*Lprime;
    % Extend the tangent vector at s = L outwards
    Xp13 = reshape(X,3,[])';
    BLast =  barymat(L,sNp1,bNp1);
    BFirst = barymat(0,sNp1,bNp1);
    TauLast =BLast*DNp1*Xp13;
    Xadded = TauLast/norm(TauLast)*Lextra+BLast*Xp13;
    % Solve for a new parameterization that ends at Xadded 
    % and goes through the other points
    XWithAdd = [Xp13; Xadded];
    sToEval = sNp1*Lfac;
    LfacNew=Lfac+Lextra/L; % add the extra length
    sToEval=[sToEval;L*LfacNew]; % pts where we evaluate new interp (max is L*Lfacs)
    Rnew = barymat(sToEval/LfacNew,sNp1,bNp1); % For the new parameterization [0,L]
    X3new = pinv(Rnew)*XWithAdd;
    % Aeq = stackMatrix([BFirst*DNp1; BFirst]);
    % beq = Aeq*X;
    % opts1=  optimset('display','off');
    % X3new = lsqlin(stackMatrix(Rnew),reshape(XWithAdd',[],1),...
    %     [],[],Aeq,beq,[],[],[],opts1);
end