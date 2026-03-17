function MWsym = LocalDragMob(X,DNp1,const,WInv)
    % Compute tangent vector
    Nx = length(X)/3;
    Xs = DNp1*reshape(X,3,[])';
    Xs = Xs./sqrt(sum(Xs.*Xs,2));
    M = zeros(3*Nx);
    for j = 1:Nx
        M(3*j-2:3*j,3*j-2:3*j)=const*(eye(3)+Xs(j,:)'*Xs(j,:));
        % Mhalf(3*j-2:3*j,3*j-2:3*j)=sqrt(const)*(eye(3)+...
        %     (sqrt(2)-1)*Xs(j,:)'*Xs(j,:));
    end
    MWsym = 1/2*(M*WInv + WInv*M');
end
