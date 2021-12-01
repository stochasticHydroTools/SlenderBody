function K = KMatBlobs(ds,Xs)
    [N,~]=size(Xs);
    K = zeros(3*N,2*N+3);
    [theta,phi,~] = cart2sph(Xs(:,1),Xs(:,2),Xs(:,3));
    theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
    n1 = [-sin(theta) cos(theta) 0*theta];
    n2 = [-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
    for iD=1:3
        K(iD:3:end,iD)=1; % constant mode
    end
    for iCol=1:N
        K(3*iCol-2:end,3+iCol) = repmat(n1(iCol,:)',N-iCol+1,1)*ds; % n1's 
        K(3*iCol-2:3*iCol,3+iCol) = K(3*iCol-2:3*iCol,3+iCol)*1/2;
        K(3*iCol-2:end,3+N+iCol) = repmat(n2(iCol,:)',N-iCol+1,1)*ds; % n2's 
        K(3*iCol-2:3*iCol,3+N+iCol) = K(3*iCol-2:3*iCol,3+N+iCol)*1/2;
    end
end
