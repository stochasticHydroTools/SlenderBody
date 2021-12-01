% Sum total RPY kernel
function velTot = EwaldTotSum(Npts,pts,forces,a,mu)
    velTot=zeros(3,Npts);
    for iPt=1:Npts
        Mself = RPYTot([0 0 0],a,mu);
        velTot(:,iPt)=velTot(:,iPt)+Mself*(forces(iPt,:)');
        for jPt=iPt+1:Npts
            % Subtract free space kernel (no periodicity)
            rvec=pts(iPt,:)-pts(jPt,:);
            Mother = RPYTot(rvec,a,mu);
            velTot(:,iPt)=velTot(:,iPt)+Mother*(forces(jPt,:)');
            velTot(:,jPt)=velTot(:,jPt)+Mother*(forces(iPt,:)');
        end
    end
end