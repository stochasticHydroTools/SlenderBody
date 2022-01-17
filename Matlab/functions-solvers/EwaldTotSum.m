% Sum total RPY kernel in free space. Used for subtracting out the self
% contribution to the total velocity obtained from Ewald splitting. 
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
    velTot = velTot';
end