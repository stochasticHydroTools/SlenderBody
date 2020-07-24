% Ewald near field velocity for RPY
function velNear = EwaldNearSum(Npts,pts,forces,xi,Lx,Ly,Lz,a,mu,g)
    velNear=zeros(3,Npts);
    % Check that the near field decays fast enough
    Mcheck = RPYNear([min([Lx Ly Lz])/2 0 0],xi,a,mu);
    if (max(max(abs(Mcheck))) > 1e-3)
        error("Increase the Ewald parameter or periodic length-interactions with more than 1 image")
    end
    % Compute r_cut (should really be done outside time loop)
    rcuts = 0.01:0.01:(min([Lx Ly Lz])/2);
    nzation = norm(RPYNear([0 0 0],xi,a,mu)*[1;0;0]);
    for iCut=1:length(rcuts)
        if (norm(RPYNear([rcuts(iCut) 0 0],xi,a,mu)*[1;0;0])/nzation < 1e-3)
            rcut = rcuts(iCut);
            break;
        end
    end
    for iPt=1:Npts
        Mself = RPYNear([0 0 0],xi,a,mu);
        velNear(:,iPt)=velNear(:,iPt)+Mself*(forces(iPt,:)');
        for jPt=iPt+1:Npts
            % Find the nearest image
            rvec=pts(iPt,:)-pts(jPt,:);
            rvec = calcShifted(rvec,g,Lx,Ly,Lz);
            if (norm(rvec) < rcut)
                Mother = RPYNear(rvec,xi,a,mu);
                velNear(:,iPt)=velNear(:,iPt)+Mother*(forces(jPt,:)');
                velNear(:,jPt)=velNear(:,jPt)+Mother*(forces(iPt,:)');
            end
        end
    end
end