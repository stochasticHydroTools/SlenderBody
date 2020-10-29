% Compute the action of the non-local mobility matrix. Inputs:
% nFib = number of fibers, N = number of points per fiber, s0 = node 
% positions (as a function of arclength) on each fiber, w = quadrature
% weights on each fiber, Lf = length of the fiber, f = force densities on 
% the fibers, X = locations of the fibers, Xs = tangent vectors. 
% f, X , and Xs are n*nFib x 3 arrays. D is the differentiation matrix.
% xi = Ewald parameter, L = periodic domain length, g = strain on the
% domain (how slanted it is). 
function nLvel = MNonLocalPeriodic(nFib,N,s0,w,Lf,epsilon,f,X,Xs,D,mu,xi,Lx,Ly,Lz,g)
    global doSpecialQuad doFP;
    % First compute the self velocity for each fiber
    localvel=zeros(N*nFib,3);
    FPintvel = zeros(N*nFib,3);
    for iFib=1:nFib
        inds=(iFib-1)*N+1:iFib*N;
        [Local,Oone] = calcSelf(N,s0,Lf,epsilon,X(inds,:),Xs(inds,:),f(:,inds),D);
        localvel(inds,:)= 1/(8*pi*mu)*Local;
        FPintvel(inds,:) = 1/(8*pi*mu)*Oone;
    end

    % Ewald splitting 
    aRPY=exp(1.5)/4*epsilon*Lf;
    fwquad = f.*repmat(w,1,nFib);
    % Compute the far field and near field
    farvel = EwaldFarVel(X,fwquad',mu,Lx,Ly,Lz,xi,aRPY,g)';
    nearvel = EwaldNearSum(N*nFib,X,fwquad',xi,Lx,Ly,Lz,aRPY,mu,g)';
    % Subtract the self term
    RPYvel = farvel+nearvel;
    for iFib=1:nFib
        inds=(iFib-1)*N+1:iFib*N;
        totEwaldi = EwaldTotSum(N,X(inds,:),(f(:,inds).*w)',aRPY,mu)';
        RPYvel(inds,:)=RPYvel(inds,:)-totEwaldi;
    end
    % Corrections for close fibers
    if (doSpecialQuad)
        for iFib=1:nFib
            for ipt=1:N
                iPt = N*(iFib-1)+ipt;
                for jFib=1:nFib
                    if (jFib~=iFib)
                        inds=(N*(jFib-1))+1:N*jFib;
                        % Mod so target is close
                        rvec=X(iPt,:)-X(inds,:);
                        [rvec,tshift1]=calcShifted(rvec,g,Lx,Ly,Lz);
                        [~,ind]=min(sum(rvec.*rvec,2));
                        tshift2=tshift1(ind,1)*[Lx 0 0]+tshift1(ind,2)*[g*Ly Ly 0]+...
                                            tshift1(ind,3)*[0 0 Lz];
                        [cvel,qtype]=nearField(X(iPt,:)-tshift2,X(inds,1),X(inds,2),X(inds,3),...
                                               f(1,inds)',f(2,inds)',f(3,inds)',Lf,...
                                               epsilon,mu,localvel(inds,:)+FPintvel(inds,:),1);
                        qtypes((iFib-1)*N+ipt,jFib)=qtype;
                        RPYvel(iPt,:)=RPYvel(iPt,:)+cvel;
                    end
                end
            end
        end
    end
%     % Finally the total velocity
    if (doFP)
        nLvel = FPintvel+RPYvel;
    else
        nLvel = RPYvel;
    end
end

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