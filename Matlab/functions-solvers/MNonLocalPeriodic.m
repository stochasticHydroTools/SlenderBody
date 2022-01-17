% Compute the action of the non-local mobility matrix. Inputs:
% nFib = number of fibers, N = number of points per fiber, s = node 
% positions (as a function of arclength) on each fiber, b = barycentric weights
% Lf = length of the fiber, f = force densities on 
% the fibers, X = locations of the fibers, Xs = tangent vectors. 
% f, X , and Xs are n*nFib x 3 arrays. D is the differentiation matrix.
% splittingparam = Ewald parameter, Ld = periodic domain length, strain = strain on the
% domain (how slanted it is). 
% NupsampleforNL = number of points to directly upsample the quadratures
% (this code only supports Ewald with direct upsampling)
% includeFPonRHS = whether to include the finite part integral in this
% velocity
function nLvel = MNonLocalPeriodic(nFib,N,s,b,L,f,X,Xs,Xss,a,D,mu,includeFPonRHS,Allb,NupsampleforNL,...
    splittingparam,Ld,strain)
    % First compute the non-local velocity from the fiber onto itself
    FPintvel = zeros(N*nFib,3);
    % Compute the local velocity
    if (includeFPonRHS)
        for iFib=1:nFib
            scinds = (iFib-1)*N+1:N*iFib;
            MFP = StokesletFinitePartMatrix(X(scinds,:),Xs(scinds,:),Xss(scinds,:),D,s,L,N,mu,Allb);
            FPintvel(scinds,:) = reshape(MFP*reshape(f(scinds,:)',3*N,1),3,N)';
        end
    end

    % Ewald splitting 
    % Upsampling matrix
    [sup,wup,bup] = chebpts(NupsampleforNL, [0 L],1);
    Rupsample = barymat(sup,s,b);
    Xup = zeros(NupsampleforNL,3);
    fup = zeros(NupsampleforNL,3);
    for iFib=1:nFib
        upinds = (iFib-1)*NupsampleforNL+1:iFib*NupsampleforNL;
        inds = (iFib-1)*N+1:iFib*N;
        Xup(upinds,:)=Rupsample*X(inds,:);
        fup(upinds,:)=Rupsample*f(inds,:);
    end
    fwquad = fup.*(repmat(wup,1,nFib))';
    % Compute the far field and near field
    farvel = EwaldFarVel(Xup,fwquad,mu,Ld,Ld,Ld,splittingparam,a,strain);
    nearvel = EwaldNearSum(NupsampleforNL*nFib,Xup,fwquad,splittingparam,Ld,Ld,Ld,a,mu,strain);
    % Subtract the self term
    RPYvel = farvel+nearvel;
    for iFib=1:nFib
        upinds = (iFib-1)*NupsampleforNL+1:iFib*NupsampleforNL;
        totEwaldi = EwaldTotSum(NupsampleforNL,Xup(upinds,:),fup(upinds,:).*wup',a,mu);
        RPYvel(upinds,:)=RPYvel(upinds,:)-totEwaldi;
    end
    % Downsample it
    othervel = zeros(nFib*N,3);
    Rdownsample = barymat(s,sup,bup);
    for iFib=1:nFib
        upinds = (iFib-1)*NupsampleforNL+1:iFib*NupsampleforNL;
        inds = (iFib-1)*N+1:iFib*N;
        othervel(inds,:)=Rdownsample*RPYvel(upinds,:);
    end
    % Corrections for close fibers - not using right now
    if (false)
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
    % Finally the total velocity
    nLvel = FPintvel+othervel;
end