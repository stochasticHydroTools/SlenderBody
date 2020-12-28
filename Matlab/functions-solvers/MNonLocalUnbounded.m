% Compute the action of the non-local mobility matrix. Inputs:
% nFib = number of fibers, N = number of points per fiber, s0 = node 
% positions (as a function of arclength) on each fiber, w = quadrature
% weights on each fiber, Lf = length of the fiber, f = force densities on 
% the fibers, X = locations of the fibers, Xs = tangent vectors. 
% f, X , and Xs are n*nFib x 3 arrays. D is the differentiation matrix.
% xi = Ewald parameter, L = periodic domain length
function nLvel = MNonLocalUnbounded(nFib,N,s0,w,Lf,epsilon,f,X,Xs,D,mu,delta)
    global doFP doSpecialQuad;
    % First compute the non-local velocity from the fiber onto itself
    localvel=zeros(N*nFib,3);
    FPintvel = zeros(N*nFib,3);
    for iFib=1:nFib
        inds=(iFib-1)*N+1:iFib*N;
        [Local,Oone] = calcSelf(N,s0,Lf,epsilon,X(inds,:),Xs(inds,:),f(:,inds),D,delta);
        localvel(inds,:)= 1/(8*pi*mu)*Local;
        FPintvel(inds,:) = 1/(8*pi*mu)*Oone;
    end
    
    % Contibutions from the other fibers
    othervel = zeros(N*nFib,3);
    for iFib=1:nFib
        for ipt=1:N
            iPt = N*(iFib-1)+ipt;
            for jFib=1:nFib
                if (jFib~=iFib)
                    inds=(N*(jFib-1))+1:N*jFib;
                    if (doSpecialQuad)
                        [vel2,~]=nearField(X(iPt,:),X(inds,1),X(inds,2),X(inds,3),...
                            f(1,inds)',f(2,inds)',...
                           f(3,inds)',Lf,epsilon,mu,localvel(inds,:)+FPintvel(inds,:),0);
                        othervel(iPt,:)=othervel(iPt,:)+vel2;
                    else
                        dCo = sqrt(exp(3)/24);
                        vel=integrate(X(inds,:),X(iPt,:),f(:,inds),w,dCo*epsilon*Lf);
                        othervel(iPt,:)=othervel(iPt,:)+1/(8*pi*mu)*vel;
                    end
                end
            end
        end
    end
    if (doFP)
        nLvel = FPintvel+othervel;
    else
        nLvel = othervel;
    end
end

function [v,vals] = integrate(X,p,f,w,radius)
    [N,~]=size(X);
    vals=zeros(N,3);
    for iPt=1:N
        R = p-X(iPt,:);
        nR=norm(R);
        R=R/nR;
        vals(iPt,:)=((eye(3)+R'*R)/nR+...
            radius^2*(eye(3)-3*(R'*R))/nR^3)*f(:,iPt);
    end
    v=w*vals;
end