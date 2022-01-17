% Compute the action of the non-local mobility matrix in an UNBOUNDED DOMAIN. Inputs:
% nFib = number of fibers, N = number of points per fiber, s = node 
% positions (as a function of arclength) on each fiber, b = barycentric weights
% Lf = length of the fiber, f = force densities on 
% the fibers, X = locations of the fibers, Xs = tangent vectors. 
% f, X , and Xs are n*nFib x 3 arrays. D is the differentiation matrix.
% NupsampleforNL = number of points to directly upsample the quadratures
% (this code only supports Ewald with direct upsampling)
% includeFPonRHS = whether to include the finite part integral in this
% velocity
% This code just performs an O(N^2) sum
function nLvel = MNonLocalUnbounded(nFib,N,s,b,L,f,X,Xs,Xss,a,D,mu,includeFPonRHS,Allb,NupsampleforNL)
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
    
    % Contibutions from the other fibers
    othervel = zeros(N*nFib,3);
    % Upsampling matrix
    [sup,wup,~] = chebpts(NupsampleforNL, [0 L],1);
    Rupsample = barymat(sup,s,b);
    for iFib=1:nFib
        for ipt=1:N
            iPt = N*(iFib-1)+ipt;
            for jFib=1:nFib
                inds = (jFib-1)*N+1:jFib*N;
                if (jFib~=iFib)
                    vel=integrate(Rupsample*X(inds,:),X(iPt,:),(Rupsample*f(inds,:))',wup,sqrt(2*a^2/3));
                    othervel(iPt,:)=othervel(iPt,:)+1/(8*pi*mu)*vel;
                end
            end
        end
    end
    nLvel = FPintvel+othervel;
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