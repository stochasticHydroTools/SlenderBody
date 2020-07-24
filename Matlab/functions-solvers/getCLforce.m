% Compute the force density due to cross-linking. 
% Inputs - the links as an nLink x 4 matrix, where each row looks like 
% (Fib1, s1star, Fib2, s2star), and the configuration X
% X is assumed to be an (nFib x N) x 3 configuration
% N = number of points per fiber, s = arclength coordinates of the points
% on the fiber, w = Chebyshev wts, L=fiber length, K = spring constant,
% rl = rest length of the cross linkers, 
% Outputs - force per length on the fibers. 
function Clf = getCLforce(links,X,N,s,w,L, K, rl,g,Ld)
    [nLink,~]=size(links);
    Clf=zeros(size(X));
    % Uniform points on [0, Lf]
    Nu =16;
    hu = L/(Nu-1);
    su = (0:Nu-1)*hu;
    thuni = acos(2*su/L-1)';
    Umat = (cos((0:N-1).*thuni));
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmat = (cos((0:N-1).*th));
    for iL=1:nLink
        fib1 = links(iL,1);
        s1star = links(iL,2);
        fib2 = links(iL,3);
        s2star = links(iL,4);
        shift = links(iL,5:7);
        % Calculate the force density on fiber 1
        inds1 = (fib1-1)*N+1:fib1*N;
        inds2 = (fib2-1)*N+1:fib2*N;
        X1=X(inds1,:);
        X1u = Umat * (Lmat \ X1);
        X1star = X1u(s1star/hu+1,:);
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        X2u = Umat * (Lmat \ X2);
        X2star = X2u(s2star/hu+1,:);
        renorm = w*deltah(s-s1star,N,L)*w*deltah(s-s2star,N,L);
        for iPt=1:N
            ds = X1(iPt,:)-X2;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s2star,N,L);
            f1(iPt,:)=-K*w*ig*deltah(s(iPt)-s1star,N,L);
        end
        % Renormalize f1
        f1=f1./renorm;
        Clf(inds1,:)=Clf(inds1,:)+f1;
         % Calculate the force density on fiber 2
        for iPt=1:N
            ds = X2(iPt,:)-X1;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s1star,N,L);
            f2(iPt,:)=-K*w*ig*deltah(s(iPt)-s2star,N,L);
        end
        % Renormalize
        f2=f2./renorm;
        Clf(inds2,:)=Clf(inds2,:)+f2;
    end
    % Total force and torque are 0 AFTER ACCOUNTING FOR SHIFT. 
end