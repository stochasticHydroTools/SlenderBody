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
    th=flipud(((2*(0:N-1)+1)*pi/(2*N))');
    Lmatn = (cos((0:N-1).*th));
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
        th1 = acos(2*s1star/L(fib1)-1)';
        U1 = (cos((0:N-1).*th1));
        X1star = U1*(Lmatn \ X1);
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        th2 = acos(2*s2star/L(fib2)-1)';
        U2 = (cos((0:N-1).*th2));
        X2star = U2*(Lmatn \ X2);
        renorm = w{fib1}*deltah(s{fib1}-s1star,N,L(fib1))*w{fib2}*deltah(s{fib2}-s2star,N,L(fib2));
        for iPt=1:N
            ds = X1(iPt,:)-X2;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s{fib2}-s2star,N,L(fib2));
            f1(iPt,:)=-K*w{fib2}*ig*deltah(s{fib1}(iPt)-s1star,N,L(fib1));
        end
        % Renormalize f1
        f1=f1./renorm;
        Clf(inds1,:)=Clf(inds1,:)+f1;
         % Calculate the force density on fiber 2
        for iPt=1:N
            ds = X2(iPt,:)-X1;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s{fib1}-s1star,N,L(fib1));
            f2(iPt,:)=-K*w{fib1}*ig*deltah(s{fib2}(iPt)-s2star,N,L(fib2));
        end
        % Renormalize
        f2=f2./renorm;
        Clf(inds2,:)=Clf(inds2,:)+f2;
    end
    % Total force and torque are 0 AFTER ACCOUNTING FOR SHIFT. 
end