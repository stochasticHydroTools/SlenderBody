% Compute the force density due to cross-linking. 
% Inputs - the links as an nLink x 4 matrix, where each row looks like 
% (Fib1, s1star, Fib2, s2star), and the configuration X
% X is assumed to be an (nFib x N) x 3 configuration
% N = number of points per fiber, s = arclength coordinates of the points
% on the fiber, w = Chebyshev wts, L=fiber length, K = spring constant,
% rl = rest length of the cross linkers, 
% Outputs - force per length on the fibers. 
function [Clf,X1stars,X2stars] = getCLforce(links,X,Runi,s,w,L, K, rls,Lfacs,g,Ld)
    [nLink,~]=size(links);
    Clf=zeros(size(X));
    X1stars = zeros(nLink,3);
    X2stars = zeros(nLink,3);
    [Nu,N]=size(Runi);
    hu=L/(Nu-1);
    for iL=1:nLink
        rl = rls(iL);
        u1Pt = links(iL,1);
        fib1 = floor((u1Pt-1)/Nu)+1;
        fib1pt = mod(u1Pt,Nu);
        fib1pt = fib1pt+Nu*(fib1pt==0);
        u2Pt = links(iL,2);
        fib2 = floor((u2Pt-1)/Nu)+1;
        fib2pt = mod(u2Pt,Nu);
        fib2pt = fib2pt+Nu*(fib2pt==0);
        shift = links(iL,3:5);
        % Calculate the force density on fiber 1
        inds1 = (fib1-1)*N+1:fib1*N;
        inds2 = (fib2-1)*N+1:fib2*N;
        X1=X(inds1,:);
        X1star = Runi(fib1pt,:)*X1;
        X1stars(iL,:)=X1star;
        s1star = (fib1pt-1)*hu;
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        X2star = Runi(fib2pt,:)*X2;
        X2stars(iL,:)=X2star;
        s2star = (fib2pt-1)*hu;
        renorm = w*deltah(s-s1star,N,L)*w*deltah(s-s2star,N,L)*Lfacs(fib1)*Lfacs(fib2);
        f1 = zeros(N,3);
        for iPt=1:N
            ds = X1(iPt,:)-X2;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s2star,N,L);
            % Integrating over fiber 2 for each pt on fiber 1
            f1(iPt,:)=-K*Lfacs(fib2)*w*ig*deltah(s(iPt)-s1star,N,L);
        end
        % Renormalize f1
        f1=f1./renorm;
        Clf(inds1,:)=Clf(inds1,:)+f1;
         % Calculate the force density on fiber 2
        f2 = zeros(N,3);
        for iPt=1:N
            ds = X2(iPt,:)-X1;
            ig = ds*(1-rl/norm(X1star-X2star));
            ig = ig.*deltah(s-s1star,N,L);
            % Integrating over fiber 1 for each pt on fiber 2
            f2(iPt,:)=-K*Lfacs(fib1)*w*ig*deltah(s(iPt)-s2star,N,L);
        end
        % Renormalize
        f2=f2./renorm;
        Clf(inds2,:)=Clf(inds2,:)+f2;
    end
    % Total force and torque are 0 AFTER ACCOUNTING FOR SHIFT. 
end