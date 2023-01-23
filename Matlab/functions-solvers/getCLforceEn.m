% Compute the force density due to cross-linking. 
% Inputs - the links as an nLink x 4 matrix, where each row looks like 
% (Fib1, s1star, Fib2, s2star), and the configuration X
% X is assumed to be an (nFib x N) x 3 configuration
% N = number of points per fiber, s = arclength coordinates of the points
% on the fiber, w = Chebyshev wts, L=fiber length, K = spring constant,
% rl = rest length of the cross linkers, 
% Outputs - force per length on the fibers. 
function [Clf,X1stars,X2stars] = getCLforceEn(links,X,Runi, K, rls,g,Ld)
    [nLink,~]=size(links);
    Clf=zeros(size(X));
    X1stars = zeros(nLink,3);
    X2stars = zeros(nLink,3);
    [Nu,N]=size(Runi);
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
        X2=X(inds2,:)-[shift(1)+g*shift(2) shift(2) shift(3)]*Ld;
        X2star = Runi(fib2pt,:)*X2;
        X2stars(iL,:)=X2star;
        prefac = (norm(X1star-X2star)-rl)*(X1star-X2star)/norm(X1star-X2star);
        R1 = Runi(fib1pt,:);
        Clf(inds1,:)=Clf(inds1,:)-K*prefac.*R1';
        R2 = Runi(fib2pt,:);
        Clf(inds2,:)=Clf(inds2,:)+K*prefac.*R2';
    end
end