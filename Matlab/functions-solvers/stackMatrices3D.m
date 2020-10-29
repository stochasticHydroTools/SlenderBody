% Column stack matrices
function [Rs,Ls,Ds,Dinv,D4BC,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf)
    R = barymat(s0, s, b); % Resampling matrix.
    Rs=zeros(3*N,3*(N+4));
    Rs(1:3:3*N,1:3:3*(N+4))=R;
    Rs(2:3:3*N,2:3:3*(N+4))=R;
    Rs(3:3:3*N,3:3:3*(N+4))=R;
    L=[diffmat([2 N+4],2,[0 Lf]); diffmat([2 N+4],3,[0 Lf])];
    Ls=zeros(12,3*(N+4));
    Ls(1:3:12,1:3:3*(N+4))=L;
    Ls(2:3:12,2:3:3*(N+4))=L;
    Ls(3:3:12,3:3:3*(N+4))=L;
    Ds=zeros(3*N);
    Ds(1:3:3*N,1:3:3*N)=diffmat(N,1,[0 Lf],'chebkind1');
    Ds(2:3:3*N,2:3:3*N)=diffmat(N,1,[0 Lf],'chebkind1');
    Ds(3:3:3*N,3:3:3*N)=diffmat(N,1,[0 Lf],'chebkind1');
    D4s=zeros(3*(N+4));
    D4s(1:3:3*(N+4),1:3:3*(N+4))=diffmat(N+4,4,[0 Lf]);
    D4s(2:3:3*(N+4),2:3:3*(N+4))=diffmat(N+4,4,[0 Lf]);
    D4s(3:3:3*(N+4),3:3:3*(N+4))=diffmat(N+4,4,[0 Lf]);
    Dinv=zeros(3*N);
    Dinvone = pinv(diffmat(N,1,[0 Lf],'chebkind1'));
    Dinv(1:3:3*(N),1:3:3*(N))=Dinvone;
    Dinv(2:3:3*(N),2:3:3*(N))=Dinvone;
    Dinv(3:3:3*(N),3:3:3*(N))=Dinvone;
    D4BC = D4s*([Rs;Ls] \ [eye(3*N); zeros(12,3*N)]);
    I=zeros(2*N,2);
    wIt=zeros(2,2*N);
    for iR=1:N
        I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
        wIt(1:3,3*(iR-1)+1:3*iR)=w0(iR)*eye(3);
    end
end
