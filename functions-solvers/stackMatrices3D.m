function [Rs,Ls,Ds,D4s,Dinv,LRLs,URLs,chebyshevmat,I,wIt]=stackMatrices3D(s0,w0,s,b,N,Lf)
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
    Dinv(1:3:3*(N),1:3:3*(N))=pinv(diffmat(N,1,[0 Lf],'chebkind1'));
    Dinv(2:3:3*(N),2:3:3*(N))=pinv(diffmat(N,1,[0 Lf],'chebkind1'));
    Dinv(3:3:3*(N),3:3:3*(N))=pinv(diffmat(N,1,[0 Lf],'chebkind1'));
    [LRLs,URLs]=lu([Rs;Ls]);
    chebyshevmat=zeros(N,N-1);
    for iDeg=1:N-1
        chebyshevmat(:,iDeg)=chebyshevT(iDeg-1,(s0-Lf/2)*2/Lf);
    end
    I=zeros(2*N,2);
    wIt=zeros(2,2*N);
    for iR=1:N
        I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
        wIt(1:3,3*(iR-1)+1:3*iR)=w0(iR)*eye(3);
    end
end
