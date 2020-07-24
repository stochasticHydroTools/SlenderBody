% THE MATRIX FOR THE FIRST INTEGRAL
function FIMat = FIMatrix(N)
    jj=(2:N-1)';
    colm1=[0; 1; 1./(2*jj)];
    colp1=[0; -1/2; -1./(2*jj).*(jj<N-1)];
    FIMat=spdiags([colm1 colp1],[-1 1],N,N+2);
    FIMat(1,N+2)=1;
end