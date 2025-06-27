deltaP = 0.1;
Lfac = 1+deltaP/L;
% Extend the last tangent vector
TauLast = barymat(L,s,b)*reshape(Xst,3,[])';
X3 = reshape(Xt,3,[])';
XLast = barymat(L,sNp1,bNp1)*X3;
Xadded = TauLast/norm(TauLast)*deltaP+XLast;

% Assuming you can reuse the same grid (later can regenerate grid)
s = Lfac*s;
w = Lfac*w;
sNp1 =Lfac*sNp1;
wNp1 = Lfac*wNp1;
DNp1 = DNp1/Lfac;
IntDNp1 = IntDNp1*Lfac;
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*...
    stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
% Bending energy matrix (2N+2 grid)
s2Np2 = Lfac*s2Np2;
w2Np2 = Lfac*w2Np2;
W2Np2 = diag(w2Np2);
D2Np2 = D2Np2/Lfac;
WTilde_Np1 = WTilde_Np1*Lfac;
WTilde_Np1_Inverse = WTilde_Np1_Inverse/Lfac;
BendingEnergyMatrix_Np1 = BendingEnergyMatrix_Np1/Lfac^3;
BendForceMat = -BendingEnergyMatrix_Np1;
BendMatHalf_Np1 = BendMatHalf_Np1/Lfac^(3/2);
L = L*Lfac;
eps = rtrue/L;
MobConst = -log(eps^2)/(8*pi*mu);


PositionsToMatch = [sNp1/Lfac; L];
Rnew = stackMatrix(barymat(PositionsToMatch,sNp1,bNp1));
% Fill in what you know
ErrorNew=@(NewDOF) (Rnew*NewPos(NewDOF,XonNp1Mat,Tau0BC',XTrk,N) - [Xt;Xadded']);
[azimuth,elevation,r] = cart2sph(Xst(4:3:end),Xst(5:3:end),Xst(6:3:end));
x0=[azimuth;elevation];
NewDOFs = lsqnonlin(ErrorNew,x0,[-pi*ones(N-1,1); -pi/2*ones(N-1,1)],...
    [pi*ones(N-1,1); pi/2*ones(N-1,1)]);
% Get new Tau's
X = NewPos(NewDOFs,XonNp1Mat,Tau0BC',XTrk,N);
TausTrk = XonNp1Mat \ X;
Xt = X;
Xst = TausTrk(1:3*N);

function X = NewPos(DOFs,XonNp1Mat,Tau0,XTrk,N)
    azimuth = DOFs(1:N-1);
    elevation = DOFs(N:2*N-2);
    r = ones(N-1,1);
    [x,y,z] = sph2cart(azimuth,elevation,r);
    tau3 = [Tau0;[x y z]];
    X = XonNp1Mat*[reshape(tau3',[],1);XTrk];
end
