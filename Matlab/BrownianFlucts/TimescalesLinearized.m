% kbT = 4.1e-3;
% lpstar = 10;
% L = 2;
% Eb = lpstar*L*kbT;
% eps = 1e-3;
% gam0=0*1.6e4*kbT/L^3;
% IdForM = 0;
% N = 24;
% upsamp = 0;
% if (eps == 1e-3)
% if (N==12)
%     eigThres = 3.2/L;
% elseif (N==24)
%     eigThres = 5.0/L;
% elseif (N==36)
%     eigThres = 6.7/L;
% end
% elseif (eps==1e-2)
% if (N==12)
%     eigThres = 1.6/L;
% elseif (N==24)
%     eigThres = 1.0/L;
% elseif (N==36)
%     eigThres = 0.34/L;
% end
% end
% CurvedX0=0;
% PenaltyForceInsteadOfFlow = 1;
% RectangularCollocation = 0;
a = eps*L;
mu=1;           % Viscosity
nFib=1;
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
% Fiber initialization 
if (CurvedX0)
    q=1; 
    X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
else
    X_s = [ones(N,1) zeros(N,2)];
end
XMP=[0;0;0];
InitFiberVarsNew;% Generalized eigenvalues
if (IdForM)
    M = eye(3*Nx)/(8*pi*mu);
else
    M = computeRPYMobility(N,Xt,DNp1,a,L,mu,sNp1,...
                bNp1,AllbS_Np1,AllbD_Np1,NForSmall,0);
end
MWsym = 1/2*(M*WTilde_Np1_Inverse + WTilde_Np1_Inverse*M');
MWsym = FilterM(MWsym,eigThres);
K = KonNp1(X_s,XonNp1Mat,I);
Mob = K*pinv(K'*(MWsym \ K))*K';
MobInv = pinv(Mob);
MobInv=(MobInv+MobInv')/2; % symmetrize
EMat_Np1 = (EMat_Np1+EMat_Np1')/2;
% Generalized eigenvalues
[V,Lam] = eig(MobInv,EMat_Np1);
%V = V(:,1:2*N+3);
%Lam = Lam(1:2*N+3,1:2*N+3);
% Check normalization
% nzation=diag(V'*EMat_Np1*V);
% V = V./sqrt(nzation)';
[Timescales,Inds] = sort(diag(Lam),'descend');
V=V(:,Inds);
Lam = diag(Timescales);
%er0 = MobInv*V-EMat_Np1*V*Lam;
%er1=V'*EMat_Np1*V-eye(2*N+3);
%er2 = V'*MobInv*V-Lam;
%er3 = MobInv-EMat_Np1*V*Lam*V'*EMat_Np1;
%tbar = L^4*5e-4/(Eb*log(1/eps));
tauSm = 8*pi*mu*L^4/(Eb*log(1/eps^2));
%loglog(1:2*N+3,Timescales(1:2*N+3)/tauSm,':s') 
%hold on