% This is the main file to simulate 4 fibers falling due to gravity, 
% which is Section 5.1.1 in Maxian et al., "Integral-based spectral method
% for inextensible slender fibers in Stokes flow"
% https://arxiv.org/pdf/2007.11728.pdf
deltaLocal = 1; % part of the fiber to make ellipsoidal
nFib = 4;
N = 20; 
direct = 0;
NupsampleHydro = 40;
L=2;   % microns
mu=1;
eps=1e-3;
impcoeff = 1/2;
exactRPY = 0;
RectangularCollocation = 1; clamp0=0; twmod=0;
includeFPonLHS = 0;
a = exp(3/2)/4*eps*L; % match SBT
Eb=1;
dt=5e-4;
t=0;
tf=0.25;
stopcount = tf/dt;
grav=-10; % for falling fibers, value of gravity
% Nodes for tangent vectors, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
d=0.2;
X_s = [zeros(nFib*N,2) ones(nFib*N,1)];
theta = (0:nFib-1)/nFib*2*pi;
XMP=[d*cos(theta); d*sin(theta); 0*theta];
D = diffmat(N, 1, [0 L], 'chebkind1');
saveEvery = 1;
InitFiberVarsNew;
gravFDen = zeros(3*Nx,1); 
gravFDen(3:3:end)=grav/L;
GravForce = WTilde_Np1*gravFDen;
GravForce = repmat(GravForce,nFib,1);
AllXs = Xst;
AllX = Xt;
AllXMP = XMP;
Xpts=[];
eigThres=1e-5;
makeMovie=0;
if (makeMovie)
    f=figure;
    frameNum=0;
end
Lamguess = zeros(3*Nx*nFib,1);
AllLambdas = zeros(3*Nx*nFib,1);
AllLambdasPrev = zeros(3*Nx*nFib,1);
AllXnp1Star = AllX;
AllXp1 = zeros(3*Nx*nFib,1);
AllXsp1 = zeros(3*N*nFib,1);
AllXMP_p1 = zeros(3,nFib);
% For second order
AllXPrev = AllX;
AllXsPrev = AllXs;
AllXMPPrev = AllXMP;
WtInvSingle = WTilde_Np1_Inverse(1:3:end,1:3:end);
%% Computations
for count=0:stopcount-1 
    t=count*dt;
    if (mod(count,saveEvery)==0)
        PtsThisT = reshape(AllX,3,Nx*nFib)';
        if (makeMovie) 
            clf;
            frameNum=frameNum+1;
            for iFib=1:nFib
                fibInds = (iFib-1)*Nx+1:iFib*Nx;
                plot3(PtsThisT(fibInds,1),PtsThisT(fibInds,2),PtsThisT(fibInds,3));
                hold on
            end
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;reshape(AllX,3,[])'];
    end
    er = 1;
    itercount=0;
    XMob = AllX;
    Lamguess = AllLambdas; % initial guess
    if (impcoeff==1/2)
        XMob = 3/2*AllX-1/2*AllXPrev;
        Lamguess = 2*AllLambdas-AllLambdasPrev;
        AllLambdasPrev = AllLambdas;
    end
    while ((er > 1e-8 && count < 1/impcoeff) || itercount < 1)
        % Compute nonlocal velocity
        ForceNL = GravForce+Lamguess;
        for iFib=1:nFib
            inds=(iFib-1)*3*Nx+1:iFib*3*Nx;
            if (itercount==0)
                ForceNL(inds) = ForceNL(inds)+BendForceMat*XMob(inds);
            else
                ForceNL(inds) = ForceNL(inds)...
                    +BendForceMat*(impcoeff*AllXnp1Star(inds)+(1-impcoeff)*AllX(inds));
            end
        end
        UAll = UnboundedRPYSum(nFib,reshape(XMob,3,[])',...
            reshape(ForceNL,3,[])',a,mu,RupsampleHydro,wup,WtInvSingle,direct,0);
        USelf = UnboundedRPYSum(nFib,reshape(XMob,3,[])',...
            reshape(ForceNL,3,[])',a,mu,RupsampleHydro,wup,WtInvSingle,direct,1);
        uNonLoc = reshape((UAll-USelf)',[],1);
        U0All = uNonLoc;
        for iFib=1:nFib
            Xst = AllXs((iFib-1)*3*N+1:iFib*3*N);
            XstPrev = AllXsPrev((iFib-1)*3*N+1:iFib*3*N);
            ForceExt = GravForce((iFib-1)*3*Nx+1:iFib*3*Nx);
            XMP = AllXMP(:,iFib);
            XMPPrev = AllXMPPrev(:,iFib);
            U0 = U0All((iFib-1)*3*Nx+1:iFib*3*Nx);
            TemporalIntegrator_TransDet;
            AllLambdas((iFib-1)*3*Nx+1:iFib*3*Nx) = Lambda;
            AllXnp1Star((iFib-1)*3*Nx+1:iFib*3*Nx) = Xp1Star;
            AllXp1((iFib-1)*3*Nx+1:iFib*3*Nx) = Xp1;
            AllXsp1((iFib-1)*3*N+1:iFib*3*N) = Xsp1;
            AllXMP_p1(:,iFib)=XMP_p1;
        end
        er=norm(Lamguess-AllLambdas)/norm(AllLambdas);
        Lamguess = AllLambdas;
        itercount=itercount+1;
    end
    AllXPrev = AllX;
    AllX = AllXp1;
    AllXsPrev = AllXs;
    AllXs = AllXsp1;
    AllXMPPrev = AllXMP;
    AllXMP = AllXMP_p1;
end
Xpts=[Xpts;reshape(AllX,3,Nx*nFib)'];