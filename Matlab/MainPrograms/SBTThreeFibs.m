% This is the main file to simulate 3 fibers in shear flow,
% which is Section 5.1.2 in Maxian et al., "Integral-based spectral method
% for inextensible slender fibers in Stokes flow"
% https://arxiv.org/pdf/2007.11728.pdf
% There is a corresponding Python file; this version is just to check the
% python version
clear AllXTrajs
%for iDt=1:6
%dt =0.8/2^(iDt);
deltaLocal = 0.1; % part of the fiber to make ellipsoidal
nFib = 3;
N = 16; 
NupsampleHydro = 32;
L=2;   % microns
mu=1;
eps=1e-3;
impcoeff = 1;
exactRPY = 1;
upsamp = 0; % -1 for direct, 0 for special quad, 1 for upsampled direct
RectangularCollocation = 0; clamp0=0; twmod=0;
includeFPonLHS = 0;
a = exp(3/2)/4*eps*L; % match SBT
gam0=1; omega=0; % shear flow
Eb=0.01;
dt=0.05;
t=0;
tf=2.4;
xi=6;
Lds=2.4*ones(1,3);
stopcount = floor(tf/dt+1e-6);
% Nodes for tangent vectors, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
% Falling fibers
X_s = [ones(N,1) zeros(N,2); zeros(N,1) ones(N,1) zeros(N,1); ones(N,1) zeros(N,2)];
XMP=[0 -0.6 -0.04; 0 0 0; 0 0.6 0.06]'; 
D = diffmat(N, 1, [0 L], 'chebkind1');
saveEvery = 1;
InitFiberVarsNew;
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
maxIts=20; % for the first 2 steps
rigid=0;
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
    [UShear,gSh] = EvalU0(gam0,omega,t+(1-impcoeff)*dt,XMob,'S');
    while ((er > 1e-6 && count < 1/impcoeff && itercount < maxIts) || itercount < 1)
        % Background flow, strain, external force
        % Compute nonlocal velocity
        ForceNL =Lamguess;
        for iFib=1:nFib
            inds=(iFib-1)*3*Nx+1:iFib*3*Nx;
            if (itercount==0)
                ForceNL(inds) = ForceNL(inds)+BendForceMat*XMob(inds);
            else
                ForceNL(inds) = ForceNL(inds)...
                    +BendForceMat*(impcoeff*AllXnp1Star(inds)+(1-impcoeff)*AllX(inds));
            end
        end
        %UAllPer = PeriodicRPYSum(nFib,reshape(XMob,3,[])',reshape(ForceNL,3,[])',...
        %   Lds,xi,gSh,a,mu,RupsampleHydro,wup,WtInvSingle,upsamp==-1);
        UAllPer = UnboundedRPYSum(nFib,reshape(XMob,3,[])',reshape(ForceNL,3,[])',...
            a,mu,RupsampleHydro,wup,WtInvSingle,upsamp==-1,0);
        USelf = UnboundedRPYSum(nFib,reshape(XMob,3,[])',...
            reshape(ForceNL,3,[])',a,mu,RupsampleHydro,wup,WtInvSingle,upsamp==-1,1);
        uNonLoc = reshape((UAllPer-USelf)',[],1);
        U0All = uNonLoc+UShear;
        for iFib=1:nFib
            Xst = AllXs((iFib-1)*3*N+1:iFib*3*N);
            XstPrev = AllXsPrev((iFib-1)*3*N+1:iFib*3*N);
            ForceExt = zeros(3*Nx,1);
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
        er=norm(Lamguess-AllLambdas)/max(norm(AllLambdas),1);
        Lamguess = AllLambdas;
        itercount=itercount+1;
        eriters(itercount)=er;
    end
    if (itercount==maxIts)
        warning('Exceeded max iters without getting below error thres')
    end
    AllXPrev = AllX;
    AllX = AllXp1;
    AllXsPrev = AllXs;
    AllXs = AllXsp1;
    AllXMPPrev = AllXMP;
    AllXMP = AllXMP_p1;
end
Xpts=[Xpts;reshape(AllX,3,Nx*nFib)'];
%AllXTrajs{iDt}=Xpts;
% end
% for iDt=1:length(AllXTrajs)-1
%     gtype=2;
%     if (RectangularCollocation)
%         gtype=1;
%     end
%     DTErrors(iDt)=max(DynamicError(nFib,2,L,AllXTrajs{iDt},AllXTrajs{iDt+1},Nx,Nx,gtype,gtype));
% end