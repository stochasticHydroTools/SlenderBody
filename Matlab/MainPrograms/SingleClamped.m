close all;
% Two clamped filaments growing against a membrane
nFib = 1;
L = 2;   % microns
N = 20;
Poly0 = 0; % Base Rate
rtrue = 4e-3; % 4 nm radius
eps = rtrue/L;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
doSterics = 0;
mu=1;
a=rtrue*exp(3/2)/4;
upsamp=0; % For hydro
makeMovie =1;
updateFrame = 1;
dt = 0.25;
tf = 200;
xtau=0.1;
%Tau0BC=[xtau -xtau 1; sqrt(1-xtau^2) sqrt(1-xtau^2) 0; 0 0 0];
Tau0BC = [0;1;0];
XMP=[0;L/2;0];
X_s=[];
Lfacs = 2/L*ones(nFib,1);
for iFib=1:nFib
    X_s=[X_s; Lfacs(iFib)*repmat(Tau0BC(:,iFib)',N,1)];
end
[s,w,b] = chebpts(N, [0 L], 2); % 1st-kind grid for ODE.
%X_s = Lfacs(1)*[cos(Lfacs(1)*s) sin(Lfacs(1)*s) zeros(N,1)];
%X_s = repmat(X_s,nFib,1);

%% Preliminaries
Nx = N+1;
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],2);
DNp1 = diffmat(Nx,[0 L],'chebkind2');
RToNp1 = barymat(sNp1,s,b);
RNp1ToN = barymat(s,sNp1,bNp1);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*(N+1),3);
for iR=1:N+1
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
Xst = zeros(3*N*nFib,1);
Xt = zeros(3*Nx*nFib,1);
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*...
    stackMatrix(IntDNp1*RToNp1) I];
InvXonNp1Mat = [stackMatrix(RNp1ToN*DNp1); BMNp1];
for iFib=1:nFib
    Xst((iFib-1)*3*N+1:iFib*3*N) = ...
        reshape(X_s((iFib-1)*N+1:iFib*N,:)',[],1);
    Xt((iFib-1)*3*Nx+1:iFib*3*Nx) = XonNp1Mat*...
        [Xst((iFib-1)*3*N+1:iFib*3*N);XMP(:,iFib)];
end
% Bending energy matrix (2N+2 grid)
[s2Np2, w2Np2, b2Np2] = chebpts(2*Nx, [0 L], 2);
W2Np2 = diag(w2Np2);
R_Np1_To_2Np2 = barymat(s2Np2,sNp1,bNp1);
D2Np2 = diffmat(2*Nx,[0 L],'chebkind2');
WTilde_Np1 = stackMatrix((R_Np1_To_2Np2'*W2Np2*R_Np1_To_2Np2));
WTilde_Np1_Inverse = WTilde_Np1^(-1);
BendingEnergyMatrix_Np1 = Eb*stackMatrix((R_Np1_To_2Np2*DNp1^2)'*...
    W2Np2*R_Np1_To_2Np2*DNp1^2);
BendForceMat = -BendingEnergyMatrix_Np1;
saveEvery=max(floor(5e-2/dt),1);
% Hydrodynamics
NupsampleHydro = 200;%ceil(2/a);
[sup,wup,~] = chebpts(NupsampleHydro, [0 L],2);
RupsampleHydro = stackMatrix(barymat(sup,sNp1,bNp1));
WUpHydro = stackMatrix(diag(wup));
BDCell = repmat({RupsampleHydro},nFib,1);
RupsampleHydro_BD = blkdiag(BDCell{:});
BDCell = repmat({WUpHydro},nFib,1);
WUpHydro_BD = blkdiag(BDCell{:});
BDCell = repmat({WTilde_Np1_Inverse},nFib,1);
WTInv_BD = blkdiag(BDCell{:});
AllbS_Np1 = precomputeStokesletInts(sNp1,L,a,N+1,1);
AllbD_Np1 = precomputeDoubletInts(sNp1,L,a,N+1,1);
NForSmall = 8; % # of pts for R < 2a integrals for exact RPY
eigThres = 1e-3;

Lprime=Poly0;
Nusteric = 101;
suSteric = (0:Nusteric-1)'*L/(Nusteric-1);
RuniSteric = barymat(suSteric,sNp1,bNp1);
Lprimes=[];

%% Initialization 
Npl=1000;
[spl,wpl,bpl]=chebpts(Npl,[0 L]);
RplNp1 = barymat(spl,sNp1,bNp1);
RplN = barymat(spl,s,b);
stopcount=floor(tf/dt+1e-5);
Xpts=[];
AllLfacs = [];
AllXs = Xst;
AllX = Xt;
if (makeMovie)
    f=figure;
    frameNum=0;
    %tiledlayout(1,4, 'Padding', 'none', 'TileSpacing', 'compact');
end

%% Computations
for count=0:stopcount
    t=count*dt;
    if (mod(count,saveEvery)==0)
        Lprimes=[Lprimes Lprime];
        PtsThisT = reshape(AllX,3,Nx*nFib)';
        %max(abs(sqrt(sum(fTw3.*fTw3,2))))
        if (makeMovie)% && abs(t-tmovies(frameNum+1)) < dt/2) 
            clf;
            %nexttile
            frameNum=frameNum+1;
            %subplot(3,3,frameNum)
            for iFib=1:nFib
                fibInds = (iFib-1)*Nx+1:iFib*Nx;
                plot3(RplNp1*PtsThisT(fibInds,1),RplNp1*PtsThisT(fibInds,2),...
                    RplNp1*PtsThisT(fibInds,3));
                hold on
                FramePts = RNp1ToN*PtsThisT(fibInds,:);
            end
            xlim([-2 2])
            %xlim([-0.15 0.15])
            %ylim([-0.15 0.15])
            %zlim([-0.5 0.5])
            %view([ -59.1403    5.5726])
            view(2)
            PlotAspect
            %title(strcat('$t=$',num2str(tmovies(frameNum))))
            title(strcat('$t=$',num2str((frameNum-1)*saveEvery*dt)))
            movieframes(frameNum)=getframe(f);
        end
        Xpts=[Xpts;PtsThisT];
        AllLfacs=[AllLfacs; Lfacs'];
        % if (t>1)
        %     keyboard
        % end
     end
    % Evolve system
    F_Ext = zeros(3*Nx*nFib,1);
    X3All = reshape(AllX,3,[])';
    X3sNp1=DNp1*X3All;
    F_Ext(end-2:end)=-28*Eb/L^2*X3sNp1(end,:)';
    if (doSterics)
        X3All = reshape(AllX,3,[])';
        StericForce = StericForces(X3All,RuniSteric,rtrue,kbT,...
            suSteric,nFib,0,0,Lfacs);
        F_Ext=F_Ext+reshape(StericForce',[],1);
    end
    
    %warning("Need to adjust matrices for polymerization")
    %MBlock = zeros(nFib*3*Nx);
    %KBlock = zeros(nFib*3*Nx);
    %ForceMatBlock = zeros(nFib*3*Nx);
    %F_Bends = zeros(nFib*3*Nx,1);
    % Compute upsampled mobility matrix
    % The net scaling with L cancels out here
    if (upsamp)
        Xup = reshape(RupsampleHydro_BD*AllX,3,[])';
        MRPY = getGrandMBlobs(nFib*NupsampleHydro,Xup,a,mu);
        MWsym = WTInv_BD*RupsampleHydro_BD'...
            *WUpHydro_BD*MRPY*WUpHydro_BD*...
            RupsampleHydro_BD*WTInv_BD;
        MBlock=MWsym;
    end
    for iFib=1:nFib
        inds=(iFib-1)*3*Nx+1:iFib*3*Nx;
        BendForceMat_L = BendForceMat/Lfacs(iFib)^3;
        %ForceMatBlock(inds,inds)=BendForceMat_L;
        WTilde_Np1_Inv_L = WTilde_Np1_Inverse/Lfacs(iFib);
        Xst = AllXs((iFib-1)*3*N+1:iFib*3*N);
        Xt = AllX((iFib-1)*3*Nx+1:iFib*3*Nx);
        F_Bend=BendForceMat_L*Xt;
        % Arguments for the solver, assuming first order
        impcoeff=1;
        % Saddle point solve
        Xs3 = reshape(Xst,3,N)';
        % if (~upsamp)
        %     AllbS_Np1 = precomputeStokesletInts(sNp1*Lfacs(iFib),...
        %         L*Lfacs(iFib),a,N+1,1);
        %     AllbD_Np1 = precomputeDoubletInts(sNp1*Lfacs(iFib),...
        %         L*Lfacs(iFib),a,N+1,1);
        %     M = TransTransMobilityMatrix(reshape(Xt,3,[])',...
        %         a,L*Lfacs(iFib),mu,sNp1*Lfacs(iFib),bNp1,...
        %         DNp1/Lfacs(iFib),AllbS_Np1,AllbD_Np1,NForSmall,0,0,0);
        %     MWsym = 1/2*(M*WTilde_Np1_Inv_L + WTilde_Np1_Inv_L*M');
        %     MWsym = FilterM(1/2*(MWsym+MWsym'),eigThres);
        %     MBlock(inds,inds) = MWsym;
        % end
        %MBlock(inds,inds)=eye(3*Nx);
        M = eye(3*Nx);
        % Solve for fiber evolution
        K = KonNp1(Xs3,XonNp1Mat,I);
        % For clamping (B matrix)
        nCons=2;
        BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
            stackMatrix([barymat(0,s,b) 0])];
        if (iFib==3)
            BProj = [stackMatrix(barymat(0,sNp1,bNp1))*K;...
                stackMatrix([barymat(0,s,b) 0]); ...
                stackMatrix(barymat(L,sNp1,bNp1))*K;...
                stackMatrix([barymat(L,s,b) 0])];
            nCons=4;
        end
        U0=zeros(3*Nx,1);
        if (t < 0.01)
            U0(1:3:end)=1;
        end
        ProjectClamp = eye(3*Nx)-BProj'*pinv(BProj*BProj')*BProj;
        Kaug = K*ProjectClamp;
        B = Kaug-impcoeff*dt*M*BendForceMat_L*Kaug;
        RHS = Kaug'*(F_Bend+F_Ext(inds)+M\U0);
        alphaU = lsqminnorm(Kaug'*(M \ B),RHS);
        % Form psuedo-inverse by only 
        maxRank = 2*N+3-2.5*nCons;
        [U,S,V]=svd(Kaug'*(M \ B));
        S=diag(S);
        S(maxRank+1:end)=0;
        S(S/S(1)<1e-10)=0;
        pinvS = 1./S;
        pinvS(S==0)=0;
        PInvSP = V*diag(pinvS)*U';
        alphaU = PInvSP*RHS;
        % SPMat = [-M B zeros(3*Nx,6); K' zeros(3*Nx) BProj'; ...
        %     zeros(6,3*Nx) BProj zeros(6)];
        % RHS = [M*(F_Bend+F_Ext)+U0;zeros(3*Nx+6,1)];
        % Everything = lsqminnorm(SPMat,RHS);
        % alphaU_2 = Everything(3*Nx+1:6*Nx);
        % if (max(abs(alphaU_1-alphaU))>1e-5)
        %     keyboard
        % end
        % Full saddle pt (more stable for whatever reason?)
        
        Omega = reshape(alphaU(1:3*N),3,N)';
        Xs3 = reshape(Xst,3,N)';
        XBC=barymat(0,s,b)*Xs3;
        % if (norm(XBC-[0 1 0])>1e-2)
        %     keyboard
        % end
        % Update with an inextensible motion
        newXs = rotateTau(Xs3,Omega,dt);
        Xsp1 = reshape(newXs',[],1);
        XMP_p1 = XMP(:,iFib)+dt*alphaU(end-2:end);
        Xp1 = XonNp1Mat*[Xsp1;XMP_p1]; 
        if (Lprime(iFib) > 0)
            [X3new,Lfacs(iFib)]=PolymerizeBarbed(Lprime(iFib),Lfacs(iFib),...
                L,Xp1,sNp1,bNp1,DNp1,dt); 
            Xp1 = reshape(X3new',[],1);
            NewConf = XonNp1Mat \ Xp1;
            Xsp1 = NewConf(1:3*N);
            XMP_p1 = NewConf(3*N+1:end);            
        end
        AllX((iFib-1)*3*Nx+1:iFib*3*Nx) = Xp1;
        AllXs((iFib-1)*3*N+1:iFib*3*N) = Xsp1;
        XMP(:,iFib)=XMP_p1;
    end
end
% norm(Xt(1:3)'-Xpts(Nx+1,:))
% Xs0=barymat(0,s,b)*Xs3;
% Xs0=Xs0/norm(Xs0);
% norm(Xs0-[0 1 0])
