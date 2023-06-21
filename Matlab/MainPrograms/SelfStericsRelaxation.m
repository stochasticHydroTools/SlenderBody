% % Main file for single relaxing fiber, to test the influence of twist
% % elasticity 
% % This Section 5.2 in the paper Maxian et al. "The hydrodynamics of a
% % twisting, bending, inextensible filament in Stokes flow"
% %close all;
% addpath('../functions-solvers')
% %PenaltyForceInsteadOfFlow = 1; kbT=0; ModifyBE=1; gam0=2000*kbT;
% exactRPY=1; 
% TransTransLDOnly=0; % Local drag only
% deltaLocal=0; % regularization for local drag
% upsamp=0; % no upsampling
% rigid=0;
% N = 16;   
% DoSterics=1;
% RectangularCollocation = 0; 
% clamp0=0; 
% L = 2;            
% epsHat = 1e-2;
% a = epsHat*L;
% kbT = 4.1e-3;
% Eb=L*kbT;           % Bending modulus
% mu=1;           % Viscosity
% makeMovie = 1;
% if (makeMovie)
%     f=figure;
%     movieframes=getframe(f);
% end
% nFib=1;
% dt = 1e-2;
% tf = 4;
% stopcount = floor(1e-6+tf/dt);
% impcoeff = 1;
% t=0;
% [s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE
% D = diffmat(N, 1, [0 L], 'chebkind1');
% % Fiber initialization 
% q=7; 
% X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
% %X_s = reshape(AllTanVecsMCMC(:,5),3,[])';
% XMPor0 = [0;0;0]; XMP=XMPor0;
% saveEvery=max(1e-4/dt,1);
% InitFiberVars;
% updateFrame=1;
% Xpts=[];
% NUni = 1/epsHat+1;
% su = (0:NUni-1)'*a;
% Runi=barymat(su,sNp1,bNp1);
% %% Computations
% for count=0:stopcount-1 
%     t=count*dt;
%     if (mod(count,saveEvery)==0)
%         if (makeMovie) 
%             clf;
%             plot3(Runi*Xt(1:3:end),Runi*Xt(2:3:end),Runi*Xt(3:3:end),'LineWidth',2)
%             movieframes(length(movieframes)+1)=getframe(f);
%         end
%         Xpts=[Xpts;reshape(Xt,3,N+1)'];
%     end
%     U0 = zeros(3*Nx,1);
%     ForceExt = zeros(3*Nx,1);
%     if (DoSterics)
%         StericForce = StericForces(reshape(Xt,3,Nx)',Runi,a,kbT,su,nFib,0,0);
%         ForceExt = ForceExt + reshape(StericForce',[],1);
%     end
%     TemporalIntegrator_TransDet;
%     Xt = Xp1;
%     Xst = Xsp1;
%     try
%     XMPor0 = XMPor0_p1;
%     catch
%     XMPor0 = XMP_p1;
%     end
%     maxU(count+1)=max(abs(K*alphaU));
% end
% Xpts=[Xpts;reshape(Xt,3,N+1)'];
% if (makeMovie)
%     movieframes(1)=[];
% end
% % For computing L^2 difference between sterics and no sterics
% nT = length(Xpts)/Nx;
% L2Diff = zeros(nT,1);
for iT=1:nT
diff=X_Ster((iT-1)*Nx+1:iT*Nx,:)-X_NoSter((iT-1)*Nx+1:iT*Nx,:);
diff = reshape(diff',[],1);
L2Diff(iT) = sqrt(diff'*WTilde_Np1*diff);
ster=X_Ster((iT-1)*Nx+1:iT*Nx,:);
noster=X_NoSter((iT-1)*Nx+1:iT*Nx,:);
plot3(Runi*ster(:,1),Runi*ster(:,2),Runi*ster(:,3),'LineWidth',2)
hold on
plot3(Runi*noster(:,1),Runi*noster(:,2),Runi*noster(:,3),'LineWidth',2)
drawnow
hold off
end