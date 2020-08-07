% addpath('../functions-solvers')
% rng(1);
% % Points and forces
% Npts=4;
% pts=[0 0 0; 1 0 0; 0.5 1 0; 1.5 1 0];
% forces=[1 1 1; -1 -1 -1; 2 2 2; -2 -2 -2];
% mu=3;
% % Define the grid
% Lx=2;
% Ly=Npts/2;
% % if (Ly > 2)
% %     pts=[pts; 0 2 0; 1 2 0; 0.5 3 0; 1.5 3 0];
% %     forces = [forces; -1 -1 -1 ; 1 1 1; -2 -2 -2; 2 2 2];
% % end
% Lz=2;
% xi=5;
% a=0.0245;
% % g=0.5+(Ly-2)*(-0.25);
% Ly=4;
g=0.1;
Lx=2.4;
Ly=2.8;
Lz=3.2;
a=0.012;
mu=1.5;
xi=3;
Npts = 200;
pts=load('../Python/points.txt');
forces=load('../Python/forces.txt');

velfar = EwaldFarVel(pts,forces,mu,Lx,Ly,Lz,xi,a,g);
velNear = EwaldNearSum(Npts,pts,forces,xi,Lx,Ly,Lz,a,mu,g);
% velEwald=velNear+velfar;

% The free space answer
% velFull=zeros(3,Npts);
% for iPt=1:Npts
%     Mself = 1/(6*pi*mu*a)*eye(3);
%     velFull(:,iPt)=velFull(:,iPt)+Mself*(forces(iPt,:)');
%     for jPt=iPt+1:Npts
%         rvec=pts(iPt,:)-pts(jPt,:);
%         r=norm(rvec);
%         rhat=rvec/r;
%         if (r > 2*a)
%             OtherRPY = 1/(6*pi*mu*a)*((3*a/(4*r)+a^3/(2*r^3))*eye(3)+...
%                 (3*a/(4*r)-3*a^3/(2*r^3))*(rhat'*rhat));
%         else 
%             OtherRPY = 1/(6*pi*mu*a)*((1-9*r/(32*a))*eye(3)+...
%                 (3*r/(32*a))*(rhat'*rhat));
%         end
%         velFull(:,iPt)=velFull(:,iPt)+OtherRPY*(forces(jPt,:)');
%         velFull(:,jPt)=velFull(:,jPt)+OtherRPY*(forces(iPt,:)');
%     end
% end
% abs((velFull-velEwald))
