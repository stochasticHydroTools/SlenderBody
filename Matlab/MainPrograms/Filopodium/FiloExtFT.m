X3All = reshape(AllX,3,[])';
% Update CLs
if (updateCL)
    [links,rCLs] = updateDynamicCLs(links,rCLs,Runi,X3All,nFib,0,KCL,ellCL,kbT,konCL,koffCL,dt);
end
%% External FORCES (not density)
[CLForce,X1stars,X2stars] = getCLforceEn(links,X3All,Runi,KCL, rCLs,0,0);
ExtForce = CLForce;
if (doSterics)
    StericForce = StericForces(X3All,RuniSteric,rtrue,kbT,suSteric,nFib,0,0);
    ExtForce=ExtForce+StericForce;
end
ExtForceDen = 0*ExtForce;
% Convert to density
for iFib=1:nFib
    WFib = Lfacs(iFib)*WTilde_Np1(1:3:end,1:3:end);
    ExtForceDen((iFib-1)*Nx+1:iFib*Nx,:)=WFib \ ExtForce((iFib-1)*Nx+1:iFib*Nx,:);
end
% Convert to density
U0All = zeros(3*Nx*nFib,1);
% Revisit this later (not independent of polymerization stuff)
%CLtorques = getCLTorque(links,X1stars,X2stars,CLCoords_1,CLCoords_2,...
%    RUniTau,sNp1,wNp1,reshape(AllXs,3,[])',D1,Ktorq,L,Lfacs,RNp1ToPsi);
% Compute angle of midpoint in unit circle
fmotors = zeros(Nx*nFib,3);
nmotors = zeros(Npsi*nFib,1);
for iFib=1:nFib
    XThis=reshape(AllX((iFib-1)*3*Nx+1:iFib*3*Nx),3,Nx)';
    angleCirc = atan2(XThis(:,2),XThis(:,1));
    RCirc = sqrt(XThis(:,1).^2+XThis(:,2).^2);
    Eligible = (RCirc > (1-MotorCircleFrac)*RFilo & (sNp1/L <= MotorLengthFrac));
    ForceVec = [-sin(angleCirc) cos(angleCirc) zeros(Nx,1)];
    NormalForce = -[cos(angleCirc) sin(angleCirc) zeros(Nx,1)];
    fmotorFib = fmot0*ForceVec.*Eligible -fmotDwn*[0 0 1].*Eligible...
        + MembraneForce*NormalForce.*(RCirc > (1-MotorCircleFrac)*RFilo);
    inds = (iFib-1)*Nx+1:iFib*Nx;
    indsTorq = (iFib-1)*Npsi+1:iFib*Npsi;
    fmotors(inds,:)=fmotors(inds,:)+fmotorFib;
    RPsi = RNp1ToPsi*RCirc;
    EligiblePsi = (RPsi > (1-MotorCircleFrac)*RFilo  & sPsi/L <= MotorLengthFrac);
    nmotors(indsTorq)=MotorTorq.*EligiblePsi;
end
fextAll = ExtForceDen+fmotors;
n_extAll = nmotors;