% Shift vector periodically along primed coordinates until it 
% is on [-L/2,L/2]^3
function [dvec,shift] = calcShifted(dvec,g,Lx,Ly,Lz)
    % Shift in oblique y and z
    shift1 = round([0*dvec(:,1) dvec(:,2)/Ly dvec(:,3)/Lz]);
    dvec = dvec-([g*Ly*shift1(:,2) Ly*shift1(:,2) Lz*shift1(:,3)]);
    % Shift in x
    shift2 = round(dvec(:,1)/Lx);
    shift = [shift2 shift1(:,2:3)];
    dvec = dvec-[shift2*Lx 0*shift2 0*shift2];
end