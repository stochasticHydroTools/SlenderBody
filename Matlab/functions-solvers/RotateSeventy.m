function outTau = RotateSeventy(inTau)
    fixedAngle = 70/180*pi; % 70 degrees
    
    axisDir = randn(1,3);
    axisDir = axisDir / norm(axisDir); % Normalize
    
    % 4. Create Rotation Matrix (using Rodrigues' Rotation Formula)
    a = cos(fixedAngle/2);
    b = -sin(fixedAngle/2)*axisDir(1);
    c = -sin(fixedAngle/2)*axisDir(2);
    d = -sin(fixedAngle/2)*axisDir(3);
    R = [a^2+b^2-c^2-d^2, 2*(b*c-a*d), 2*(b*d+a*c);
         2*(b*c+a*d), a^2+c^2-b^2-d^2, 2*(c*d-a*b);
         2*(b*d-a*c), 2*(c*d+a*b), a^2+d^2-b^2-c^2];
    outTau = reshape(inTau,1,3)*R';
end