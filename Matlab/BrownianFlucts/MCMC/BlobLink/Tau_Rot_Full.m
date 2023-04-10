function [T_out,U_out,V_out] = Tau_Rot_Full(dt,Omega,U,V,T_hat)
% Update Fiber Positions from Omega
% Omega is a full frame velocity in [T U V] coordinates


T_out = NaN*T_hat;
U_out = NaN*T_hat;
V_out = NaN*T_hat;



[~,q] = size(T_hat);

Om = reshape(Omega,3,q);

for jj = 1:q
    frame = [T_hat(:,jj) U(:,jj) V(:,jj)];
    Omega_i = frame*Om(:,jj);
    mag_O = norm(Omega_i);
    if (mag_O > 1e-10)
    theta = dt*mag_O;
    axis = Omega_i./mag_O;
    axis = repmat(axis,1,3);
    
    new_frame = cos(theta)*frame + (sin(theta))*cross(axis,frame) + (1-cos(theta))*(axis(:,1)*axis(:,1)')*frame;
    new_frame = new_frame./sqrt(sum(new_frame.^2));
    else
    new_frame=frame;
    end
    T_out(:,jj) = new_frame(:,1);
    U_out(:,jj) = new_frame(:,2);
    V_out(:,jj) = new_frame(:,3);
end

end
