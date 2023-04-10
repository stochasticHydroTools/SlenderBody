function [U,V] = Bishop(T_hat,u1)
% Generate the Bishop frame by paralel transporting u1
% along T_hat

t1 = T_hat(:,1);
v1 = cross(t1,u1);

U = [u1];
V = [v1];

[~,q] = size(T_hat);

for k= 1:q-1
   % calc rot quantities
   t_k = T_hat(:,k); 
   t_kp = T_hat(:,k+1); 
   cos_th = dot(t_k,t_kp);
   rot_x = cross(t_k,t_kp);
   %rot_x = rot_x./norm(rot_x);
   
   % update bishop fram using Rod. formula
   u_k = U(:,k);
   v_k = V(:,k);
   u_kp = u_k + cross(rot_x,u_k) + (1./(1 + cos_th)) * cross(rot_x,cross(rot_x,u_k));
   v_kp = cross(t_kp,u_kp);
   
   U = [U u_kp];
   V = [V v_kp];
end

end