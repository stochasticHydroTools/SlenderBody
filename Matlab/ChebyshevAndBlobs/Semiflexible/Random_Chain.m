function taus = Random_Chain(Nlinks,alphas)
% Generate a random chain according to the equilibrium distribution of
% angles \alpha_i = \tau_i \cdot \tau_{i-1} ~ exp(-gamma sim(\alpha_i/2)^2)

taus = NaN(3,Nlinks);


tau = randn(3,1);
tau = tau./norm(tau);

u = randn(3,1);
u = u - dot(u,tau)*tau;
u = u./norm(u);
v = cross(u,tau);



taus(:,1) = tau;

for i = 2:Nlinks
    t = taus(:,i-1);
    
    % generate a uniform random axis orthog. to tau
    % and rotate by alpha around it
    % NOTE: x=unit_vec(randn(2,1)) is uni on circle    
    om = randn(2,1);
    om = om./norm(om);
    axis = [u v]*om;
    
    theta = alphas(i-1);
    
    
    taus(:,i) = cos(theta)*t + sin(theta)*cross(axis,t);
    taus(:,i) = taus(:,i)./norm(taus(:,i));
    
    u = cos(theta)*u + (sin(theta))*cross(axis,u) + dot(axis,u)*(1-cos(theta))*axis;
    v = cross(u,taus(:,i));
    u = u./norm(u);
    v = v./norm(v);
    
end

end


% function taus = Random_Chain(Nlinks,alphas)
% % Generate a random chain according to the equilibrium distribution of
% % angles \alpha_i
% 
% taus = NaN(3,Nlinks);
% 
% 
% t = randn(3,1);
% t = t./norm(t);
% 
% b = randn(3,1);
% b = b - dot(b,t)*t;
% b = b./norm(b);
% n = cross(b,t);
% n = n./norm(n);
% 
% Rot = @(psi,theta) [cos(psi)*cos(theta), cos(psi)*sin(theta), -sin(psi);...
%                     -sin(theta), cos(theta), 0; ...
%                     sin(psi)*cos(theta), sin(psi)*sin(theta), cos(psi)];
% 
% taus(:,1) = t;
% 
% frame = [n';b';t'];
% frame = Rot(alphas(1),0)*frame;
% frame = frame./sqrt(sum(frame.^2,2));
% taus(:,2) = frame(3,:)';
% 
% 
% 
% for i = 2:Nlinks-1
%     rt = randn(3,3); 
% 	rt = rt./sqrt(sum(rt.^2));
%     b_1 = cross(rt(:,1),rt(:,2));
% 	b_2 = cross(rt(:,2),rt(:,3));
% 	b_1 = b_1./norm(b_1);
% 	b_2 = b_2./norm(b_2);
%     theta = acos(dot(b_1,b_2));
%     psi = alphas(i);
%     
%     
%     frame = Rot(psi,theta)*frame;
%     frame = frame./sqrt(sum(frame.^2,2));
%     taus(:,i+1) = frame(3,:)';
% end
% 
% end

