% Function that initializes the fibers as an (Nfib x N) x 3 array of
% points. Inputs = [a,b] = bounding box for the fibers (approximate), 
% Nfib = number of fibs, s = arclength coordinates for the fibers, 
% Lf = length of fibers
function Xfib=initFibs(a,b,Nfib,s)   
    Xfib = zeros(Nfib*length(s),3);
    for iFib=1:Nfib
        spt = a+rand(1,3)*(b-a);
        % Archmides? Don't remember what it's called. Projecting from
        % cylinder. 
        u = 1-2*rand;
        v = sqrt(1-u^2);
        w=2*pi*rand;
        pts=spt+s.*[v*cos(w) v*sin(w) u];
        Xfib(length(s)*(iFib-1)+1:length(s)*iFib,:)=pts;
    end
end