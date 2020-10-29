% Near fiber target quadrature test from Appendix C
% Toggle between long and short distances on lines 50-51
Nfib=100;
Ntarg=100;
er=zeros(Nfib,Ntarg);
qtypes=zeros(Nfib,Ntarg);
dists=zeros(Nfib,Ntarg);
Lf=2;
epsilon=1e-3;
mu=1/(8*pi);
rng(1);
K = 16;
k = 0:K-1;
for iFib=1:Nfib
    % Generate a fiber with exponentially decaying coefficients
    xhats=ones(K,3);
    while (sum(sum(abs(xhats(3:end,:)) >  exp(-0.61*k(3:end)'))) >  0)
        nk = numel(k);
        ampl = exp(-k/nk*10);   % Exponential decay to 4 digits
        c = randn(3,nk) .* ampl;    % cmplx F coeffs in each coord
        xpf = @(t) c(1,:)*(cos(k'.*acos(2/Lf*t(:)'-1)));
        ypf = @(t) c(2,:)*(cos(k'.*acos(2/Lf*t(:)'-1)));
        zpf = @(t) c(3,:)*(cos(k'.*acos(2/Lf*t(:)'-1)));
        s = @(t) sqrt(xpf(t(:)').^2+ypf(t(:)').^2+zpf(t(:)').^2);
        % Normalize
        xp = @(t) xpf(t)./(s(t));
        yp = @(t) ypf(t)./(s(t));
        zp = @(t) zpf(t)./(s(t));
        % Put 16 pts on the fiber
        s0 = chebpts(K,[0 Lf],1);
        % Compute fiber positions
        opts = odeset('RelTol',5e-14,'AbsTol',5e-14);
        [~,x] = ode45(@(t,x) xp(t),[0;s0],0,opts);
        [~,y] = ode45(@(t,y) yp(t),[0;s0],0,opts);
        [~,z] = ode45(@(t,z) zp(t),[0;s0],0,opts);
        fpts = [x(2:end) y(2:end) z(2:end)];
        forces = [xp(s0)' yp(s0)' zp(s0)'];
        % Compute coefficients
        n = length(s0);
        th=flipud(((2*(0:n-1)+1)*pi/(2*n))');
        Lmat = (cos((0:n-1).*th));
        xhats = Lmat \ fpts;
        fhats = Lmat \ forces;
    end
    % Check it's inextensible
    D=diffmat(length(s0),1,[0 Lf],'chebkind1');
    Xs = D*fpts;
    sqrt(sum((Xs).*(Xs),2));
    % Put some targets around it
    t=sort(rand(Ntarg,1)*Lf); % *still need to check what happens if slightly displaced from curve
    dist = rand(Ntarg,1)*0.19*Lf+0.01*Lf;  % LONG DISTANCES
    %dist = rand(Ntarg,1)*8*epsilon*Lf+2*epsilon*Lf; % SHORT DISTANCES
    dists(iFib,:)=dist;
    utang = [xp(t);yp(t);zp(t)]; % sloppy unit tangents
    v = randn(3,Ntarg); 
    vdotutang = sum(v.*utang,1); v = v - utang.*vdotutang;  % orthog v against the tangent
    v = v./sqrt(sum(v.*v,1));    % normalize all the v vecs
    % Resample curve at those t's
    thT=acos(2*t/Lf -1);
    RsTmat = (cos((0:K-1).*thT));
    curvt = RsTmat*xhats;
    X = curvt(:,1) + dist.*v(1,:)';   % displace by v vecs from pts on curve
    Y = curvt(:,2) + dist.*v(2,:)';
    Z = curvt(:,3) + dist.*v(3,:)';
    % Targets in a mesh grid
    % Compute reference answer 
    % Upsample position to 6000 points and do directly
    [sLg,wLg]=chebpts(6000,[0 Lf],1);
    thLg=acos(2*sLg/Lf -1);
    ptsUP = (cos((0:K-1).*thLg))*xhats;
    fsUP = (cos((0:K-1).*thLg))*fhats; 
    % Compute centerline velocity
    % Local part
    for iT=1:length(X)
        [uref1, uref2, uref3] = quadsum(ptsUP(:,1), ptsUP(:,2), ptsUP(:,3), wLg, ...
            fsUP(:,1),fsUP(:,2),fsUP(:,3),[X(iT) Y(iT) Z(iT)], 6000, epsilon,Lf);
        % Now compute the answer using the near field routine
        [utarg,qtype] = nearFieldNoCL([X(iT) Y(iT) Z(iT)],fpts(:,1),fpts(:,2),fpts(:,3),forces(:,1),...
            forces(:,2),forces(:,3),Lf,epsilon,mu,0);
        er(iFib,iT) = compute_error(uref1, uref2, uref3, utarg(1),utarg(2),utarg(3));
        qtypes(iFib,iT) = qtype;
    end
end
                                       
function [q1, q2, q3] = quadsum(xj, yj, zj, wj, f1j, f2j, f3j, targ, n, epsilon,Lf)
    q1 = 0; q2 = 0; q3 = 0;    
    for k=1:n
        r1 = xj(k)-targ(1);
        r2 = yj(k)-targ(2);
        r3 = zj(k)-targ(3);                               
        [u1, u2, u3] = slender_body_kernel(r1, r2, r3, f1j(k), f2j(k), f3j(k), ...
                                           epsilon*Lf*sqrt(2)*sqrt(exp(3)/24));
        q1 = q1 + u1*wj(k);
        q2 = q2 + u2*wj(k);
        q3 = q3 + u3*wj(k);                    
    end
end

% Stokeslet + e^2/2*doublet
function [u1, u2, u3] = slender_body_kernel(r1, r2, r3, f1, f2, f3, e)
    R = sqrt(r1.^2 + r2.^2 + r3.^2);
    R3 = R.*R.*R;
    R5 = R3.*R.*R;
    rdotf = r1.*f1 + r2.*f2 + r3.*f3;
    u1 = f1./R + r1.*rdotf./R3 + e^2/2*(f1./R3 - 3*r1.*rdotf./R5);
    u2 = f2./R + r2.*rdotf./R3 + e^2/2*(f2./R3 - 3*r2.*rdotf./R5);
    u3 = f3./R + r3.*rdotf./R3 + e^2/2*(f3./R3 - 3*r3.*rdotf./R5);    
end

function errmax = compute_error(uref1, uref2, uref3, q1, q2, q3)
    unorm = norm([uref1(:);uref2(:);uref3(:)], inf);
    err1 = abs(uref1-q1) ./ unorm;
    err2 = abs(uref2-q2) ./ unorm;
    err3 = abs(uref3-q3) ./ unorm;
    errmax = max(max(err1, err2), err3);
end


