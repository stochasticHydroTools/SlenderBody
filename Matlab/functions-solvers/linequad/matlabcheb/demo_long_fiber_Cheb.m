% Evaluate Stokes slender body potential from closed-loop squiggle curve.
% Computes using interpolatory quadrature, and compares to semi-smart adaptive
% quadrature, which is reasonably fast and seems to compute the right thing.
% Slow computations of reference using integral() is also available.
%
% Matlab code attempts to be fast by avoiding inner loop vector ops, so that
% we can compute meaningful running times.

% Alex variant also includes all targs close to curve test - see targtype.

function [specquad_errmax,dir16_errmax,dir32_errmax,unid16, unid32,groots, specstats,C] = ...
    demo_long_fiber_Cheb(Xscos,t,v,dist) % Function makes Matlab JIT compile better

%% Default setup
slender_eps = 1e-3;

nquad = 32;
rho = 3; % Interpolatory quadrature limit rule **NEED TO PLAY WITH THIS**
%rho = 1 % Disable specquad

% When running near points
%% Case setup
fixedpan = 1; % number of panels

%% Run
% Setup fiber
Lf=2;
t=Lf*t;
[xp, yp, zp] = squiggle(Xscos,Lf);
s = @(t) sqrt(xp(t).^2 + yp(t).^2 + zp(t).^2);

% Sample at Chebyshev points based on curvature
s0 = chebpts(16,[0 Lf],1); % using first kind grid
XCheb = [x(s0,xp) x(s0,yp) x(s0,zp)];
XCheb = XCheb(2:end,:);
D2 = diffmat(16,2,[0 Lf],'chebkind1');
% Compute curvature
Xss = D2*XCheb;
C = sqrt(sum(Xss.*Xss,2));
% Normalize by curvature of a circle 
Ccirc = 1/(sqrt(2))*2*pi/Lf;
C = C/Ccirc;

% Artificial density (trivial)
f1 = @(t) xp(t);
f2 = @(t) yp(t);
f3 = @(t) zp(t);

% Discretize
%n=(cross(rand(3,1),[xp(0.5) yp(0.5) zp(0.5)]));
% The key is to use a guess for the closest root here. 
%xclose = x(0.5,xp); xclose=xclose(end);
%yclose = x(0.5,yp); yclose=yclose(end);
%zclose = x(0.5,zp); zclose=zclose(end);
%closetar = [xclose yclose zclose] + dist*(n/norm(n));
%troot = 0.5+dist*1i;
% Pick whatever discretization is finer
% [tj1, wj1, npan1] = adaptive_panelization(s, nquad, 1e-6);
[tj, wj, npan] = makePanels(nquad,fixedpan,1,Lf);
fprintf('nquad=%d, npan=%d\n', nquad, npan);
xj = x(tj,xp)'; yj = x(tj,yp)'; zj = x(tj,zp)';
xj(1)=[]; yj(1)=[]; zj(1)=[];
sj = s(tj);
f1j = f1(tj); f2j = f2(tj); f3j = f3(tj);

% targtype='n';  % all targs near curve, in random normal directions a fixed dist away
% Ne = 5e3;  % # targs
%t = sort(rand(1,Ne)); % these are now inputs
%v = randn(3,Ne);      % now an input 
utang = [xp(t);yp(t);zp(t)]./s(t); % sloppy unit tangents
vdotutang = sum(v.*utang,1); v = v - utang.*vdotutang;  % orthog v against the tangent
v = v./sqrt(sum(v.*v,1));    % normalize all the v vecs
xt = x(t,xp)'; yt = x(t,yp)'; zt = x(t,zp)';
xt(1)=[]; yt(1)=[]; zt(1)=[];
X = xt + dist*v(1,:);   % displace by v vecs from pts on curve
Y = yt + dist*v(2,:);
Z = zt + dist*v(3,:);

% Adaptive quadrature
disp('* Reference: lots of points')
% Make sure that we get a different discretization for reference computations,
% otherwise errors get artifically small.
Nref=6000;
[tj_ref,wj_ref] = chebpts(Nref,[0 Lf],1);
xtj = x(tj_ref,xp)'; ytj = x(tj_ref,yp)'; ztj = x(tj_ref,zp)';
xtj(1)=[]; ytj(1)=[]; ztj(1)=[];
[uref1, uref2, uref3] = quadsum(xtj, ytj, ztj, s(tj_ref), wj_ref, f1(tj_ref), f2(tj_ref), f3(tj_ref),...
                                           X, Y, Z, Nref, slender_eps);
% Compute special quadrature
disp(' ')
disp('* Interpolatory quadrature')
[specquad1,specquad2,specquad3, specstats,groots] = interpolatory_quadrature(...
    xj, yj, zj, sj, wj, f1j, f2j, f3j, X, Y, Z, nquad, rho, slender_eps);

specquad_errmax = compute_error(uref1, uref2, uref3, specquad1, specquad2, specquad3);
specquad_errmaxmax = max(specquad_errmax(:))

% Compute DIRECT quadrature at 16 type 1 Chebyshev nodes. 
[tj,wj] = chebpts(16,[0 Lf],1);
xj = x(tj,xp)'; yj = x(tj,yp)'; zj = x(tj,zp)';
xj(1)=[]; yj(1)=[]; zj(1)=[];
sj = s(tj);
f1j = f1(tj); f2j = f2(tj); f3j = f3(tj);
[dir1, dir2, dir3] = quadsum(xj, yj, zj, sj, wj, f1j, f2j, f3j, X, Y, Z, 16, slender_eps);
dir16_errmax = compute_error(uref1, uref2, uref3, dir1, dir2, dir3);
dir16_errmaxmax = max(dir16_errmax(:))

% Measure the min distance of targets from 16, 32 uniform points
u16 = 0:Lf/15:Lf;
thuni = acos(2*u16/Lf-1)';
n = 16;
th=flipud(((2*(0:n-1)+1)*pi/(2*n))');
Lmat = (cos((0:n-1).*th));
Unifmat = (cos((0:n-1).*thuni));
xu16 = Unifmat*(Lmat \ [xj' yj' zj']);
u32 = 0:Lf/31:Lf;
xu32 = [x(u32(2:end),xp) x(u32(2:end),yp) x(u32(2:end),zp)];
Ne=100;
unid16=zeros(Ne,1);
unid32=zeros(Ne,1);
for iTarg=1:Ne
    disp16 = xu16-[X(iTarg) Y(iTarg) Z(iTarg)];
    disp32 = xu32-[X(iTarg) Y(iTarg) Z(iTarg)];
    unid16(iTarg)=sqrt(min(sum(disp16.*disp16,2)));
    unid32(iTarg)=sqrt(min(sum(disp32.*disp32,2)));
end


% Compute DIRECT quadrature at 32 type 1 Chebyshev nodes. 
[tj,wj] = chebpts(32,[0 Lf],1);
xj = x(tj,xp)'; yj = x(tj,yp)'; zj = x(tj,zp)';
xj(1)=[]; yj(1)=[]; zj(1)=[];
sj = s(tj);
f1j = f1(tj); f2j = f2(tj); f3j = f3(tj);
[dir1, dir2, dir3] = quadsum(xj, yj, zj, sj, wj, f1j, f2j, f3j, X, Y, Z, 32, slender_eps);
dir32_errmax = compute_error(uref1, uref2, uref3, dir1, dir2, dir3);
dir32_errmaxmax = max(dir32_errmax(:))

sfigure(1);
clf; 
plot3(xj, yj, zj, '.-k')
%scatter3(X,Y,Z)
axis equal tight vis3d
grid on
box on

sfigure(2);
clf; 
semilogy(t,specquad_errmax,'.'); xlabel('t'); ylabel('err')
title('Special quadrature')

    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function errmax = compute_error(uref1, uref2, uref3, q1, q2, q3)
    unorm = norm([uref1(:);uref2(:);uref3(:)], inf);
    err1 = abs(uref1-q1) ./ unorm;
    err2 = abs(uref2-q2) ./ unorm;
    err3 = abs(uref3-q3) ./ unorm;
    errmax = max(max(err1, err2), err3);
end

function h = sfigure(f)
    if nargin==0
        h = figure();
    elseif ishandle(f)
        set(0, 'CurrentFigure', f);
    else
        h = figure(f);
    end
end


function [q1, q2, q3] = quadsum(xj, yj, zj, sj, wj, f1j, f2j, f3j, x, y, z, n, ...
                                slender_eps)
    q1 = 0; q2 = 0; q3 = 0;    
    for k=1:n
        r1 = xj(k)-x;
        r2 = yj(k)-y;
        r3 = zj(k)-z;                               
        [u1, u2, u3] = slender_body_kernel(r1, r2, r3, f1j(k), f2j(k), f3j(k), ...
                                           slender_eps*2*sqrt(2));
        q1 = q1 + u1*sj(k)*wj(k);
        q2 = q2 + u2*sj(k)*wj(k);
        q3 = q3 + u3*sj(k)*wj(k);                    
    end
end

%%%%% INTERPOLATORY QUADRATURE

function [specquad1,specquad2,specquad3,stats,groots] = interpolatory_quadrature(...
    xj, yj, zj, sj, wj, f1j, f2j, f3j, X, Y, Z, nquad, rho, slender_eps)
    [specquad1,specquad2,specquad3] = deal(zeros(size(X)));
    npan = numel(xj)/nquad;    
    [tgl, wgl] = chebpts(nquad,[-1 1],1);
    time_weights = 0;
    kerevals_near = 0;
    maintic = tic();
    time_ker_near = 0;
    time_far = 0;
    for j=1:npan
        % Load panel
        idx = (1:nquad) + nquad*(j-1);
        xjpan = xj(idx);
        yjpan = yj(idx);
        zjpan = zj(idx);
        sjpan = sj(idx);
        wjpan = wj(idx);
        f1jpan = f1j(idx);
        f2jpan = f2j(idx);
        f3jpan = f3j(idx);    
        % Compute quadrature weights
        [all_w1, all_w3, all_w5, specquad_needed,groots] = line3_near_weights_Cheb(tgl, wgl, xjpan, yjpan, zjpan, ...
                                                          X, Y, Z, rho);
        
        stats = specquad_needed;
        % Evaluate each panel-to-point pair
        for i=1:numel(X)    
            Xi = X(i);
            Yi = Y(i);
            Zi = Z(i);
            q1 = 0; q2 = 0; q3 = 0;
            if specquad_needed(i)
                for k=1:nquad
                    r1 = xjpan(k)-Xi;
                    r2 = yjpan(k)-Yi;
                    r3 = zjpan(k)-Zi;
                    [u1R1, u1R3, u1R5, u2R1, u2R3, u2R5, u3R1, u3R3, u3R5] ...
                        = slender_body_kernel_split(r1, r2, r3, f1jpan(k), f2jpan(k), f3jpan(k), ...
                                                    slender_eps*2*sqrt(2));
                    q1 = q1 + ...
                         all_w1(k,i)*sjpan(k)*u1R1 + ...
                         all_w3(k,i)*sjpan(k)*u1R3 + ...
                         all_w5(k,i)*sjpan(k)*u1R5;                
                    q2 = q2 + ...
                         all_w1(k,i)*sjpan(k)*u2R1 + ...
                         all_w3(k,i)*sjpan(k)*u2R3 + ...
                         all_w5(k,i)*sjpan(k)*u2R5;                
                    q3 = q3 + ...
                         all_w1(k,i)*sjpan(k)*u3R1 + ...
                         all_w3(k,i)*sjpan(k)*u3R3 + ...
                         all_w5(k,i)*sjpan(k)*u3R5;                                
                end      
                % Rescale (weights are for [-1,1])
                q1 = q1/2*sum(wjpan);
                q2 = q2/2*sum(wjpan);
                q3 = q3/2*sum(wjpan);            
            else            
                [q1, q2, q3] = quadsum(xjpan, yjpan, zjpan, sjpan, wjpan, f1jpan, f2jpan, f3jpan, ...
                                       Xi, Yi, Zi, nquad, slender_eps);
            end
            specquad1(i) = specquad1(i) + q1;        
            specquad2(i) = specquad2(i) + q2;        
            specquad3(i) = specquad3(i) + q3;                
        end
    end
end    

%%%%% GENERAL FUNCTIONS

function [xp, yp, zp] = squiggle(c,L)
%     xp = @(t) sin(12*pi*t(:)');
%     yp = @(t) cos(12*pi*t(:)');
%     zp = @(t) ones(size(t(:)'));
    K = 15;   % max Chebyshev mode in derivatives
    k = 0:K-1;
%    nk = numel(k);
%    rng(0);
%    ampl = exp(-k/nk*10);   % Exponential decay to 4 digits
%    rng('shuffle')
%    c = randn(3,nk) .* ampl;    % cmplx F coeffs in each coord
    xpf = @(t) c(1,:)*(cos(k'.*acos(2/L*t(:)'-1)));
    ypf = @(t) c(2,:)*(cos(k'.*acos(2/L*t(:)'-1)));
    zpf = @(t) c(3,:)*(cos(k'.*acos(2/L*t(:)'-1)));
    s = @(t) sqrt(xpf(t(:)').^2+ypf(t(:)').^2+zpf(t(:)').^2);
    % Normalize
    xp = @(t) xpf(t)./(s(t));
    yp = @(t) ypf(t)./(s(t));
    zp = @(t) zpf(t)./(s(t));
end

function x = x(t,xp)
    % Integrate to get position
    opts = odeset('RelTol',5e-14,'AbsTol',5e-14);
    [~,x] = ode45(@(t,x) xp(t),[0 t(:)'],0,opts);
    if (numel(t)==1)
        x=x(end);
    end
end

function [tj, wj, npan] = makePanels(nquad,fixedpan,cheb,L)
    if (cheb)
        [tgl, wgl] = chebpts(nquad,[0 1],1);
    else
        [tgl, wgl] = legendre.gauss(nquad); 
        tgl = (tgl+1)/2; wgl = wgl/2; % [0,1]
    end
    npan = fixedpan;
    edges = 0:L/npan:L;
    [tj, wj] = deal(zeros(npan*nquad, 1));
    for i=1:npan
        idx = (1:nquad) + nquad*(i-1);
        ta = edges(i);
        tb = edges(i+1);
        dt = tb-ta;
        tj(idx) = ta + tgl*dt;
        wj(idx) = wgl*dt;
    end
end

%% SLENDER BODY KERNEL
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

function [u1R1, u1R3, u1R5, u2R1, u2R3, u2R5, u3R1, u3R3, u3R5] = ...
        slender_body_kernel_split(r1, r2, r3, f1, f2, f3, e)
    R = sqrt(r1.^2 + r2.^2 + r3.^2);
    R3 = R.*R.*R;
    R5 = R3.*R.*R;    
    rdotf = r1.*f1 + r2.*f2 + r3.*f3;
    u1R1 = f1./R;
    u2R1 = f2./R;
    u3R1 = f3./R;
    u1R3 = (r1.*rdotf + e^2/2*f1)./R3;
    u2R3 = (r2.*rdotf + e^2/2*f2)./R3;
    u3R3 = (r3.*rdotf + e^2/2*f3)./R3;    
    u1R5 = -3*e^2/2*r1.*rdotf./R5;
    u2R5 = -3*e^2/2*r2.*rdotf./R5;
    u3R5 = -3*e^2/2*r3.*rdotf./R5;        
end
