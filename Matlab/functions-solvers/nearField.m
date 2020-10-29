% Evaluate the near field integrals using the flowchart we derived
% targ = (x,y,z) location of target
% (xy,yj,zj) = points on the fiber (assumed to be type 1 Chebyshev)
% Lf = length of fiber
% epsilon = r/Lf = slenderness ratio
function [utarg,qtype] = nearField(targ,xj,yj,zj,f1j,f2j,f3j,Lf,epsilon,mu,CLvel,ewald)
    % Resample to 16 uniform points
    b = exp(3/2)/4;
    n = numel(xj);
    th=flipud(((2*(0:n-1)+1)*pi/(2*n))');
    Lmat = (cos((0:n-1).*th));
    u16 = 0:Lf/15:Lf;
    thuni = acos(2*u16/Lf-1)';
    Umat = (cos((0:n-1).*thuni));
    xu16 = Umat*(Lmat \ [xj(:) yj(:) zj(:)]);
    % Perform a discrete minimization problem to estimate the distance
    % between the fiber and target
    dmin = min(sqrt(sum((xu16-targ).*(xu16-targ),2)));
    % First check: if the point is far enough to do direct with N = 16, do
    % it
    utargm = [0 0 0];
    if (dmin/Lf > 0.15*1.05)
        % Direct quadrature with N=16
        if (ewald)
            utarg = [0 0 0];
            qtype = -1;
            return;
        end
        x16 = [xj yj zj];
        f16 = [f1j f2j f3j];
        if (n~=16) % Resample to 16 if the input is a different # of points
            th16=flipud(((2*(0:16-1)+1)*pi/(2*16))');
            Lmat16 = (cos((0:n-1).*th16));
            x16 = Lmat16*(Lmat \ [xj yj zj]);
            f16 = Lmat16*(Lmat \ [f1j f2j f3j]);
        end
        [~,w16] = chebpts(16,[0 Lf],1);
        [ux, uy, uz] = quadsum(x16(:,1),x16(:,2),x16(:,3),w16, f16(:,1),f16(:,2),f16(:,3),targ, 16, epsilon,Lf);
        utarg = [ux uy uz]/(8*pi*mu);
        qtype = 1;
        return;
    end
    % Here we SUBTRACT THE FREE SPACE KERNEL WITH N =n if we were
    % doing Ewald
    if (ewald)
        [~,wn] = chebpts(n,[0 Lf],1);
        [ux, uy, uz] = quadsumRPY(xj,yj,zj,wn, f1j,f2j,f3j,targ, n, epsilon,Lf,mu);
        utargm = -[ux uy uz];
    end
    % We need at least 32 points from here on out. Resample to 32 points
    x32 = [xj yj zj];
    f32 = [f1j f2j f3j];
    th32=flipud(((2*(0:32-1)+1)*pi/(2*32))');
    Lmat32 = (cos((0:n-1).*th32));
    if (n~=32) % Resample to 32 if the input is a different # of points
        x32 = Lmat32*(Lmat \ [xj yj zj]);
        f32= Lmat32*(Lmat \ [f1j f2j f3j]);
    end
    [~,w32] = chebpts(32,[0 Lf],1);
    % Can we do direct quadrature with N = 32?
    if (dmin/Lf > 0.06*1.2)
        % Direct quadrature with N=32
        [ux, uy, uz] = quadsum(x32(:,1),x32(:,2),x32(:,3),w32, f32(:,1),f32(:,2),f32(:,3),targ, 32, epsilon,Lf);
        utarg = utargm+[ux uy uz]/(8*pi*mu);
        qtype =2;
        return;
    end
    % Now call the special quadrature to COMPUTE THE ROOT ONLY with N = 32
    xhat = (cos((0:31).*th32)) \ x32(:,1);
    yhat = (cos((0:31).*th32)) \ x32(:,2);
    zhat = (cos((0:31).*th32)) \ x32(:,3);
    % Cap expansion at 16 coefficients.
    % This seems to be more stable near boundary
    xhat = xhat(1:16);
    yhat = yhat(1:16);
    zhat = zhat(1:16);
    % Rho = 1.114 for 3 digits is the critical Bernstein ellipse radius
    % with N = 32 points
    rho = 1.114;
    t32 = chebpts(32,[-1 1],1);
    [~,w32] = chebpts(32,[0 Lf],1);
    % Rootfinding: initial guess
    tinit = rootfinder_initial_guess(t32, x32(:,1), x32(:,2), x32(:,3),...
        targ(1), targ(2), targ(3)); % O(n) per point
    if (bernstein_radius(tinit) > 1.5*rho) % direct quadrature if initial guess is too far
        [ux, uy, uz] = quadsum(x32(:,1),x32(:,2),x32(:,3),w32, f32(:,1),f32(:,2),f32(:,3),...
                               targ, 32, epsilon,Lf);
        utarg = utargm+[ux uy uz]/(8*pi*mu);
        qtype = 2;
        return;
    end
    % Run rootfinder 
    [troot, converged] = rootfinderCheb(xhat, yhat, zhat, ...
        targ(1),targ(2),targ(3), tinit); % O(n) per point 
    % Compute approximate distance
    tapprox = troot;
    if (real(troot) < -1)
        tapprox = -1+1i*imag(tapprox);
    elseif (real(troot) > 1)
        tapprox = 1+1i*imag(tapprox);
    end
    v=(cos((0:min(n,16)-1).*acos(tapprox)));
    gimag = norm(real([v*xhat v*yhat v*zhat]) -targ);
    %gimag = norm(imag([v*xhat v*yhat v*zhat]));
    bradius = bernstein_radius(troot);
    specquad_needed = converged & (bradius < rho);
    if (gimag/(epsilon*Lf) < 4*b)
        % Compute the centerline velocity at real(tapprox)
        thCL = acos(real(tapprox));
        RSCL = (cos((0:n-1).*thCL));
        CLnearest = RSCL*(Lmat \ CLvel);
        % Return the centerline velocity at real(tapprox)
        if (gimag/(epsilon*Lf) < 2*b)
            disp('Target close to fiber - setting velocity = CL vel')
            utarg = utargm+CLnearest;
            qtype=0;
            return;
        end
    end
    if (~specquad_needed) % Direct quadrature
        [ux, uy, uz] = quadsum(x32(:,1),x32(:,2),x32(:,3),w32, f32(:,1),f32(:,2),f32(:,3),...
                               targ, 32, epsilon,Lf);
        utarg = [ux uy uz]/(8*pi*mu);
        qtype = 2;
        if (gimag/(epsilon*Lf) < 4*b && converged) 
            % Linear combination with the centerline velocity
            dstar = gimag/(epsilon*Lf);
            fromCL = (4*b-dstar)/(2*b);
            fromFar = 1-fromCL;
            qtype = 0.5;
            utarg = fromFar*utarg + fromCL*CLnearest;
            disp('Target close to fiber - interpolating with CL vel')
        end
        utarg = utargm+utarg;
        return;
    end
    % If the hats don't decay exponetially, print warning and switch to
    % direct quadrature with lots of points
    if (sum(sum(abs([xhat(3:end,:) yhat(3:end,:) zhat(3:end,:)]) > exp(-0.55*(2:15))')) > 0)
        warning('Coefficients do not decay exponentially - switching to direct with lots of points')
        Nup = ceil(3/(2*log10(bradius)));
        %Nup = 1000;
        thUp=flipud(((2*(0:Nup-1)+1)*pi/(2*Nup))');
        LmatUp = (cos((0:n-1).*thUp));
        xup = LmatUp*(Lmat \ [xj yj zj]);
        fup = LmatUp*(Lmat \ [f1j f2j f3j]);
        [~,wup] = chebpts(Nup,[0 Lf],1);
        [ux, uy, uz] = quadsum(xup(:,1),xup(:,2),xup(:,3),wup, fup(:,1),fup(:,2),fup(:,3),...
                               targ, Nup, epsilon,Lf);
        utarg = [ux uy uz]/(8*pi*mu);
        qtype = Nup;
        if (gimag/(epsilon*Lf) < 4*b) 
            % Linear combination with the centerline velocity
            dstar = gimag/(epsilon*Lf);
            fromCL = (4*b-dstar)/(2*b);
            fromFar = 1-fromCL;
            qtype = 0.5;
            utarg = fromFar*utarg + fromCL*CLnearest;
            disp('Target close to fiber - interpolating with CL vel')
        end
        utarg = utargm+utarg;
        return;
    end 
    if (gimag/(epsilon*Lf) > 8.8) % Can just proceed with 1 panel of 32
        [w1, w3, w5] = rsqrt_pow_weights(t32, troot);
        % Evaluate each panel-to-point pair
        q1 = 0; q2 = 0; q3 = 0;
        for k=1:32
            r1 = x32(k,1)-targ(1);
            r2 = x32(k,2)-targ(2);
            r3 = x32(k,3)-targ(3);
            [u1R1, u1R3, u1R5, u2R1, u2R3, u2R5, u3R1, u3R3, u3R5] ...
                = slender_body_kernel_split(r1, r2, r3, f32(k,1), f32(k,2), f32(k,3), ...
                                            epsilon*Lf*sqrt(2)*sqrt(exp(3)/24));
            q1 = q1 +w1(k)*u1R1 + w3(k)*u1R3 + w5(k)*u1R5;                
            q2 = q2 +w1(k)*u2R1 + w3(k)*u2R3 + w5(k)*u2R5;                
            q3 = q3 +w1(k)*u3R1 + w3(k)*u3R3 + w5(k)*u3R5;                                
        end      
        % Rescale (weights are for [-1,1])
        q1 = q1/2*Lf;
        q2 = q2/2*Lf;
        q3 = q3/2*Lf;
        utarg = utargm+[q1 q2 q3]/(8*pi*mu);
        qtype = 3;
        return;
    end
    % Now the hard case: when the point is really close to the fiber. We
    % know 2.2 < gimag/(epsilon*Lf) < 8.8. Need 2 panels of 32 here
    % Upsample the curve to the 2 panels of 32
    t2p32 = [(t32+1)/2*Lf/2; Lf/2+(t32+1)/2*Lf/2]';
    th2pan = acos(2*t2p32/Lf-1)';
    TwoPanMat = (cos((0:32-1).*th2pan));
    Lmat32 = (cos((0:32-1).*th32));
    x2p32 = TwoPanMat*(Lmat32 \ x32);
    f2p32 = TwoPanMat*(Lmat32 \ f32);
    specquad1=0; specquad2 = 0; specquad3 = 0;
    for j=1:2 % Loop over 2 panels
        % Load panel
        idx = (1:32) + 32*(j-1);
        wjpan = w32/2;
        xjpan = x2p32(idx,1); yjpan = x2p32(idx,2); zjpan = x2p32(idx,3);
        f1jpan = f2p32(idx,1); f2jpan = f2p32(idx,2); f3jpan = f2p32(idx,3);    
        % Compute quadrature weights
        [w1, w3, w5, specquad_needed] = line3_near_weights_Cheb(t32, wjpan, xjpan, yjpan, zjpan, ...
                                                          targ(1),targ(2),targ(3), rho);
        % Evaluate each panel-to-point pair
        q1 = 0; q2 = 0; q3 = 0;
        if (specquad_needed)
            for k=1:32
                r1 = xjpan(k)-targ(1);
                r2 = yjpan(k)-targ(2);
                r3 = zjpan(k)-targ(3);
                [u1R1, u1R3, u1R5, u2R1, u2R3, u2R5, u3R1, u3R3, u3R5] ...
                    = slender_body_kernel_split(r1, r2, r3, f1jpan(k), f2jpan(k), f3jpan(k), ...
                                                epsilon*Lf*sqrt(2)*sqrt(exp(3)/24));
                q1 = q1 +w1(k)*u1R1 + w3(k)*u1R3 + w5(k)*u1R5;                
                q2 = q2 +w1(k)*u2R1 + w3(k)*u2R3 + w5(k)*u2R5;                
                q3 = q3 +w1(k)*u3R1 + w3(k)*u3R3 + w5(k)*u3R5;                             
            end      
            % Rescale (weights are for [-1,1]) - 2 panels so Lpanel = Lf/2
            q1 = q1/2*Lf/2;
            q2 = q2/2*Lf/2;
            q3 = q3/2*Lf/2;            
        else            
            [q1, q2, q3] = quadsum(xjpan, yjpan, zjpan, wjpan, f1jpan, f2jpan, f3jpan, ...
                                   targ, 32, epsilon,Lf);
        end
        specquad1 = specquad1 + q1;        
        specquad2 = specquad2 + q2;        
        specquad3 = specquad3 + q3;                
    end
    utarg = [specquad1 specquad2 specquad3]/(8*pi*mu);
    qtype = 4;
    if (gimag/(epsilon*Lf) < 4*b) 
        % Linear combination with the centerline velocity
        dstar = gimag/(epsilon*Lf);
        fromCL = (4*b-dstar)/(2*b);
        fromFar = 1-fromCL;
        qtype = 0.5;
        utarg = fromFar*utarg + fromCL*CLnearest;
        disp('Target close to fiber - interpolating with CL vel')
    end
    utarg = utargm+utarg;
end    

        
function [q1, q2, q3] = quadsum(xj, yj, zj, wj, f1j, f2j, f3j, targ, n, epsilon,Lf)
    q1 = 0; q2 = 0; q3 = 0;    
    for k=1:n
        r1 = xj(k)-targ(1);
        r2 = yj(k)-targ(2);
        r3 = zj(k)-targ(3);                               
        [u1, u2, u3] = slender_body_kernel(r1, r2, r3, f1j(k), f2j(k), f3j(k),epsilon*Lf*sqrt(2)*sqrt(exp(3)/24));
        q1 = q1 + u1*wj(k);
        q2 = q2 + u2*wj(k);
        q3 = q3 + u3*wj(k);                    
    end
end

function [q1, q2, q3] = quadsumRPY(xj, yj, zj, wj, f1j, f2j, f3j, targ, n, epsilon,Lf,mu)
    q1 = 0; q2 = 0; q3 = 0;
    a=exp(1.5)/4*epsilon*Lf;
    for k=1:n
        r1 = xj(k)-targ(1);
        r2 = yj(k)-targ(2);
        r3 = zj(k)-targ(3); 
        M = RPYTot([r1 r2 r3],a,mu);
        uthis = M*[f1j(k); f2j(k); f3j(k)];
        q1 = q1 + uthis(1)*wj(k);
        q2 = q2 + uthis(2)*wj(k);
        q3 = q3 + uthis(3)*wj(k);                    
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

