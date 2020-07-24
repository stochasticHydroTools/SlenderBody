function tinit = rootfinder_initial_guess(tj, xj, yj, zj, x0, y0, z0)
% Compute initial guess for 3D root finder
    
    guess_order = 2;
    
    if guess_order == 1
        % First order guess:
        % Initial guess based on location from closest point
        R2j = ( (xj-x0).^2 + (yj-y0).^2 + (zj-z0).^2 );    
        [Rmin, imin] = min(R2j);
        panel_length = sqrt( (xj(1)-xj(end)).^2 + (yj(1)-yj(end)).^2 + (zj(1)-zj(end)).^2 );
        ainit = tj(imin);
        binit = Rmin*2/panel_length;
        tinit = ainit + 1i*binit;
    else
        % Second order guess;
        % Initial guess based on orientation relative to two closest points
        % Find indices of two closest points
        R2j = ( (xj-x0).^2 + (yj-y0).^2 + (zj-z0).^2 );    
        [~, isort] = mink(R2j, 2);
        i1 = isort(1);
        i2 = isort(2);
        % Locations of closest points in R3 and C
        v1 = [xj(i1) yj(i1) zj(i1)];
        v2 = [xj(i2) yj(i2) zj(i2)];    
        t1 = tj(i1);
        t2 = tj(i2);
        p = v1-v2;
        pnorm = sqrt(p(1)^2 + p(2)^2 + p(3)^2);
        x = [x0 y0 z0];
        r = x-v1;
        rdotp = r(1)*p(1) + r(2)*p(2) + r(3)*p(3);
        a = (t1-t2)*rdotp/pnorm^2;
        b = sqrt(norm(r)^2-rdotp^2/pnorm^2)*(t1-t2)/pnorm;
        tinit = complex(t1 + a + 1i*b);
    end
end