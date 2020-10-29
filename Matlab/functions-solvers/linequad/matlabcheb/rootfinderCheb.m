function [troot, converged] = rootfinderCheb(xhat, yhat, zhat, x0, y0, z0, tinit)
% Find roots using Newton and Muller
    VERBOSE = 0;
    
    t = tinit;
    n = numel(xhat);
    tol = 1e-4;
    maxiter_newton = 10;
    maxiter_muller = 10;    
    % === Newton
    % Setup history variables (needed in Muller)
    Fp = 0; tp = 0; Fpp = 0; tpp = 0;
    converged = false;
    dxhat = chebCoeffDiff(xhat,n);
    dyhat = chebCoeffDiff(yhat,n);
    dzhat = chebCoeffDiff(zhat,n);
    for iter=1:maxiter_newton      
        th_t = acos(t); %works even for complex
        cvals = cos((0:n-1)*th_t); % Chebyshev polynomials evaluated at t
        F = (cvals*xhat-x0)^2 + (cvals*yhat-y0)^2 + (cvals*zhat-z0)^2;
        Fprime = 2*(cvals*xhat-x0)*(cvals*dxhat) + ...
                 2*(cvals*yhat-y0)*(cvals*dyhat) + ...
                 2*(cvals*zhat-z0)*(cvals*dzhat);
        dt = -F/Fprime;                        
        % Update history
        tpp = tp;
        Fpp = Fp;        
        Fp = F;
        tp = t;       
        % Update root
        t = t+dt;
        absres = abs(dt);
        if absres < tol
            converged = true;
            break
        end
    end
    if converged
        if VERBOSE
            fprintf('Newton converged in %d iterations.\n', iter);
        end
        troot = t;
        return;
    end    
    % === Mulleri
    if VERBOSE
        fprintf('Newton did not converge after %d iterations (abs(dt)=%g), switching to Muller\n',...
                iter, absres);
    end
    converged = false;
    for iter=1:maxiter_muller
        th_t = acos(t); %works even for complex
        cvals = cos((0:n-1)*th_t); % Chebyshev polynomials evaluated at t
        F = (cvals*xhat-x0)^2 + (cvals*yhat-y0)^2 + (cvals*zhat-z0)^2;            
        % Mullers method
        q = (t-tp)/(tp - tpp);
        A = q*F - q*(q+1)*Fp + q^2*Fpp;
        B = (2*q+1)*F - (1+q)^2*Fp + q^2*Fpp;
        C =(1+q)*F;            
        denoms = [B+sqrt(B^2-4*A*C), B-sqrt(B^2-4*A*C)];
        [~,idx] = max(abs(denoms));            
        dt = -(t-tp)*2*C/denoms(idx);
        % Update history
        tpp = tp;
        Fpp = Fp;        
        Fp = F;
        tp = t;       
        % Update root
        t = t+dt;
        absres = abs(dt);            
        if absres < tol
            converged = true;
            break
        end
    end
    if VERBOSE    
        if converged

            fprintf('Muller converged in %d iterations.\n', iter);
        else
            warning('Muller did not converge after %d iterations. abs(dt)=%g', iter, absres);
            target_point = [x0 y0 z0]
        end            
    end
    if (~converged)
         warning('Root finder did not converge after %d iterations. abs(dt)=%g', iter, absres);
    end
    troot=t;
end
