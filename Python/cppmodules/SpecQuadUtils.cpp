#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "coeffs.cpp"

#ifdef VERBOSE
#define VERBINFO(...) printf(__VA_ARGS__)
#else
#define VERBINFO(...) /* nothing */
#endif

#define MAX_EXPANSION_ORDER 64


std::complex<double> rootfinder_initial_guess(const std::vector<double> &tj,
                                        const std::vector<double> &xj,
                                        const std::vector<double> &yj,
                                        const std::vector<double> &zj,
                                        int n, double x0, double y0, double z0)
{
    // Find two closest point
    double Rsqmin1 = INFINITY;
    double Rsqmin2 = INFINITY;
    int imin1, imin2;
    for (int i=0; i<n; i++)
    {
        double dx = xj[i]-x0;
        double dy = yj[i]-y0;
        double dz = zj[i]-z0;
        double Rsq = dx*dx + dy*dy + dz*dz;
        if (Rsq < Rsqmin1)
        {
            Rsqmin2 = Rsqmin1;
            imin2 = imin1;
            Rsqmin1 = Rsq;
            imin1 = i;
        }
        else if (Rsq < Rsqmin2)
        {
            Rsqmin2 = Rsq;
            imin2 = i;
        }
    }
    // Now compute initial guess
    int i1 = imin1;
    int i2 = imin2;
    double t1 = tj[i1];
    double t2 = tj[i2];
    double p[3];
    p[1] = xj[i1]-xj[i2];
    p[2] = yj[i1]-yj[i2];
    p[3] = zj[i1]-zj[i2];
    double pnorm = sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3]);
    double r[3];
    r[1] = x0-xj[i1];
    r[2] = y0-yj[i1];
    r[3] = z0-zj[i1];
    double rnormsq = r[1]*r[1] + r[2]*r[2] + r[3]*r[3];  
    double rdotp = r[1]*p[1] + r[2]*p[2] + r[3]*p[3];
    double a = (t1-t2)*rdotp/(pnorm*pnorm);
    double b = sqrt(rnormsq-rdotp*rdotp/(pnorm*pnorm)) * (t1-t2)/pnorm;
    std::complex<double> tinit(t1+a,b);
    return tinit;
}

std::complex<double> eval_cheb(const std::vector<double> &fhat,
                               std::complex<double> x, int N){
    // Compute the complex theta
    std::complex <double> theta = std::acos(x);
    std::complex <double> val(0,0);
    // Series
    for (int i=0; i < N; i++){
        val+=fhat[i]*std::cos(1.0*i*theta);
    }
    return val;
}


std::tuple<std::complex <double>, int> rootfinder(const std::vector<double> &xhat,
               const std::vector<double> &yhat,
               const std::vector<double> &zhat,
               const std::vector<double> &dxhat,
               const std::vector<double> &dyhat,
               const std::vector<double> &dzhat,
               int n,
               double x0,
               double y0,
               double z0,
               std::complex<double> tinit)
{
    // Find roots using Newton and Muller
    std::complex <double> t = tinit;
    double tol = 1e-10;
    int maxiter_newton = 10;
    int maxiter_muller = 10;
    // === Newton
    // Setup history variables (needed in Muller)
    std::complex <double> Fp, tp, Fpp, tpp;
    std::complex <double> dt, F, Fprime;
    int converged = 0;
    int iter;
    double absres;
    
    for (iter=0; iter<maxiter_newton; iter++)
    {
        std::complex <double> x, y, z, xp, yp, zp;
        std::complex <double> dx, dy, dz;
        // Chebyshev eval
        x = eval_cheb(xhat, t, n);
        y = eval_cheb(yhat, t, n);
        z = eval_cheb(zhat, t, n);
        xp = eval_cheb(dxhat, t, n);
        yp = eval_cheb(dyhat, t, n);
        zp = eval_cheb(dzhat, t, n);
        // Compute F and F'
        dx = x-x0;
        dy = y-y0;
        dz = z-z0;
        F = dx*dx + dy*dy + dz*dz;
        Fprime = 2.0*(dx*xp + dy*yp + dz*zp);
        dt = -F/Fprime;
        // Update history
        tpp = tp;
        Fpp = Fp;
        Fp = F;
        tp = t;
        // Update root
        t = t+dt;
        absres = std::abs(dt);
        if (absres < tol)
        {
            converged = 1;
            break;
        }
    }
    if (converged==1)
    {
        VERBINFO("Newton converged in %d iterations.\n", iter);
        //std::cout << "The root " << t << "\n";
        return std::make_tuple(t,converged);
    }
    
    // === Muller
    VERBINFO("Newton did not converge after %d iterations (abs(dt)=%g), switching to Muller\n", iter, absres);
    converged = 0;
    for (iter=0; iter<maxiter_muller; iter++)
    {
        std::complex <double> x, y, z;
        std::complex <double> dx, dy, dz;
        std::complex <double> q, A, B, C, d1, d2;
        // Chebyshev eval
        x = eval_cheb(xhat, t, n);
        y = eval_cheb(yhat, t, n);
        z = eval_cheb(zhat, t, n);
        dx = x-x0;
        dy = y-y0;
        dz = z-z0;
        F = dx*dx + dy*dy + dz*dz;
        // Mullers method
        q = (t-tp)/(tp - tpp);
        A = q*F - q*(q+1.0)*Fp + q*q*Fpp;
        B = (2.0*q+1.0)*F - (1.0+q)*(1.0+q)*Fp + q*q*Fpp;
        C =(1.0+q)*F;
        d1 = B + std::sqrt(B*B-4.0*A*C);
        d2 = B - std::sqrt(B*B-4.0*A*C);
        if (std::abs(d1) > std::abs(d2))
        dt = -(t-tp)*2.0*C/d1;
        else
        dt = -(t-tp)*2.0*C/d2;
        // Update history
        tpp = tp;
        Fpp = Fp;
        Fp = F;
        tp = t;
        // Update root
        t = t+dt;
        absres = std::abs(dt);
        if (absres < tol)
        {
            converged = 1;
            break;
        }
    }    
    if (converged){
        VERBINFO("Muller converged in %d iterations.\n", iter);
    } else{
    VERBINFO("Muller did not converge after %d iterations. abs(dt)=%g",
             iter, absres);
    }
    return std::make_tuple(t,converged);
}

std::vector<double> rsqrt_pow_integrals(std::complex<double> z, int N)
{
    std::vector<double> I1(N), I3(N), I5(N);
    double zr = z.real();
    double zi = z.imag();
    // (t-zr)^2+zi^2 = t^2-2*zr*t+zr^2+zi^2 = t^2 + b*t + c
    double b = -2*zr;
    double c = zr*zr+zi*zi;
    double d = zi*zi; // d = c - b^2/4;
    
    //u1 = sqrt(1 - b + c);
    //u2 = sqrt(1 + b + c);
    // The form below gets *much* better accuracy than ui = sqrt(ti^2 + b*ti + c)
    double u1 = sqrt((1+zr)*(1+zr) + zi*zi);
    double u2 = sqrt((1-zr)*(1-zr) + zi*zi);
    
    // Compute I1
    // k=1
    // Series expansion inside rhombus
    if (4*std::abs(zi) < 1-std::abs(zr))
    {
        double arg1 = (1-std::abs(zr))*eval_series(coeffs_I1, 1-std::abs(zr), zi, 11);
        double arg2 = 1+std::abs(zr) + sqrt((1+std::abs(zr))*(1+std::abs(zr)) + zi*zi);
        I1[0] = log(arg2)-log(arg1);
    }
    else
    {
        // Evaluate after substitution zr -> -|zr|
        double arg1 = -1+std::abs(zr) + sqrt((-1+std::abs(zr))*(-1+std::abs(zr)) + zi*zi);
        double arg2 =  1+std::abs(zr) + sqrt((1+std::abs(zr))*(1+std::abs(zr)) + zi*zi);
        I1[0] = log(arg2)-log(arg1);
    }
    // k>1
    if (N>1)
    I1[1] = u2-u1 - b/2*I1[0];
    double s = 1.0;
    for (int n=2; n<N; n++) {
        s = -s; // (-1)^(n-1)
        I1[n] = (u2-s*u1 + (1-2*n)*b/2*I1[n-1] - (n-1)*c*I1[n-2])/n;
    }
    
    // Compute I3
    // Series is needed in cones extending around real axis from
    // interval endpoints
    double w = fmin(std::abs(1+zr), std::abs(1-zr)); // distance interval-singularity
    zi = std::abs(zi);
    double zi_over_w = zi/w;
    bool in_cone = (zi_over_w < 0.6);
    bool outside_interval = (std::abs(zr)>1);
    bool use_series = (outside_interval & in_cone);
    if (!use_series)
    {
        I3[0] = (b+2)/(2*d*u2) - (b-2)/(2*d*u1);
    }
    else
    {
        // Power series for shifted integral
        // pick reasonable number of terms
        int Ns;
        if (zi_over_w < 0.01)
        Ns = 4;
        else if (zi_over_w < 0.1)
        Ns = 10;
        else if (zi_over_w < 0.2)
        Ns = 15;
        else // zi_over_w < 0.6
        Ns = 30;
        double x1 = -1-zr;
        double x2 = 1-zr;
        double Fs1 = std::abs(x1)/(x1*x1*x1) * (-0.5 + eval_series(coeffs_I3, x1, zi, Ns));
        double Fs2 = std::abs(x2)/(x2*x2*x2) * (-0.5 + eval_series(coeffs_I3, x2, zi, Ns));
        I3[0] = Fs2 - Fs1;
    }
    if (N>1)
    I3[1] = 1/u1-1/u2 - b/2*I3[0];
    for (int n=2; n<N; n++)
    I3[n] = I1[n-2] - b*I3[n-1] - c*I3[n-2];
    
    // Compute I5
    // Here too use power series for first integral, in cone around real axis
    in_cone = (zi_over_w < 0.7);
    use_series = (outside_interval && in_cone);
    if (!use_series)
    {
        I5[0] = (2+b)/(6*d*u2*u2*u2) - (-2+b)/(6*d*u1*u1*u1) + 2/(3*d)*I3[0];
    }
    else
    {
        // Power series for shifted integral
        int Ns;
        if (zi_over_w < 0.01)
        Ns = 4;
        else if (zi_over_w < 0.2)
        Ns = 10;
        else if (zi_over_w < 0.5)
        Ns = 24;
        else if (zi_over_w < 0.6)
        Ns = 35;
        else // zi/w < 0.7
        Ns = 50;
        double x1 = -1-zr;
        double x2 = 1-zr;
        double Fs1 = 1/(x1*x1*x1*std::abs(x1)) *(-0.25 + eval_series(coeffs_I5, x1, zi, Ns));
        double Fs2 = 1/(x2*x2*x2*std::abs(x2)) *(-0.25 + eval_series(coeffs_I5, x2, zi, Ns));
        I5[0] = Fs2 - Fs1;
    }
    
    if (N>1)
    {
        // Second integral computed using shifted version, and then corrected by first
        // integral (which is correctly evaluated using power series)
        // This is analogous to the formula for I3(1), but was overlooked by Tornberg & Gustavsson
        I5[1] = 1/(3*u1*u1*u1) - 1/(3*u2*u2*u2)  - b/2*I5[0];
    }
    
    for (int n=2; n<N; n++)
    I5[n] = I3[n-2] - b*I5[n-1] - c*I5[n-2];
    // Group weights together to be returned
    I1.insert(std::end(I1), std::begin(I3), std::end(I3));
    I1.insert(std::end(I1), std::begin(I5), std::end(I5));
    return I1;
}


PYBIND11_MODULE(SpecQuadUtils, m) {
    m.doc() = "C++ functions for the special quadrature"; // optional module docstring

    m.def("rf_iguess", &rootfinder_initial_guess, "Initial guess for the rootfinder");
    m.def("rootfinder", &rootfinder, "Find the root using Chebyshev series");
    m.def("spec_ints",&rsqrt_pow_integrals, "Integrals for special quadrature");
}
