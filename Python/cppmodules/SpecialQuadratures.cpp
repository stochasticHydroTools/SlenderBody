#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "coeffs.cpp"
#include "types.h"
#include "VectorMethods.cpp"
#include "Chebyshev.cpp"

#ifdef VERBOSE
#define VERBINFO(...) printf(__VA_ARGS__)
#else
#define VERBINFO(...) /* nothing */
#endif

#define MAX_EXPANSION_ORDER 16 // cap Chebyshev expansions at 16 points
#define ROOT_FINDER_TOL 1e-4   // tolerance for the special quad root finder
#define MAX_RF_ITERS 10        // maximum number of iterations to find the root

// Global variable: the Bernstein radius
double rho_crit;
vec SpecialVandermonde;
int ipivVander [40]; // pivots for Vandermonde (maximum of 40)
bool useByorck = false;

// LAPACK CALLS
extern "C" void dgetrf_(int* dim1, int* dim2, double* a, int* lda, int* ipiv, int* info);
extern "C" void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO );

void setRhoCrit(double rhoIn){
    rho_crit = rhoIn;
}

void setVandermonde(const vec &SpecialVanderIn, int Nupsample){
    SpecialVandermonde = SpecialVanderIn;
    // Precompute LU factorization
    int info;
    dgetrf_(&Nupsample, &Nupsample, &*SpecialVandermonde.begin(), &Nupsample, ipivVander, &info);
}


// 
// Inputs: quadrature nodes tj, fiber points xj, yj, zj, number of fiber 
// points n, and target point x0, y0, z0. 
// Outputs: the initial guess for the rootfinder
complex rootfinder_initial_guess(const vec &nodes, const vec &Points, const vec3 &target){
    /**
    The initial guess for the rootfinder.
    @param nodes = quadrature nodes on [-1,1] along the fiber
    @param Points = row stacked vector of Chebyshev points on the fiber
    @param target = 3 array target point
    @return complex initial guess
    **/
    
    // Find two closest point
    double Rsqmin1 = INFINITY;
    double Rsqmin2 = INFINITY;
    int imin1, imin2;
    vec3 dvec;
    for (int i=0; i<nodes.size(); i++) {
        for (int d=0; d < 3; d++){
            dvec[d] = Points[3*i+d]-target[d];
        }
        double Rsq = dot(dvec,dvec);
        if (Rsq < Rsqmin1){
            Rsqmin2 = Rsqmin1;
            imin2 = imin1;
            Rsqmin1 = Rsq;
            imin1 = i;
        } else if (Rsq < Rsqmin2){
            Rsqmin2 = Rsq;
            imin2 = i;
        }
    }
    // Now compute initial guess
    int i1 = imin1;
    int i2 = imin2;
    double t1 = nodes[i1];
    double t2 = nodes[i2];
    vec3 p, r;
    for (int d=0; d < 3; d++){
        p[d] = Points[3*i1+d] - Points[3*i2+d];
        r[d] = target[d] - Points[3*i1+d];
    }
    double pnorm = sqrt(dot(p,p));
    double rnormsq = dot(r,r);  
    double rdotp = dot(r,p);
    double a = (t1-t2)*rdotp/(pnorm*pnorm);
    double b = sqrt(std::abs(rnormsq-rdotp*rdotp/(pnorm*pnorm))) * (t1-t2)/pnorm;
    complex tinit(t1+a,b);
    return tinit;
}

double bernstein_radius(complex z){
    /**
    Bernstein radius for a complex number z.
    @param z = complex number
    @return the Bernstein radius
    **/
    return std::abs(z + std::sqrt(z - 1.0)*std::sqrt(z+1.0));
}

int rootfinder(const vec &Coefficients,const vec &DerivCoefficients,const vec3 &target, complex &t){
    /**
    Root finder for special quadrature. 
    @param Coefficients = row stacked vector of Chebyshev coefficients for the fiber position
    @param DerivCoefficients = row stacked vector of derivative coefficients for the fiber position
    @param target = target point (3 array)
    @param t = the complex root (will be modifed here)
    @return whether the root finder converged
    **/
    // Find roots using Newton and Muller
    double tol = ROOT_FINDER_TOL;
    int maxiter_newton = MAX_RF_ITERS;
    int maxiter_muller = MAX_RF_ITERS;
    // === Newton
    // Setup history variables (needed in Muller)
    complex Fp, tp, Fpp, tpp;
    complex dt, F, Fprime;
    int converged = 0;
    int iter;
    double absres;
    int Ncoeffs = Coefficients.size()/3;
    int Nexpnd = std::min(Ncoeffs, MAX_EXPANSION_ORDER);
    compvec3 xiter, xprimeiter;
    compvec3 dvec;
    for (iter=0; iter<maxiter_newton; iter++){
        // Chebyshev eval
        eval_Cheb3Dirs(Coefficients,Nexpnd,t,xiter);
        eval_Cheb3Dirs(DerivCoefficients,Nexpnd,t,xprimeiter);
        // Compute F and F'
        for (int d=0; d < 3; d++){
            dvec[d] = xiter[d]-target[d];
        }
        F = dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2];
        Fprime = 2.0*(dvec[0]*xprimeiter[0] + dvec[1]*xprimeiter[1] + dvec[2]*xprimeiter[2]);
        dt = -F/Fprime;
        // Update history
        tpp = tp;
        Fpp = Fp;
        Fp = F;
        tp = t;
        // Update root
        t = t+dt;
        absres = std::abs(dt);
        if (absres < tol){
            converged = 1;
            break;
        }
    }
    if (converged==1){
        VERBINFO("Newton converged in %d iterations.\n", iter);
        return converged;
    } 
    // === Muller
    VERBINFO("Newton did not converge after %d iterations (abs(dt)=%g), switching to Muller\n", iter, absres);
    converged = 0;
    complex q, A, B, C, d1, d2;
    for (iter=0; iter<maxiter_muller; iter++){
        // Chebyshev eval
        eval_Cheb3Dirs(Coefficients,Nexpnd,t,xiter);
        for (int d=0; d < 3; d++){
            dvec[d] = xiter[d]-target[d];
        }
        F = dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2];
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
        if (absres < tol){
            converged = 1;
            break;
        }
    }
    if (converged){
        VERBINFO("Muller converged in %d iterations.\n", iter);
    } else{
    VERBINFO("Muller did not converge after %d iterations. abs(dt)=%g", iter, absres);
    }
    return converged;
}

int calculateRoot(const vec &nodes, const vec &Points, const vec &Coefficients,
                  const vec &DerivCoefficients,const vec3 &target, complex &troot){
    /**
    Method to find the root for special quadrature and determine if special quad is needed
    @param nodes = quadrature nodes on [-1,1] for the fiber
    @param Points = row stacked vector of Chebyshev points on the fiber
    @param Coefficients = row stacked vector of Chebyshev coefficients for the fiber position
    @param DerivCoefficients = row stacked vector of derivative coefficients for the fiber position
    @param target = target point (3 array)
    @param troot = the complex root (will be modifed here)
    @return whether special quad is needed 
    **/
    // First make the initial guess and return 0 if too far. 
    troot = rootfinder_initial_guess(nodes, Points, target);
    if (bernstein_radius(troot) > 1.5*rho_crit){
        return 0; 
    }
    // Calculate the root
    int converged = rootfinder(Coefficients,DerivCoefficients,target,troot);
    // Special quad needed if converged AND the radius of the root is less than critical one
    return (bernstein_radius(troot) <= rho_crit && converged);
}     

// Integrals for a given root. 
void rsqrt_pow_integrals(complex z, int N,vec &I1, vec &I3, vec &I5){
    /**
    Integrals for a given root. 
    I really have no idea what this method does, it comes from Ludvig. 
    @param z = complex root
    @param N = number of fiber points 
    @param I1, I3, I5 = integrals 
    **/
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
}

void pvand(int n, const vec &alpha, vec &x, const vec &b){
  /** x = pvand(n, alpha, x, b)
  //
  // Solves system A*x = b
  // A is Vandermonde matrix, with nonstandard definition
  // A(i,j) = alpha(j)^i
  //
  // Algorithm by Bjorck & Pereyra
  // Mathematics of Computation, Vol. 24, No. 112 (1970), pp. 893-903
  // https://doi.org/10.2307/2004623   
  This algorithm is poorly conditioned for n >=32. We use it for N=32
  exactly because Barnett and af Klintenberg said it was ok, but it will 
  give at most 5 digits. 
  @param n = number of rows in the system
  @param alpha = quadrature nodes on [-1,1]
  @param x = solution 
  @param b = RHS
  **/

  for (int i=0; i<n; i++)

    x[i] = b[i];

  for (int k=0; k<n; k++)

    for (int j=n-1; j>k; j--)

      x[j] = x[j]-alpha[k]*x[j-1];      

  for (int k=n-1; k>0; k--)

    {

      for (int j=k; j<n; j++)

	x[j] = x[j]/(alpha[j]-alpha[j-k]);	

      for (int j=k-1; j<n-1; j++)

	x[j] = x[j]-x[j+1];

    }

}

void specialWeights(const vec &tnodes, const complex &troot, vec &w1s, vec &w3s, vec &w5s, double Lf){
    /** 
    Algorithm to actually compute the special quadrature weights from the complex root. 
    @param tnodes = special quadrature nodes 
    @param troot = the root 
    @param (w1s, w3s, w5s) = the quadrature weights for powers of R 1, 3, 5 in the SBT kernel
    (determined in this method)
    @param Lf = fiber length
    **/
    // Compute the integrals
    int n = w1s.size();
    vec I1s(n), I3s(n), I5s(n);
    rsqrt_pow_integrals(troot,n, I1s, I3s, I5s);
    // Vandermonde solve
    if (useByorck){
        pvand(n,tnodes,w1s,I1s);
        pvand(n,tnodes,w3s,I3s);
        pvand(n,tnodes,w5s,I5s);
        for (int i=0; i< n; i++){
            double tdist = std::abs(tnodes[i]-troot);
            w1s[i]*=tdist*Lf*0.5;
            w3s[i]*=pow(tdist,3)*Lf*0.5;
            w5s[i]*=pow(tdist,5)*Lf*0.5;
        }
    } else { 
        // Precomputed LU
        char trans = 'T';
        int nrhs = 1;
        int info;
        dgetrs_(&trans, &n, &nrhs, & *SpecialVandermonde.begin(), &n, ipivVander, & *I1s.begin(), &n, &info);
        dgetrs_(&trans, &n, &nrhs, & *SpecialVandermonde.begin(), &n, ipivVander, & *I3s.begin(), &n, &info);
        dgetrs_(&trans, &n, &nrhs, & *SpecialVandermonde.begin(), &n, ipivVander, & *I5s.begin(), &n, &info);
        for (int i=0; i< n; i++){
            double tdist = std::abs(tnodes[i]-troot);
            w1s[i]=I1s[i]*tdist*Lf*0.5;
            w3s[i]=I3s[i]*pow(tdist,3)*Lf*0.5;
            w5s[i]=I5s[i]*pow(tdist,5)*Lf*0.5;
        }
    }
}