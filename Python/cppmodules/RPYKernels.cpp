#include <stdio.h>
#include <math.h>
#include "Domain.cpp"
#include <omp.h>
#include "utils.h"

//This is a set of C++ functions that are being used
//for Ewald splitting (and other RPY calculations)

// ================================================
// GLOBAL VARIABLES AND THEIR INITIALIZATION
// ================================================
// Donev: These global variables seem unwise -- what if we had fibers of different thickness?
// No need to change it (it is not easy to fix, which is the problem) but just FYI in terms of programming choices
double a; // hydrodynamic radius
double mu; // fluid viscosity
double mu_inv;
int NEwaldPts;
vec NearFieldVelocity;
double outFront; // 1/(6*pi*mu*a)
double sqrtpi = sqrt(M_PI);

// Donev: Every routine that allocates memory must have a corresponding routine that deallocates it
void initRPYVars(double ain, double muin, int NEwaldPtsin, vec3 Lengths){
    /**
    Initialize the global variables for the RPY Ewald calculations. 
    @param ain = hydrodynamic radius
    @param muin = fluid viscosity
    @param NEwaldPtsin = total # of points for Ewald
    @param Lengths = periodic domain lengths
    **/
    a = ain;
    mu = muin;
    mu_inv = 1.0/mu;
    NEwaldPts = NEwaldPtsin;
    NearFieldVelocity = vec(NEwaldPtsin*3); // Donev: Where is this memory freed/deallocated?
    // Imagine you had fibers growing and shrinking, so points were added/removed so one would need to re-initialize later
    initLengths(Lengths[0],Lengths[1],Lengths[2]);
    outFront = 1.0/(6.0*M_PI*mu*a);
}

//The split RPY near field can be written as M_near = F(r,xi,a)*I+G(r,xi,a)*(I-RR)
//The next 2 functions are those F and G functions.
double Fnear(double r, double xi){
    /**
    Part of the RPY near field that multiplies I. 
    @param r = distance between points 
    @param xi = Ewald parameter
    @return value of F, so that M_near = F*I + G*(I-RR^T)
    **/
	if (r < 1e-10){ // Taylor series
        double val = 1.0/(4.0*sqrtpi*xi*a)*(1.0-exp(-4.0*a*a*xi*xi)+
            4*sqrtpi*a*xi*erfc(2*a*xi));
        return val;
	}
	double f0, f1, f2, f3, f4, f5, f6;
    if (r>2*a){
        f0=0.0;
        f1=(18.0*r*r*xi*xi+3.0)/(64.0*sqrtpi*a*r*r*xi*xi*xi);
        f2=(2.0*xi*xi*(2.0*a-r)*(4.0*a*a+4.0*a*r+9.0*r*r)-2*a-3*r)/ 
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f3=(-2.0*xi*xi*(2.0*a+r)*(4.0*a*a-4.0*a*r+9.0*r*r)+2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f4=(3.0-36.0*r*r*r*r*xi*xi*xi*xi)/(128.0*a*r*r*r*xi*xi*xi*xi);
        f5=(4.0*xi*xi*xi*xi*(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)-3.0)/
			(256.0*a*r*r*r*xi*xi*xi*xi);
        f6=(4.0*xi*xi*xi*xi*(r+2.0*a)*(r+2.0*a)*(4.0*a*a-4.0*a*r+9.0*r*r)-3.0)/
			(256.0*a*r*r*r*xi*xi*xi*xi);
    } else{
        f0=-(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)/(32.0*a*r*r*r);
        f1=(18.0*r*r*xi*xi+3.0)/(64.0*sqrtpi*a*r*r*xi*xi*xi);
        f2=(2.0*xi*xi*(2.0*a-r)*(4.0*a*a+4.0*a*r+9.0*r*r)-2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f3=(-2.0*xi*xi*(2.0*a+r)*(4.0*a*a-4.0*a*r+9.0*r*r)+2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f4=(3.0-36.0*r*r*r*r*xi*xi*xi*xi)/(128.0*a*r*r*r*xi*xi*xi*xi);
        f5=(4.0*xi*xi*xi*xi*(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)-3.0)/
            (256.0*a*r*r*r*xi*xi*xi*xi);
        f6=(4.0*xi*xi*xi*xi*(r+2.0*a)*(r+2.0*a)*(4.0*a*a-4.0*a*r+9.0*r*r)-3.0)/
            (256.0*a*r*r*r*xi*xi*xi*xi);
	}
    double val = f0+f1*exp(-r*r*xi*xi)+f2*exp(-(r-2.0*a)*(r-2.0*a)*xi*xi)+
        f3*exp(-(r+2.0*a)*(r+2.0*a)*xi*xi)+f4*erfc(r*xi)+f5*erfc((r-2*a)*xi)+
        f6*erfc((r+2*a)*xi);
    return val;
}

double Gnear(double r, double xi){
    /**
    Part of the RPY near field that multiplies I-RR^T. 
    @param r = distance between points 
    @param xi = Ewald parameter
    @return value of G, so that M_near = F*I + G*(I-RR^T)
    **/
	if (r < 1e-10){
        return 0;
	}
	double g0, g1, g2, g3, g4, g5, g6;
    if (r>2*a){
        g0=0;
        g1=(6.0*r*r*xi*xi-3.0)/(32.0*sqrtpi*a*r*r*xi*xi*xi);
        g2=(-2.0*xi*xi*(r-2.0*a)*(r-2.0*a)*(2.0*a+3.0*r)+2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g3=(2.0*xi*xi*(r+2.0*a)*(r+2.0*a)*(2.0*a-3.0*r)-2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g4=-3.0*(4.0*r*r*r*r*xi*xi*xi*xi+1.0)/(64.0*a*r*r*r*xi*xi*xi*xi);
        g5=(3.0-4.0*xi*xi*xi*xi*(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
        g6=(3.0-4.0*xi*xi*xi*xi*(2.0*a-3.0*r)*(2.0*a+r)*(2.0*a+r)*(2.0*a+r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
    } else {
        g0=(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r)/(16.0*a*r*r*r);
        g1=(6.0*r*r*xi*xi-3.0)/(32.0*sqrtpi*a*r*r*xi*xi*xi);
        g2=(-2.0*xi*xi*(r-2.0*a)*(r-2.0*a)*(2.0*a+3.0*r)+2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g3=(2.0*xi*xi*(r+2.0*a)*(r+2.0*a)*(2.0*a-3.0*r)-2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g4=-3.0*(4.0*r*r*r*r*xi*xi*xi*xi+1.0)/(64.0*a*r*r*r*xi*xi*xi*xi);
        g5=(3.0-4.0*xi*xi*xi*xi*(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
        g6=(3.0-4.0*xi*xi*xi*xi*(2.0*a-3.0*r)*(2.0*a+r)*(2.0*a+r)*(2.0*a+r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
	}
    double val = g0+g1*exp(-r*r*xi*xi)+g2*exp(-(r-2.0*a)*(r-2.0*a)*xi*xi)+
        g3*exp(-(r+2.0*a)*(r+2.0*a)*xi*xi)+g4*erfc(r*xi)+g5*erfc((r-2.0*a)*xi)+
        g6*erfc((r+2.0*a)*xi);
    return val;
}

vec RPYNKerPairs(int Npairs,const intvec &PairPts, const vec &Points, const vec &Forces, 
                  double xi, double g, double rcut, int nThreads){
    /**
    Near field velocity, done in pairs. 
    @param Npairs = number of pairs 
    @param PairPts = 1D vector of pairs of points, where the odd entries
    are the first element in the pair and the next entry is the second element
    @param Points = 1D vector (row stacked) of the Chebyshev points 
    @param Forces = 1D vector (row stacked) of the forces at the Chebyshev points
    @param xi =  Ewald parameter
    @param g = strain in the coordinate system
    @param rcut = cutoff distance for near field Ewald
    @param nThreads = number of threads for the calculation
    @return near field velocity 
    **/
    omp_set_num_threads(nThreads);
    // Self terms
    double nearSelfMobility = outFront*Fnear(0.0,xi);
    #pragma omp parallel
    {
    // Donev: Why is the schedule fixed to static here?
    #pragma omp for schedule(static)
    for (int iPtDir=0; iPtDir < 3*NEwaldPts; iPtDir++){
        NearFieldVelocity[iPtDir]=nearSelfMobility*Forces[iPtDir];
    }
    // Pairs
    #pragma omp barrier 
    #pragma omp for schedule(static)
    for (int iPair=0; iPair < Npairs; iPair++){
        int iPt = PairPts[2*iPair];
        int jPt = PairPts[2*iPair+1];
        vec3 rvec;
        for (int d=0; d < 3; d++){
            rvec[d]=Points[3*iPt+d]-Points[3*jPt+d];
        }
        // Periodic shift and normalize
        calcShifted(rvec,g);
        double r = normalize(rvec);
        if (r < rcut){ // only do the computation below truncation
            double rdotfj = 0;
            double rdotfi = 0;
            for (int d=0; d < 3; d++){
                rdotfj+=rvec[d]*Forces[3*jPt+d];
                rdotfi+=rvec[d]*Forces[3*iPt+d];
            }
            double F = Fnear(r, xi);
            double G = Gnear(r, xi);
            double co2i = rdotfj*G-rdotfj*F; // coefficients in front of r term
            double co2j = rdotfi*G-rdotfi*F;
            for (int d=0; d < 3; d++){
                # pragma omp atomic update
                NearFieldVelocity[3*iPt+d]+= outFront*(F*Forces[3*jPt+d]+co2i*rvec[d]);
                # pragma omp atomic update
                NearFieldVelocity[3*jPt+d]+= outFront*(F*Forces[3*iPt+d]+co2j*rvec[d]);
            }
        } // end doing the computation
    } // end looping over pairs
    }
    return NearFieldVelocity;
    omp_set_num_threads(1);
}

// The near field kernel, multiplies M_near * f
// Inputs: 3 vector r (not normalized), 3 vector force,
// mu = viscosity, xi = Ewald parameter, and a = blob radius
// Returns the velocity M_near*f
void RPYNKer(vec3 &rvec, const vec3 &force, double xi, vec3 &unear){
    /**
    Near field velocity for a single pair of points
    @param rvec = displacement vector between the points
    @param force = force at one of the points 
    @param unear = 3 array with the near field velocity 
    **/
    double r = normalize(rvec);
    double rdotf = dot(rvec,force);
    double F = Fnear(r, xi);
    double G = Gnear(r, xi);
    double co2 = rdotf*G-rdotf*F;
    for (int d=0; d < 3; d++){
        unear[d] = outFront*(F*force[d]+co2*rvec[d]);
    }
}

//-------------------------------------------------------
// Plain unsplit RPY kernel evaluation
//-------------------------------------------------------
// Donev: Suggest breaking long codes up into sections, here it seems you switch from split RPY (PSE method) to plain RPY

// Donev: I think the next line of documentation is wrong -- no chi here since this is just "plain" RPY
//The RPY kernel can be written as M = Ft(r,xi,a)*I+Gt(r,xi,a)*(I-RR)
//The next 2 functions are those Ft and Gt functions.
double FtotRPY(double r){
    /**
    Part of the RPY kernel that multiplies I. 
    @param r = distance between points 
    @return value of F, so that M_RPY = F*I + G*(I-RR^T)
    **/
	if (r > 2*a){
        return (2.0*a*a+3.0*r*r)/(24*M_PI*r*r*r);
    }
    return (32.0*a-9.0*r)/(192.0*a*a*M_PI);
}

double GtotRPY(double r){
    /**
    Part of the RPY kernel that multiplies I. 
    @param r = distance between points 
    @return value of G, so that M_RPY = F*I + G*(I-RR^T)
    **/
	if (r > 2*a){
        return (-2.0*a*a+3.0*r*r)/(12*M_PI*r*r*r);
    }
    return (16.0*a-3.0*r)/(96.0*a*a*M_PI);
}

void RPYTot(vec3 &rvec, const vec3 &force, vec3 &utot){
    /**
    Total RPY kernel M_ij*f_j
    @param rvec = displacement vector between points
    @param force = force on one of the points (point j) 
    @param utot = velocity at point i due to point j (modified here)
    **/
    double r = normalize(rvec);
    double rdotf = dot(rvec,force);
    double F = FtotRPY(r);
    double G = GtotRPY(r);
    double co2 = rdotf*G-rdotf*F;
    for (int d=0; d < 3; d++){
        utot[d] = mu_inv*(F*force[d]+co2*rvec[d]);
    }
}                                 
