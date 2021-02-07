#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "Chebyshev.cpp"
#include "VectorMethods.cpp"
#include "types.h"

/**
This is a set of C++ functions that are being used
for cross linker related calculations in the C++ code
**/

// GLOBAL VARIABLES
vec sUniform, sChebyshev, weights;
int NCheb, Nuni, nThreads; 
double sigma,Kspring,restlen;

void initCLForcingVariables(const vec &sU, const vec &sC, const vec &win, double sigin, double Kin, double rl, int nThreadsIn){
    /**
    Initialize variables related to force density calculations in CLs
    @param sU = uniformly spaced nodes on the fiber on [0,L]
    @param sC = Chebyshev nodes on the fiber on [0,L]
    @param win = Clenshaw-Curtis quadrature weights on the fiber
    @param sigin = standard deviation for the regularized cross-linker delta function
    @param Kin = spring constant for the CLs
    @param rl = rest length of the CLs
    @param nCL = maximum number of CLs
    **/
    sUniform = sU;
    Nuni = sUniform.size();
    sChebyshev = sC;
    weights = win;
    NCheb = weights.size();
    sigma = sigin;
    Kspring = Kin;
    restlen = rl;
    nThreads = nThreadsIn;
}
        
// ================================
// METHODS FOR CL FORCE CALCULATION
// ================================
double deltah(double a){
    /**
    Regularized delta function for the CL force density calculations. 
    @param a = distance between two fiber points
    @return value of delta_h(a,sigma). Right now a Gaussian with standard dev = sigma
    **/
    return 1.0/sqrt(2.0*M_PI*sigma*sigma)*exp(-a*a/(2.0*sigma*sigma));
}

void calcCLForces(const intvec &iPts, const intvec &jPts, const vec &Shifts, const vec &uniPoints,
    const vec &chebPoints,vec &CLForces){
    /**
    Compute force densities on the fibers from the list of links. 
    @param iPts = nLinks vector of uniform point numbers where one CL end is
    @param jPts = nLinks vector of uniform point numbers where the other end is
    @param Shifts = rowstacked vector of nLinks shifts in link displacements due to periodicity
    @param uniPoints = rowstacked vector of the uniform fiber points 
    @param chebPoints = rowstacked vector of Chebyshev fiber points
    @param CLForces = force densities on the fibers (row stacked) to be modified
    @param nThreads = number of threads for parallel processing
    **/
    #pragma omp parallel for num_threads(nThreads)
    for (int iL=0; iL < iPts.size(); iL++){
        int iPtstar = iPts[iL];
        int jPtstar = jPts[iL];
        int iFib = iPtstar/Nuni; // integer division
        int jFib = jPtstar/Nuni; // integer division
        double s1star = sUniform[iPtstar % Nuni];
        double s2star = sUniform[jPtstar % Nuni];
        double totwt1=0;
        double totwt2=0;
        for (int kPt=0; kPt < NCheb; kPt++){
            totwt1+=deltah(sChebyshev[kPt]-s1star)*weights[kPt];
            totwt2+=deltah(sChebyshev[kPt]-s2star)*weights[kPt];
        }
        double rnorm = 1.0/(totwt1*totwt2); // normalization factor
        vec3 ds;
        for (int d =0; d < 3; d++){
            ds[d] = uniPoints[3*iPtstar+d] -  uniPoints[3*jPtstar+d] - Shifts[3*iL+d];
        }
        double nds = sqrt(dot(ds,ds));
        //vec3 totForce = {0,0,0};
        //vec3 totTorq = {0,0,0};
        for (int iPt=0; iPt < NCheb; iPt++){
            for (int jPt=0; jPt < NCheb; jPt++){
                // Displacement vector
                double deltah1 = deltah(sChebyshev[iPt]-s1star);
                double deltah2 = deltah(sChebyshev[jPt]-s2star);
                // Multiplication due to rest length and densities
                double factor = (1.0-restlen/nds)*deltah1*deltah2*rnorm;
                vec3 forceij;//, force1, force2, X1, X2;
                for (int d =0; d < 3; d++){
                    forceij[d] = -Kspring*(chebPoints[iFib*3*NCheb+3*iPt+d] -  chebPoints[jFib*3*NCheb+3*jPt+d] - Shifts[3*iL+d])*factor;
                    #pragma omp atomic update
                    CLForces[iFib*3*NCheb+3*iPt+d] += forceij[d]*weights[jPt];
                    //force1[d] = forceij[d]*weights[jPt];
                    //X1[d] = chebPoints[iFib*3*NCheb+3*iPt+d];
                    #pragma omp atomic update
                    CLForces[jFib*3*NCheb+3*jPt+d] -= forceij[d]*weights[iPt];
                    //force2[d] = -forceij[d]*weights[iPt];
                    //X2[d] = chebPoints[jFib*3*NCheb+3*jPt+d] + Shifts[3*iL+d];
                }
                //vec3 torq1 = {X1[1]*force1[2]-X1[2]*force1[1], X1[2]*force1[0]-X1[0]*force1[2],X1[0]*force1[1]-X1[1]*force1[0]};
                //vec3 torq2 = {X2[1]*force2[2]-X2[2]*force2[1], X2[2]*force2[0]-X2[0]*force2[2],X2[0]*force2[1]-X2[1]*force2[0]};
                //for (int d =0; d < 3; d++){
                //    totTorq[d]+=torq1[d]*weights[iPt]+torq2[d]*weights[jPt];
                //    totForce[d]+=force1[d]*weights[iPt]+force2[d]*weights[jPt];
                // }
            }
        }
        //std::cout << "Total force due to link " << iL << "(" << totForce[0] << " , " << totForce[1] << " , " << totForce[2] << std::endl;
        //std::cout << "Total torque due to link " << iL << "(" << totTorq[0] << " , " << totTorq[1] << " , " << totTorq[2] << std::endl;
    }
}

void calcCLForces2(const intvec &iFibs, const intvec &jFibs, const vec &iSstars, const vec & jSstars, const vec &Shifts,    
    const vec &chebPoints, const vec &chebCoefficients, vec &CLForces, double Lfib){
    /**
    Compute force densities on the fibers from the list of links. 
    @param iFibs = nLinks vector of fiber numbers where one CL end is
    @param jFibs = nLinks vector of fiber numbers where the other end is
    @param iSstars = nLinks vector of s locations where one end is
    @param jSstars = nLinks vector of s locations for the other end
    @param Shifts = rowstacked vector of nLinks shifts in link displacements due to periodicity
    @param uniPoints = rowstacked vector of the uniform fiber points 
    @param chebPoints = rowstacked vector of Chebyshev fiber points
    @param chebCoefficients = rowstacked vector of Chebyshev coefficients
    @param CLForces = force densities on the fibers (row stacked) to be modified
    @param nThreads = number of threads for parallel processing
    **/
    #pragma omp parallel for num_threads(nThreads)
    for (int iL=0; iL < iFibs.size(); iL++){
        int iFib = iFibs[iL];
        int jFib = jFibs[iL];
        double s1star = iSstars[iL];
        double s2star = jSstars[iL];
        double totwt1=0;
        double totwt2=0;
        for (int kPt=0; kPt < NCheb; kPt++){
            totwt1+=deltah(sChebyshev[kPt]-s1star)*weights[kPt];
            totwt2+=deltah(sChebyshev[kPt]-s2star)*weights[kPt];
        }
        double rnorm = 1.0/(totwt1*totwt2); // normalization factor
        vec3 ds;
        // Find the two points on the fibers
        // TEMP: copy coefficients into two vectors
        vec Xihat(3*NCheb), Xjhat(3*NCheb);
        for (int iPt=0; iPt < 3*NCheb; iPt++){
            Xihat[iPt] = chebCoefficients[iFib*3*NCheb+iPt];
            Xjhat[iPt] = chebCoefficients[jFib*3*NCheb+iPt];
        }
        vec3 Xistar={0,0,0};
        vec3 Xjstar={0,0,0};
        eval_Cheb3Dirs(Xihat, 2.0*s1star/Lfib-1, Xistar);
        eval_Cheb3Dirs(Xjhat, 2.0*s2star/Lfib-1, Xjstar);
        for (int d =0; d < 3; d++){
            ds[d] = Xistar[d] -  Xjstar[d] - Shifts[3*iL+d];
        }
        double nds = sqrt(dot(ds,ds));
        for (int iPt=0; iPt < NCheb; iPt++){
            for (int jPt=0; jPt < NCheb; jPt++){
                // Displacement vector
                double deltah1 = deltah(sChebyshev[iPt]-s1star);
                double deltah2 = deltah(sChebyshev[jPt]-s2star);
                // Multiplication due to rest length and densities
                double factor = (1.0-restlen/nds)*deltah1*deltah2*rnorm;
                vec3 forceij;
                for (int d =0; d < 3; d++){
                    forceij[d] = -Kspring*(chebPoints[iFib*3*NCheb+3*iPt+d] -  chebPoints[jFib*3*NCheb+3*jPt+d] - Shifts[3*iL+d])*factor;
                    #pragma omp atomic update
                    CLForces[iFib*3*NCheb+3*iPt+d] += forceij[d]*weights[jPt];
                    #pragma omp atomic update
                    CLForces[jFib*3*NCheb+3*jPt+d] -= forceij[d]*weights[iPt];
                }
            }
        }
    }
}

vec calcCLStress(const intvec &iPts, const intvec &jPts, const vec &Shifts, const vec &uniPoints, const vec &chebPoints){
    /**
    Compute force densities on the fibers from the list of links. 
    @param iPts = nLinks vector of uniform point numbers where one CL end is
    @param jPts = nLinks vector of uniform point numbers where the other end is
    @param Shifts = rowstacked vector of nLinks shifts in link displacements due to periodicity
    @param uniPoints = rowstacked vector of the uniform fiber points 
    @param chebPoints = rowstacked vector of Chebyshev fiber points
    @param nThreads = number of threads for parallel processing
    @return stress tensor
    **/
    vec stress(9,0.0);
    #pragma omp parallel for num_threads(nThreads)
    for (int iL=0; iL < iPts.size(); iL++){
        vec forceDeni(3*NCheb,0.0);
        vec forceDenj(3*NCheb,0.0);
        int iPtstar = iPts[iL];
        int jPtstar = jPts[iL];
        int iFib = iPtstar/Nuni; // integer division
        int jFib = jPtstar/Nuni; // integer division
        double s1star = sUniform[iPtstar % Nuni];
        double s2star = sUniform[jPtstar % Nuni];
        double totwt1=0;
        double totwt2=0;
        for (int kPt=0; kPt < NCheb; kPt++){
            totwt1+=deltah(sChebyshev[kPt]-s1star)*weights[kPt];
            totwt2+=deltah(sChebyshev[kPt]-s2star)*weights[kPt];
        }
        double rnorm = 1.0/(totwt1*totwt2); // normalization factor
        vec3 ds;
        for (int d =0; d < 3; d++){
            ds[d] = uniPoints[3*iPtstar+d] -  uniPoints[3*jPtstar+d] - Shifts[3*iL+d];
        }
        double nds = sqrt(dot(ds,ds));
        for (int iPt=0; iPt < NCheb; iPt++){
            for (int jPt=0; jPt < NCheb; jPt++){
                // Displacement vector
                double deltah1 = deltah(sChebyshev[iPt]-s1star);
                double deltah2 = deltah(sChebyshev[jPt]-s2star);
                // Multiplication due to rest length and densities
                double factor = (1.0-restlen/nds)*deltah1*deltah2*rnorm;
                vec3 forceij;
                for (int d =0; d < 3; d++){
                    forceij[d] = -Kspring*(chebPoints[iFib*3*NCheb+3*iPt+d] -  chebPoints[jFib*3*NCheb+3*jPt+d] - Shifts[3*iL+d])*factor;
                    forceDeni[3*iPt+d] += forceij[d]*weights[jPt];
                    forceDenj[3*jPt+d] -= forceij[d]*weights[iPt];
                }
            }
        }
        for (int iPt=0; iPt < NCheb; iPt++){ // outer product to get stress
            for (int dX=0; dX < 3; dX++){
                for (int df=0; df < 3; df++){
                    #pragma omp atomic update
                    stress[3*dX+df]-=weights[iPt]*chebPoints[iFib*3*NCheb+3*iPt+dX]*forceDeni[3*iPt+df];
                    #pragma omp atomic update
                    stress[3*dX+df]-=weights[iPt]*(chebPoints[jFib*3*NCheb+3*iPt+dX]+Shifts[3*iL+dX])*forceDenj[3*iPt+df];
                }
            }
        }
    }
    return stress;
}
