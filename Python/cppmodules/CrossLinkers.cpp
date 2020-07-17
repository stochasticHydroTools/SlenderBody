#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include "Domain.cpp"
#include "types.h"

/**
This is a set of C++ functions that are being used
for cross linker related calculations in the C++ code
**/

// GLOBAL VARIABLES
std::uniform_real_distribution<double> unif(0.0, 1.0);
std::mt19937_64 rng;
double kon0, koff0, cl_cut; // dynamic info
vec sUniform, sChebyshev, weights;
int NCheb, Nuni, maxCLs; 
double sigma,Kspring,restlen;

// ==========================
// METHODS FOR INITIALIZATION
// ==========================
void seedRandomCLs(int myseed){
    /**
    Seed random number generator
    @param myseed = the seed. 0 for a random seed
    **/
    if (myseed==0){
        // initialize the random number generator with time-dependent seed
        static uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        static std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
        rng.seed(ss);
    } else {
        rng.seed(myseed);
    }
}

void initDynamicCLVariables(double konin, double koffin, double cl_cutin, vec3 Lengths){
    /**
    Initialize variables related to dynamic cross linking. 
    @param konin = base binding rate in 1/s
    @param koffin = base unbinding rate in 1/s
    @param cl_cutin = cutoff distance to form a link 
    @param Lengths = 3-vector of periodic lengths
    **/
    initLengths(Lengths[0],Lengths[1],Lengths[2]);
    kon0 = konin;
    koff0 = koffin;
    cl_cut = cl_cutin;
}

void initCLForcingVariables(const vec &sU, const vec &sC, const vec &win, double sigin, double Kin, double rl, int nCL){
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
    maxCLs = nCL;
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
    const vec &chebPoints,vec &CLForces, int nThreads){
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

vec calcCLStress(const intvec &iPts, const intvec &jPts, const vec &Shifts, const vec &uniPoints, 
    const vec &chebPoints, int nThreads){
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

// =====================================
// METHODS FOR DYNAMIC NETWORK EVOLUTION
// =====================================
double calcKoffOne(int iPt,int jPt,const vec &uniPts, double g){
    /**
    Calculate the rate of unbinding for a single link. We are assuming
    in this method a UNIFORM unbinding rate. Later it will be strain dependent. 
    @param iPt = one end of a link (index of uniform fiber points)
    @param jPt = other end of the link (index of uniform fiber points)
    @param uniPts = Nuni*Nfib*3 row stacked vector of uniform fiber points
    (binding site locations)
    @param g = strain in the coordinate system
    @return unbinding rate for this link 
    **/
    return koff0;
}

double calcKonOne(int iPt, int jPt,const vec &uniPts,double g){
    /**
    Calculate the rate of binding for a single link. We are assuming
    in this method a UNIFORM binding rate WITHIN DISTANCE cl_cut.
    @param iPt = one end of a link (index of uniform fiber points)
    @param jPt = other end of the link (index of uniform fiber points)
    @param uniPts = Nuni*Nfib*3 row stacked vector of uniform fiber points
    (binding site locations)
    @param g = strain in the coordinate system
    @return binding rate for this link. Will be 0 if distance between iPt and 
    jPt is larger than cl_cut, and kon0 otherwise. 
    **/
    // Compute displacement vector and nearest image
    vec3 rvec;
    for (int d=0; d < 3; d++){
        rvec[d] = uniPts[3*iPt+d]-uniPts[3*jPt+d];
    }
    calcShifted(rvec,g);
    double r = normalize(rvec);
    if (r < cl_cut){
        return kon0;
    }
    return 0;
}

intvec newEventsList(vec &rates, const intvec &iPts, const intvec &jPts, intvec nowBound,
    intvec added, int nLinks, const vec &uniPts, double g, double tstep) {
    /**
    Master method that determines the events that occur in a given timestep. 
    @param rates = rate of each event happening (units 1/time)
    @param iPts = one endpoint index for each link (uniform point index)
    @param jPts = other endpoint index
    @param nowBound = bound state of each event/link
    @param added = whether a link exists at each uniform point (only 1 link allowed)
    @param nLinks = number of bound CLs going into this method
    @param uniPts = row stacked vector of uniform points on the fibers
    @param g = strain in the coordinate system
    @param tstep = timestep
    @return integer vector of the events that will happen during the timestep. 
    Each "event" is an index in iPts and jPts that says that pair will bind
    **/
    // Initialize times and events
    std::vector <double> times(rates.size()); 
    // Compute times from rates by sampling from an exponential distribution
    for (int iPair = 0; iPair < rates.size(); iPair++){
        times[iPair] = -log(1.0-unif(rng))/rates[iPair];
    }
    std::cout << "TEMPORARY: allowing multiple links per site" << std::endl;
    double systime = 0.0;
    std::vector <int> events;
    int nEvents=0;
    while (systime < tstep){
        
        // Find the minimum time and the nextEvent
        int nextEvent = std::distance(times.begin(), std::min_element(times.begin(), times.end()));
        int iPt = iPts[nextEvent];
        int jPt = jPts[nextEvent];
        systime = times[nextEvent];
        
        // Add the event to the list of events if possible
        if (systime < tstep){
            // Check if the event can actually happen
            //if (nowBound[nextEvent]==1 || (added[iPt]==0 && added[jPt]==0 && nLinks < maxCLs)){
            if (nowBound[nextEvent]==1 || nLinks < maxCLs){
                nEvents++;
                
                // Find if event has already been done, so we need to undo it
                auto iF = std::find(events.begin(), events.end(), nextEvent); 
                if (iF != events.end()){ 
                    // already done, now going back, remove from list
                    events.erase(iF);
                } else{
                    events.push_back(nextEvent);
                }
                nowBound[nextEvent] = 1 - nowBound[nextEvent]; // change bound state
                added[iPt]=1-added[iPt]; // change local copy of added
                added[jPt]=1-added[jPt];
                nLinks+=2*nowBound[nextEvent]-1; // -1 if now unbound, 1 if now bound
                if (nowBound[nextEvent]){ // just became bound, calc rate to unbind
                    rates[nextEvent] = calcKoffOne(iPt,jPt,uniPts,g);
                } else {// just became unbound, calculate rate to bind back
                    rates[nextEvent] = calcKonOne(iPt,jPt,uniPts,g);
                }
            }
            // Redraw random time for that event
            times[nextEvent] = -log(1.0-unif(rng))/rates[nextEvent]+systime;
        }
        if (nEvents == maxCLs){ // save time since all we do now is fill up links at t =0
            std::cout << "TEMPORARY:  returning as all links filled" << std::endl;
            return events;
        }
    }
    return events;
}
