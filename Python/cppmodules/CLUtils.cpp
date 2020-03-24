#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include "EwaldUtils.cpp"

//This is a pybind11 module of C++ functions that are being used
//for cross linker related calculations in the C++ code

// Density function in the force/energy of crosslinking.
double deltah(double a,double sigma){
    return 1.0/sqrt(2.0*M_PI*sigma*sigma)*exp(-a*a/(2.0*sigma*sigma));
}


// Compute the forces from the list of links
// Inputs (there are many!): nLinks = number of links, iPts = nLinks vector of uniform point numbers where
// one end is, jPts = nLinks vector of uniform point numbers where the other end is. su = arclength
// coordinates of the uniform points. xdshifts, ydshifts, zdshifts = nLinks vectors that contain
// the integer number of periods that you shift in the (x',y',z') directions. Xpts, Ypts, Zpts = N*nFib
// vectors of the Chebyshev points on the fibers, nFib = integer number of fibers, N = number of
// points per fiber, s = N vector of the Chebyshev nodes, w = N vector of the Chebyshev weights,
// Kspring = spring constant for cross linking, restlen = rest length of the springs, g = strain in the
// coordinate system. Lx, Ly, Lz = periodic lengths in each direction, sigma = standard deviation of the
// Gaussian in the cross linker energy.
// Return: an N*nFib*3 vector of the forces due to cross linking.
std::vector <double> CLForces(int nLinks, std::vector<int> iPts, std::vector<int> jPts,std::vector<double> su,
                std::vector<double> xdshifts, std::vector<double> ydshifts, std::vector<double> zdshifts,
                std::vector<double> Xpts,std::vector<double> Ypts, std::vector<double> Zpts,
                int nFib, int N,std::vector<double> s, std::vector<double> w, double Kspring,
                double restlen, double sigma){
    std::vector <double> forces(nFib*N*3,0.0);
    int iFib, jFib, iPtstar, jPtstar;
    double s1s, s2s, dsx, dsy, dsz, nds, deltah1, deltah2, factor, totwt1, totwt2, rnorm;
    double fx, fy, fz;
    for (int iL=0; iL < nLinks; iL++){
        iPtstar = iPts[iL];
        jPtstar = jPts[iL];
        iFib = iPtstar/N; // integer division
        jFib = jPtstar/N; // integer division
        s1s = su[iPtstar % N];
        s2s = su[jPtstar % N];
        totwt1=0;
        totwt2=0;
        for (int kPt=0; kPt < N; kPt++){
            totwt1+=deltah(s[kPt]-s1s,sigma)*w[kPt];
            totwt2+=deltah(s[kPt]-s2s,sigma)*w[kPt];
        }
        rnorm = 1.0/(totwt1*totwt2);
        for (int iPt=0; iPt < N; iPt++){
            for (int jPt=0; jPt < N; jPt++){
                // Displacement vector
                dsx = Xpts[iFib*N+iPt] - Xpts[jFib*N+jPt] - xdshifts[iL];
                dsy = Ypts[iFib*N+iPt] - Ypts[jFib*N+jPt] - ydshifts[iL];
                dsz = Zpts[iFib*N+iPt] - Zpts[jFib*N+jPt] - zdshifts[iL];
                nds = sqrt(dsx*dsx+dsy*dsy+dsz*dsz);
                deltah1 = deltah(s[iPt]-s1s,sigma);
                deltah2 = deltah(s[jPt]-s2s,sigma);
                // Multiplication due to rest length and densities
                factor = (1.0-restlen/nds)*deltah1*deltah2*rnorm;
                fx = -Kspring*dsx*factor;
                fy = -Kspring*dsy*factor;
                fz = -Kspring*dsz*factor;
                // Increment forces
                forces[iFib*3*N+3*iPt]+=fx*w[jPt];
                forces[iFib*3*N+3*iPt+1]+=fy*w[jPt];
                forces[iFib*3*N+3*iPt+2]+=fz*w[jPt];
                forces[jFib*3*N+3*jPt]-=fx*w[iPt];
                forces[jFib*3*N+3*jPt+1]-=fy*w[iPt];
                forces[jFib*3*N+3*jPt+2]-=fz*w[iPt];
            }
        }
    }
    return forces;
}

// Calculate the rate of unbinding for a single link. 
// Inputs: (iPt,jPt) = the two ends of the link, uniPts = 3*N row-stacked array
// of locations corresponding to the binding sites, g = strain in the coordinate
// system, Lens = periodic lengths, koff0 = base unbinding rate
// Output: the rate of unbinding for that link
double calcKoffOne(int iPt,int jPt,const std::vector <double> &uniPts, double g,
    const std::vector <double> &Lens, double koff0){
    return koff0;
}

// Calculate the rate of binding for a single link. 
// Inputs: (iPt,jPt) = the two ends of the link, uniPts = 3*N row-stacked array
// of locations corresponding to the binding sites, g = strain in the coordinate
// system, Lens = periodic lengths, kon0 = base binding rate, cl_cut = cutoff 
// distance for CL to form
// Output: the rate of binding for that link
double calcKonOne(int iPt, int jPt,const std::vector <double> &uniPts,double g,
    const std::vector <double> &Lens, double kon0, double cl_cut){
    // Compute displacement vector and nearest image
    std::vector <double> rvec(3);
    for (int d=0; d < 3; d++){
        rvec[d] = uniPts[3*iPt+d]-uniPts[3*jPt+d];
    }
    rvec = calcShifted(rvec,g,Lens[0],Lens[1],Lens[2]);
    double r = normalize(rvec);
    if (r < cl_cut){
        return kon0;
    }
    return 0;
}

// Method to calculate the rate of binding for a set of possible newLinks.  
// Inputs: nPairs = # of new potential new binding locations, (iPts, jPts) = the endpoints
// of the potential new links. uniPoints = 3*N x 1 row-stacked array of the locations of the 
// uniform points on the fiber, cl_cut = cutofff distance for CLs to form, kon0 = base binding 
// rate, g = strain in domain, Lens = periodic lengths
// Outputs: the rates of binding for each pair, Kons
std::vector <double> calcKon(int nPairs, const std::vector<int> &iPts, const std::vector<int> &jPts, 
    const std::vector <double> &uniPoints, double cl_cut, double kon0, double g, 
    const std::vector <double> &Lens){
    std::vector <double> Kons(nPairs,0.0);
    for (int iPair=0; iPair < nPairs; iPair++){
        // Call single link method
        Kons[iPair] = calcKonOne(iPts[iPair],jPts[iPair],uniPoints,g,Lens,kon0,cl_cut);
    }
    return Kons;
}

// Update the times of binding for changed states of the network. 
// Inputs: times = array or original times for each event
// (iPts, jPts) = the endpoints of each event, 
// nowBound =  bound state for each event (1 if bound, 0 if unbound),
// added = Nsites*Nfib array indicating whether there is a link at each site
// nLinks = current number of links in the system, nCL = number of cross-linkers
// Output: the new updated array of times
std::vector <double> updateTimes(const std::vector <double> &times, const std::vector<int> &iPts, 
    const std::vector<int> &jPts,const std::vector<int> &nowBound, const std::vector<int> &added,
    int nLinks, int nCL){
    std::vector<double> newTimes(times.begin(), times.end()); // vector newTimes that is copy of times
    for (int iPair=0; iPair < nowBound.size(); iPair++){
        if (nowBound[iPair]==0){ // want to bind
            // Can't bind if no links available or if site blocked
            if (nLinks == nCL || added[iPts[iPair]]==1 || added[jPts[iPair]]==1){
                newTimes[iPair] = std::numeric_limits<double>::infinity();
            }
        } else{ // want to unbind
            // Can't unbind if there is nothing there to unbind
            if (added[iPts[iPair]]==0 || added[jPts[iPair]]==0){
                newTimes[iPair] = std::numeric_limits<double>::infinity();
            }
        }
    }
    return newTimes;
}

// Master method that determines the events that occur in a given timestep. 
// Inputs: times = vector of time each event should happen, (iPts, jPts) = endpoints
// for each event (points on the fibers), nowBound = whether each pair is currently 
// bound with a CL or not, added = whether each fiber point has a link attached to it
// or not. nLinks = initial number of links, nCL = number of possible CLs, 
// uniPts = 3*N row-stacked array of locations, g = strain in the coordinate system, 
// Lens = periodic lengths, tstep = timestep for move, kon0, koff0 = base rates of 
// binding and unbinding, cl_cut = cutoff distance for CL formulation
// Output: vector of events in order they occur. Each "event" is an index in iPts, jPts, etc.
std::vector <int> newEventsCL(std::vector <double> times, const std::vector<int> &iPts, 
    const std::vector<int> &jPts, std::vector<int> &nowBound, std::vector<int> &added,
    int nLinks, int nCL,std::vector<double> uniPts, double g, std::vector <double> Lens, double tstep,
    double kon0, double koff0, double cl_cut){
    // Seed random number generator
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    // Initialize times and events
    double systime = 0.0;
    std::vector <int> events;
    const int nAlreadyBound = nLinks; // the first nAlreadyBound events are already bound links
    while (systime < tstep){
        // Update the vector of times
        std::vector <double> newTimes = updateTimes(times,iPts,jPts,nowBound,added,nLinks,nCL);
        // Find the minimum time and the nextEvent
        int nextEvent = std::distance(newTimes.begin(), std::min_element(newTimes.begin(), newTimes.end()));
        int iPt = iPts[nextEvent];
        int jPt = jPts[nextEvent];
        systime+= newTimes[nextEvent];
        // Add the event to the list of events if possible
        if (systime < tstep){
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
            double newRate; // the rate for the next event
            if (nowBound[nextEvent]){ // just became bound, calc rate to unbind
                newRate = calcKoffOne(iPt,jPt,uniPts,g,Lens,koff0);
            } else {// just became unbound, calculate rate to bind back
                if (nextEvent < nAlreadyBound){ // those originally bound
                    newRate = 0; // original pairs cannot rebind (avoid double count)
                } else { 
                    newRate = calcKonOne(iPt,jPt,uniPts,g,Lens,kon0,cl_cut);
                }
            }
            // Redraw random time for that event
            //std::cout << "Random number " << 1.0-dis(gen) << "\n";
            times[nextEvent] = -log(1.0-dis(gen))/newRate;
            //times[nextEvent] = 0.1/newRate; // temp to compare w python
        }
    }
    return events;
}
   

    
// Module for python
PYBIND11_MODULE(CLUtils, m) {
    m.doc() = "The C++ functions for cross linking"; // optional module docstring

    m.def("CLForces", &CLForces, "Forces due to the cross linkers");
    m.def("calcKon", &calcKon, "Rate binding constants");
    m.def("newEventsCL", &newEventsCL, "Vector of new events");
}
