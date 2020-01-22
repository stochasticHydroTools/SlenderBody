#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>

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
                std::vector<int> xdshifts, std::vector<int> ydshifts, std::vector<int> zdshifts,
                std::vector<double> Xpts,std::vector<double> Ypts, std::vector<double> Zpts,
                int nFib, int N,std::vector<double> s, std::vector<double> w, double Kspring,
                double restlen, double g, double Lx, double Ly,double Lz, double sigma ){
    std::vector <double> forces(nFib*N*3,0.0);
    int iFib, jFib, iPtstar, jPtstar;
    double xsh, ysh, zsh;
    double s1s, s2s, dsx, dsy, dsz, nds, deltah1, deltah2, factor;
    double fx, fy, fz;
    for (int iL=0; iL < nLinks; iL++){
        iPtstar = iPts[iL];
        jPtstar = jPts[iL];
        iFib = iPtstar/N; // integer division
        jFib = jPtstar/N; // integer division
        s1s = su[iPtstar % N];
        s2s = su[jPtstar % N];
        xsh = xdshifts[iL]*Lx+ydshifts[iL]*Ly*g;
        ysh = ydshifts[iL]*Ly;
        zsh = zdshifts[iL]*Lz;
        for (int iPt=0; iPt < N; iPt++){
            for (int jPt=0; jPt < N; jPt++){
                // Displacement vector
                dsx = Xpts[iFib*N+iPt] - Xpts[jFib*N+jPt] + xsh;
                dsy = Ypts[iFib*N+iPt] - Ypts[jFib*N+jPt] + ysh;
                dsz = Zpts[iFib*N+iPt] - Zpts[jFib*N+jPt] + zsh;
                nds = sqrt(dsx*dsx+dsy*dsy+dsz*dsz);
                deltah1 = deltah(s[iPt]-s1s,sigma);
                deltah2 = deltah(s[jPt]-s2s,sigma);
                // Multiplication due to rest length and densities
                factor = (1.0-restlen/nds)*deltah1*deltah2;
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

    
// Module for python
PYBIND11_MODULE(CLUtils, m) {
    m.doc() = "The C++ functions for cross linking"; // optional module docstring

    m.def("CLForces", &CLForces, "Forces due to the cross linkers");
}
