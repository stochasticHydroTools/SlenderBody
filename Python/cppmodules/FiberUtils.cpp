#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>

// This is a pybind11 module of C++ functions that are being used
// for fiber related calculations in the C++ code
// Right now it only has the computation of the finite part integral
// density g in it.

// C++ function to compute the relevant density g(s_i,s_j) for the finite part integration.
// Inputs: N = number of points on fiber, X, Y, Z = N vectors of Chebyshev points on the
// fiber. Xs, Xss, fprime = 3 vectors that are the tangent vector Xs, Xss, and f' at iPt.
// fx, fy, fz = N vectors with the force densities, s = arclength coordinates on the
// fiber centerline, iPt = the point we are building for.
// Ouputs the density g as a 3*N vector, to be reformatted in the Python. 
std::vector <double> FPDensity(int N, std::vector <double> X,std::vector <double> Y,std::vector <double> Z,
                std::vector <double> Xs,std::vector <double> Xss,std::vector<double> fprime,
                std::vector <double> fx,std::vector <double> fy,std::vector <double> fz,
                std::vector <double> s, int iPt){
    std::vector<double> gloc(3*N);
    double rx, ry, rz, nr, rdotf, oneoverr, oneoverds;
    double Xsdotf = Xs[0]*fx[iPt]+Xs[1]*fy[iPt]+Xs[2]*fz[iPt];
    double Xssdotf = Xss[0]*fx[iPt]+Xss[1]*fy[iPt]+Xss[2]*fz[iPt];
    double Xsdotfprime = Xs[0]*fprime[0]+Xs[1]*fprime[1]+Xs[2]*fprime[2];
    for (int jPt=0; jPt < N; jPt++){
        rx = X[iPt]-X[jPt];
        ry = Y[iPt]-Y[jPt];
        rz = Z[iPt]-Z[jPt];
        nr = sqrt(rx*rx+ry*ry+rz*rz);
        oneoverr = 1.0/nr;
        oneoverds = 1.0/(s[jPt]-s[iPt]);
        rdotf = rx*fx[jPt]+ry*fy[jPt]+rz*fz[jPt];
        gloc[jPt*3] = ((fx[jPt] + rx*rdotf*oneoverr*oneoverr)*oneoverr*std::abs(s[jPt]-s[iPt])
                        -(fx[iPt]+Xs[0]*Xsdotf))*oneoverds; // x dir
        gloc[jPt*3+1] = ((fy[jPt] + ry*rdotf*oneoverr*oneoverr)*oneoverr*std::abs(s[jPt]-s[iPt])
                        -(fy[iPt]+Xs[1]*Xsdotf))*oneoverds; // x dir
        gloc[jPt*3+2] = ((fz[jPt] + rz*rdotf*oneoverr*oneoverr)*oneoverr*std::abs(s[jPt]-s[iPt])
                        -(fz[iPt]+Xs[2]*Xsdotf))*oneoverds; // z dir
    }
    // Corrections at iPt
    gloc[iPt*3]=0.5*(Xs[0]*Xssdotf+Xss[0]*Xsdotf)+fprime[0]+Xs[0]*Xsdotfprime;
    gloc[iPt*3+1]=0.5*(Xs[1]*Xssdotf+Xss[1]*Xsdotf)+fprime[1]+Xs[1]*Xsdotfprime;
    gloc[iPt*3+2]=0.5*(Xs[2]*Xssdotf+Xss[2]*Xsdotf)+fprime[2]+Xs[2]*Xsdotfprime;
    return gloc;
}

    
// Module for python
PYBIND11_MODULE(FiberUtils, m) {
    m.doc() = "The C++ function for finite part integral"; // optional module docstring

    m.def("FPDensity", &FPDensity, "Density g for finite part integral");
}
