#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include "VectorMethods.cpp"

/**
Domain.cpp
C++ companion to Domain.py 
Handles computations related to 
periodic sheared domain
**/

// Global variables for periodic Domain
double Lx, Ly, Lz; // periodic lengths

void initLengths(double Lxx, double Lyy, double Lzz){
    /**
    Initialize global variables. 
    @param Lxx = x periodic length
    @param Lyy = y periodic length
    @param Lzz = z periodic length
    **/
    Lx = Lxx;
    Ly = Lyy;
    Lz = Lzz;
}


// Inputs: rvec = 3 vector to shift, g = slantedness of grid
void calcShifted(vec3 &rvec, double g){
    /**
    Compute shifted coordinates of a vector rvec. The goal is to shift a vector
    rvec periodically until it's on [-Lx/2,Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2].
    This must be done carefully, however, since the periodicity is not standard
    (it's perioidic on a slanted grid with strain g).
    @param rvec = the displacement vector (to be modified)
    @param g = strain in coordinate system
    **/
    double s1 = round(rvec[1]/Ly);
    // Shift in (y',z') directions
    rvec[0]-= g*Ly*s1;
    rvec[1]-= Ly*s1;
    rvec[2]-= Lz*round(rvec[2]/Lz);
    // Shift in x direction
    rvec[0]-= Lx*round(rvec[0]/Lx);
}

void PrimeCoords(vec3 &rvec, double g){
    rvec[0]-=g*rvec[1];    
}

vec3 calcShiftedPython(vec3 rvec, double g){
    /**
    Python wrapper for calcShifted. 
    @param rvec = the input displacement vector
    @param g = strain in coordinate system
    @return the new displacement
    **/
    calcShifted(rvec,g);
    return rvec;
}

PYBIND11_MODULE(DomainCpp, m){
    m.def("calcShifted", &calcShiftedPython, "Shift in primed variables to [-L/2, L/2]^3");
    m.def("initLengths",&initLengths, "initialize the lengths for the domain");
}
