#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include "VectorMethods.cpp"
#include "types.h"

/**
Domain.cpp
C++ companion to Domain.py 
Handles computations related to 
periodic sheared domain
**/

namespace py=pybind11;

// Global variables for periodic Domain
class DomainC {
    
    public:
    
    DomainC(vec3 Lengths){
        /*
        Constructor.
        Input: lengths of domain
        */
        _Ld = Lengths;
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
        double s1 = round(rvec[1]/_Ld[1]);
        // Shift in (y',z') directions
        rvec[0]-= g*_Ld[1]*s1;
        rvec[1]-= _Ld[1]*s1;
        rvec[2]-= _Ld[2]*round(rvec[2]/_Ld[2]);
        // Shift in x direction
        rvec[0]-= _Ld[0]*round(rvec[0]/_Ld[0]);
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
        //std::cout << "Calling calc shifted in new cpp class for python" << std::endl;
        calcShifted(rvec,g);
        return rvec;
    }
    
    private:
    vec3 _Ld;
    
};

PYBIND11_MODULE(DomainC, m) {
    py::class_<DomainC>(m, "DomainC")
        .def(py::init<vec3>())
        .def("calcShifted", &DomainC::calcShiftedPython);
}
