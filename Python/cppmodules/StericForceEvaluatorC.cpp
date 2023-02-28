#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#include "Chebyshev.cpp"
#include "VectorMethods.cpp"
#include "DomainC.cpp"
#include "types.h"

/**
C++ class to evaluate steric forces
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class StericForceEvaluatorC {

    public: 
    
    StericForceEvaluatorC(int nFib, int NCheb, int Nuni,npDoub pyRuni, double ObjSize, vec3 DomLengths,int nThr):_Dom(DomLengths){
        /**
        Initialize variables related to force density calculations in CLs
        @param sU = list of uniformly spaced nodes on all fibers (size = total number of uniform nodes = 
            Nuni*Nfib if all fibers have the same number of uniform nodes)
        @param sC = list of all Chebyshev nodes on the fibers
        @param win = list of Clenshaw-Curtis quadrature weights on all fibers
        @param sigin = vector of standard deviations for the regularized cross-linker delta function on each fiber
        @param Kin = spring constant for the CLs
        @param rl = rest length of the CLs
        @param nCL = maximum number of CLs
        @param nThr = number of threads for OpenMP
        **/
        _NFib = nFib;
        _nXPerFib = NCheb;
        _nUni = Nuni;
        _ObjectSize = ObjSize;
        _RUniform = vec(pyRuni.size());
        std::memcpy(_RUniform.data(),pyRuni.data(),pyRuni.size()*sizeof(double));
        _nThreads = nThr;  
    }  
 
    py::array getUniformPoints(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec AllUniPoints(3*_NFib*_nUni);
        #pragma omp parallel for num_threads(_nThreads)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalUniPts(3*_nUni);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(_nUni,_nXPerFib,3,1.0,0.0,_RUniform,false,FiberPoints,LocalUniPts);
            for (int i=0; i < 3*_nUni; i++){
                AllUniPoints[3*_nUni*iFib+i] = LocalUniPts[i];
            }
        }
        return makePyDoubleArray(AllUniPoints);
    }
    
    py::array getStericForces(npInt pyUnipairs, npDoub pyUniPts, double g, double dcrit, double delta, double F0){
        /*
        Find steric forces
        */
        // Convert numpy to C++ vector and reset heap for the beginning of the step
        intvec PairsToRepel(pyUnipairs.size());
        std::memcpy(PairsToRepel.data(),pyUnipairs.data(),pyUnipairs.size()*sizeof(int));
        vec uniPts(pyUniPts.size());
        std::memcpy(uniPts.data(),pyUniPts.data(),pyUniPts.size()*sizeof(double));

        int nPairs = PairsToRepel.size()/2;
        vec StericForces(_NFib*_nXPerFib*3);
        for (int iPair=0; iPair < nPairs; iPair++){
            int iUniPt = PairsToRepel[2*iPair];
            int jUniPt = PairsToRepel[2*iPair+1];
            int iuPtMod = iUniPt % _nUni;
            int juPtMod = jUniPt % _nUni;
            int iFib = iUniPt/_nUni;
            int jFib = jUniPt/_nUni;
            vec3 rvec;
            for (int d=0; d < 3; d++){
                rvec[d] = uniPts[3*iUniPt+d]-uniPts[3*jUniPt+d];
            }
            _Dom.calcShifted(rvec,g);
            // Compute the potential at r 
            double nr = normalize(rvec);
            double dUdr;
            // The function dU/dr = F_0 when r < dcrit and F_0*exp((dcrit-r)/delta) r > dcrit. 
            // The truncation distance is 4*delta
            if (nr < dcrit){
                dUdr = -F0;
            } else {
                dUdr = -F0*exp((dcrit-nr)/delta);
            }
            // Multiplication due to rest length and densities
            vec3 forceij;//, force1, force2, X1, X2;
            for (int d =0; d < 3; d++){
                forceij[d] = -dUdr*rvec[d];
            }
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                for (int d =0; d < 3; d++){
                    #pragma omp atomic update
                    StericForces[3*(iPt+_nXPerFib*iFib)+d] += forceij[d]*_RUniform[iuPtMod*_nXPerFib+iPt];
                }
            }
            for (int jPt=0; jPt < _nXPerFib; jPt++){
                for (int d =0; d < 3; d++){
                    #pragma omp atomic update
                    StericForces[3*(jPt+_nXPerFib*jFib)+d] -= forceij[d]*_RUniform[juPtMod*_nXPerFib+jPt];
                }
            }
        }
        return make1DPyArray(StericForces);
    }
    
    private:
    
    int _nXPerFib, _nUni, _NFib, _nThreads;
    double _ObjectSize;
    vec _RUniform;
    DomainC _Dom;
    
   // TODO: Add potential prime, and
    double PotentialPrime(double r){
        if (r > _ObjectSize){
            return 0;
        }
        // r <=ObjectSize, do pair potential prime
        return 1;
    }

    npDoub makePyDoubleArray(vec &cppvec){
        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { (long) cppvec.size()/3 , 3 };
        std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

        // return 2-D NumPy array
        return py::array(py::buffer_info(
        cppvec.data(),                       /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
        ));
    }
    
    npDoub make1DPyArray(vec &cppvec){
        // allocate py::array (to pass the result of the C++ function to Python)
        auto pyArray = py::array_t<double>(cppvec.size());
        auto result_buffer = pyArray.request();
        double *result_ptr    = (double *) result_buffer.ptr;
        // copy std::vector -> py::array
        std::memcpy(result_ptr,cppvec.data(),cppvec.size()*sizeof(double));
        return pyArray;
    }
    
    
};

PYBIND11_MODULE(StericForceEvaluatorC, m) {
    py::class_<StericForceEvaluatorC>(m, "StericForceEvaluatorC")
        .def(py::init<int,int,int, npDoub, double, vec3, int>())
        .def("getUniformPoints", &StericForceEvaluatorC::getUniformPoints)
        .def("getStericForces", &StericForceEvaluatorC::getStericForces);
}


