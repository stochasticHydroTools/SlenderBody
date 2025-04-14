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
#include "types.h"

/**
C++ class to evaluate forces and stresses on a collection of cross-linked fibers
Bound to python using pybind11. 
The binding code is at the end of this file
Documentation last updated: 03/12/2021
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class CrossLinkForceEvaluator {

    public: 
    
    CrossLinkForceEvaluator(int Nuni,vec sUni, npDoub pyRuni,npDoub pyWTilde, intvec FibFromSite, intvec NChebs, vec sCheby, 
        vec wtsCheb, vec sigin, double Kin, double rl, int nThr){
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
        _sUniform = sUni;
        _Nuniform = Nuni;
        _RUniform = vec(pyRuni.size());
        _WTilde = vec(pyWTilde.size());
        std::memcpy(_RUniform.data(),pyRuni.data(),pyRuni.size()*sizeof(double));
        std::memcpy(_WTilde.data(),pyWTilde.data(),pyWTilde.size()*sizeof(double));
        _FibFromSiteIndex = FibFromSite;
        _sChebyshev = sCheby;
        _weights = wtsCheb;
        _NCheb = NChebs;
        _chebStart = intvec(_NCheb.size()+1,0);
        for (uint i = 0; i < _NCheb.size(); i++){
            _chebStart[i+1]=_chebStart[i]+_NCheb[i];
        }
        //for (int i = 0; i < _chebStart.size(); i++){
        //    std::cout << "Starting Cheb node on fiber " << i << ": " << _chebStart[i] << std::endl;
        //}
        _sigma = sigin;
        _Kspring = Kin;
        _restlen = rl;
        _nThreads = nThr;  
    }  
 
    py::array calcCLForces(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints){
        /**
        Compute forces (not density) on the fibers from the list of links. 
        @param iPts = nLinks vector of uniform point numbers where one CL end is
        @param jPts = nLinks vector of uniform point numbers where the other end is
        @param pyShifts = 2D numpy array of shifts in link displacements due to periodicity
        @param pyuniPoints = 2D numpy array of uniform fiber points 
        @param py chebPoints = 2D numpy array of Chebyshev fiber points
        @return CLForces = force densities on all fibers (row stacked) due to CLs
        **/
        // Copying input
        vec Shifts(pyShifts.size());
        vec uniPoints(pyUnipoints.size());
        vec chebPoints(pyChebPoints.size());

        // copy py::array -> std::vector
        std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
        std::memcpy(uniPoints.data(),pyUnipoints.data(),pyUnipoints.size()*sizeof(double));
        std::memcpy(chebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));

        // Pure C++ function
        vec CLForceDensities(pyChebPoints.shape()[0]*3,0.0);
        #pragma omp parallel for num_threads(_nThreads)
        for (uint iL=0; iL < iPts.size(); iL++){
            int iPtstar = iPts[iL];
            int jPtstar = jPts[iL];
            int iFib = _FibFromSiteIndex[iPtstar];
            int jFib = _FibFromSiteIndex[jPtstar];
            //std::cout << "Link " << iL << " (" << iPtstar << "," << jPtstar << ") on fibers (" << iFib << "," << jFib << ") \n";
            double s1star = _sUniform[iPtstar];
            double s2star = _sUniform[jPtstar];
            double totwt1=0;
            double totwt2=0;
            vec deltah1(_NCheb[iFib]);
            vec deltah2(_NCheb[jFib]);
            for (int kPt=_chebStart[iFib]; kPt < _chebStart[iFib+1]; kPt++){
                deltah1[kPt-_chebStart[iFib]] = deltah(_sChebyshev[kPt]-s1star,iFib);
                totwt1+=deltah1[kPt-_chebStart[iFib]]*_weights[kPt];
            }
            for (int kPt=_chebStart[jFib]; kPt < _chebStart[jFib+1]; kPt++){
                deltah2[kPt-_chebStart[jFib]] = deltah(_sChebyshev[kPt]-s2star,jFib);
                totwt2+=deltah2[kPt-_chebStart[jFib]]*_weights[kPt];
            }
            double rnorm = 1.0/(totwt1*totwt2); // normalization factor
            // Displacement vector
            vec3 ds;
            for (int d =0; d < 3; d++){
                ds[d] = uniPoints[3*iPtstar+d] -  uniPoints[3*jPtstar+d] - Shifts[3*iL+d];
            }
            double nds = sqrt(dot(ds,ds));
            //vec3 totForce = {0,0,0};
            //vec3 totTorq = {0,0,0};
            for (int iPt=_chebStart[iFib]; iPt < _chebStart[iFib]+_NCheb[iFib]; iPt++){
                for (int jPt=_chebStart[jFib]; jPt < _chebStart[jFib]+_NCheb[jFib]; jPt++){
                    // Multiplication due to rest length and densities
                    double factor = (1.0-_restlen/nds)*deltah1[iPt-_chebStart[iFib]]*deltah2[jPt-_chebStart[jFib]]*rnorm;
                    vec3 forceij;//, force1, force2, X1, X2;
                    for (int d =0; d < 3; d++){
                        forceij[d] = -_Kspring*(chebPoints[3*iPt+d] -  chebPoints[3*jPt+d] - Shifts[3*iL+d])*factor;
                        #pragma omp atomic update
                        CLForceDensities[3*iPt+d] += forceij[d]*_weights[jPt];
                        //force1[d] = forceij[d]*_weights[jPt];
                        //X1[d] = chebPoints[3*iPt+d];
                        #pragma omp atomic update
                        CLForceDensities[3*jPt+d] -= forceij[d]*_weights[iPt];
                        //force2[d] = -forceij[d]*_weights[iPt];
                        //X2[d] = chebPoints[3*jPt+d] + Shifts[3*iL+d];
                    }
                    /*vec3 torq1 = {X1[1]*force1[2]-X1[2]*force1[1], X1[2]*force1[0]-X1[0]*force1[2],X1[0]*force1[1]-X1[1]*force1[0]};
                    vec3 torq2 = {X2[1]*force2[2]-X2[2]*force2[1], X2[2]*force2[0]-X2[0]*force2[2],X2[0]*force2[1]-X2[1]*force2[0]};
                    for (int d =0; d < 3; d++){
                        totTorq[d]+=torq1[d]*_weights[iPt]+torq2[d]*_weights[jPt];
                        totForce[d]+=force1[d]*_weights[iPt]+force2[d]*_weights[jPt];
                    }*/
                }
            }
            //std::cout << "Total force due to link " << iL << "(" << totForce[0] << " , " << totForce[1] << " , " << totForce[2] << std::endl;
            //std::cout << "Total torque due to link " << iL << "(" << totTorq[0] << " , " << totTorq[1] << " , " << totTorq[2] << std::endl;
        }
        // Convert to force
        vec CLForces(pyChebPoints.shape()[0]*3,0.0);
        ForceDensityToForce(CLForceDensities,CLForces);
        return make1DPyArray(CLForces);
    }
    
    py::array calcCLForcesEnergy(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints){
        /**
        Compute forces on the fibers from the list of links. 
        @param iPts = nLinks vector of uniform point numbers where one CL end is
        @param jPts = nLinks vector of uniform point numbers where the other end is
        @param pyShifts = 2D numpy array of shifts in link displacements due to periodicity
        @param pyuniPoints = 2D numpy array of uniform fiber points 
        @param py chebPoints = 2D numpy array of Chebyshev fiber points
        @return CLForces = force densities on all fibers (row stacked) due to CLs
        
        This method is distinct from the one above it because it uses a well-defined energy functional
        that is a spring energy between the points iPt and jPt that are linked. 
        **/
        
        // Copying input
        vec Shifts(pyShifts.size());
        vec uniPoints(pyUnipoints.size());
        vec chebPoints(pyChebPoints.size());

        // copy py::array -> std::vector
        std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
        std::memcpy(uniPoints.data(),pyUnipoints.data(),pyUnipoints.size()*sizeof(double));
        std::memcpy(chebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));

        // Pure C++ function
        vec CLForces(pyChebPoints.shape()[0]*3,0.0);
        #pragma omp parallel for num_threads(_nThreads)
        for (uint iL=0; iL < iPts.size(); iL++){
            int iPtstar = iPts[iL];
            int iuPtMod = iPtstar % _Nuniform;
            int jPtstar = jPts[iL];
            int juPtMod = jPtstar % _Nuniform;
            int iFib = _FibFromSiteIndex[iPtstar];
            int jFib = _FibFromSiteIndex[jPtstar];
            //std::cout << "Link " << iL << " (" << iPtstar << "," << jPtstar << ") on fibers (" << iFib << "," << jFib << ") \n";
            // Displacement vector
            vec3 ds;
            for (int d =0; d < 3; d++){
                ds[d] = uniPoints[3*iPtstar+d] -  uniPoints[3*jPtstar+d] - Shifts[3*iL+d];
            }
            double nds = normalize(ds);
            // Multiplication due to rest length and densities
            vec3 forceij;//, force1, force2, X1, X2;
            for (int d =0; d < 3; d++){
                forceij[d] = -_Kspring*(nds-_restlen)*ds[d];
            }
            for (int iPt=0; iPt < _NCheb[iFib]; iPt++){
                for (int d =0; d < 3; d++){
                    #pragma omp atomic update
                    CLForces[3*(iPt+_chebStart[iFib])+d] += forceij[d]*_RUniform[iuPtMod*_NCheb[iFib]+iPt];
                }
            }
            for (int jPt=0; jPt < _NCheb[jFib]; jPt++){
                for (int d =0; d < 3; d++){
                    #pragma omp atomic update
                    CLForces[3*(_chebStart[jFib]+jPt)+d] -= forceij[d]*_RUniform[juPtMod*_NCheb[jFib]+jPt];
                }
            }
        }
        return make1DPyArray(CLForces);
    }
        
    npDoub calcCLStressEnergy(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints){
        /**
        Compute force densities on the fibers from the list of links. 
        @param iPts = nLinks vector of uniform point numbers where one CL end is
        @param jPts = nLinks vector of uniform point numbers where the other end is
        @param pyShifts = 2D numpy array of shifts in link displacements due to periodicity
        @param pyuniPoints = 2D numpy array of uniform fiber points 
        @param py chebPoints = 2D numpy array of Chebyshev fiber points
        @return [1,0] component of stress due to CLs
        **/
        vec Shifts(pyShifts.size());
        vec uniPoints(pyUnipoints.size());
        vec chebPoints(pyChebPoints.size());

        // copy py::array -> std::vector
        std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
        std::memcpy(uniPoints.data(),pyUnipoints.data(),pyUnipoints.size()*sizeof(double));
        std::memcpy(chebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        vec stress(9);
        #pragma omp parallel for num_threads(_nThreads)
        for (uint iL=0; iL < iPts.size(); iL++){
            int iPtstar = iPts[iL];
            int iuPtMod = iPtstar % _Nuniform;
            int jPtstar = jPts[iL];
            int juPtMod = jPtstar % _Nuniform;
            int iFib = _FibFromSiteIndex[iPtstar];
            int jFib = _FibFromSiteIndex[jPtstar];
            //std::cout << "Link " << iL << " (" << iPtstar << "," << jPtstar << ") on fibers (" << iFib << "," << jFib << ") \n";
            // Displacement vector
            vec3 ds;
            for (int d =0; d < 3; d++){
                ds[d] = uniPoints[3*iPtstar+d] -  uniPoints[3*jPtstar+d] - Shifts[3*iL+d];
            }
            double nds = normalize(ds);
            // Multiplication due to rest length and densities
            vec3 Force;//, force1, force2, X1, X2;
            for (int d =0; d < 3; d++){
                Force[d] = -_Kspring*(nds-_restlen)*ds[d];
            }
            for (int iPt=0; iPt < _NCheb[iFib]; iPt++){
                for (int iD=0; iD < 3; iD++){
                    for (int jD=0; jD < 3; jD++){   
                        #pragma omp atomic update
                        stress[3*iD+jD]-=chebPoints[3*(iPt+_chebStart[iFib])+iD]*Force[jD]*_RUniform[iuPtMod*_NCheb[iFib]+iPt]; // Fiber i
                        #pragma omp atomic update
                        stress[3*iD+jD]-=(chebPoints[3*(iPt+_chebStart[jFib])+iD]+Shifts[3*iL+iD])*-1.0*Force[jD]*_RUniform[juPtMod*_NCheb[jFib]+iPt]; // Fiber j
                    }
                }
            }
        }
        return makePyDoubleArray(stress); 
    }

    private:
    
    vec _sUniform, _sChebyshev, _weights, _sigma;
    vec _RUniform, _WTilde;
    intvec _NCheb, _chebStart, _FibFromSiteIndex;
    int _Nuniform, _nThreads;
    double _Kspring, _restlen;

    double deltah(double a, int iFib){
        /**
        Regularized delta function for the CL force density calculations. 
        @param a = distance between two fiber points
        @return value of delta_h(a,sigma). Right now a Gaussian with standard dev = sigma
        **/
        return 1.0/sqrt(2.0*M_PI*_sigma[iFib]*_sigma[iFib])*exp(-a*a/(2.0*_sigma[iFib]*_sigma[iFib]));
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
    
    void ForceDensityToForce(const vec &ForceDensity, vec &Forces){
        #pragma omp parallel for num_threads(_nThreads)
        for (int iFib = 0; iFib < _NCheb.size(); iFib++){
            vec LocalForces(3*_NCheb[iFib]);
            vec LocalForceDen(3*_NCheb[iFib]);
            for (int i=0; i < 3*_NCheb[iFib]; i++){
                LocalForceDen[i] = ForceDensity[3*_chebStart[iFib]+i];
            }
            BlasMatrixProduct(3*_NCheb[iFib],3*_NCheb[iFib],1,1.0,0.0,_WTilde,false,LocalForceDen,LocalForces);
            for (int i=0; i < 3*_NCheb[iFib]; i++){
                Forces[3*_chebStart[iFib]+i] = LocalForces[i];
            }
        }
    }
    
};

PYBIND11_MODULE(CrossLinkForceEvaluator, m) {
    py::class_<CrossLinkForceEvaluator>(m, "CrossLinkForceEvaluator")
        .def(py::init<int,vec, npDoub, npDoub, intvec, intvec, vec, vec, vec, double, double, int>())
        .def("calcCLForces", &CrossLinkForceEvaluator::calcCLForces)
        .def("calcCLForcesEnergy", &CrossLinkForceEvaluator::calcCLForcesEnergy)
        .def("calcCLStressEnergy", &CrossLinkForceEvaluator::calcCLStressEnergy);
}


