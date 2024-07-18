#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include "DomainC.cpp"
#include <omp.h>
#include "utils.h"

//This is a class of C++ functions that are being used
//for Ewald splitting (and other RPY calculations)
// See end of file for python binding code

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class RPYKernelEvaluator {

    public:
    
    RPYKernelEvaluator(double ain, double muin, int NEwaldPtsin, vec3 Lengths):_Dom(Lengths){
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
        outFront = 1.0/(6.0*M_PI*mu*a);
    }
    
    npDoub RPYVelocityFreeSpace(npDoub pyPoints, npDoub pyForces, int nThreads){
        /**
        RPY velocity based on free space points and forces
        **/
        // Copy  py -> cpp
        // allocate std::vector (to pass to the C++ function)
        vec Points(pyPoints.size());
        vec Forces(pyForces.size());

        // copy py::array -> std::vector
        int nPts = Points.size()/3;
        std::memcpy(Points.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec TotalVelocity = vec(nPts*3); 
        #pragma omp parallel for num_threads(nThreads)
        for (int iPt=0; iPt < nPts; iPt++){
            for (int jPt=0; jPt < nPts; jPt++){
                vec3 rvec, force, uthis;
                for (int d=0; d < 3; d++){
                    rvec[d]=Points[3*iPt+d]-Points[3*jPt+d];
                    force[d]=Forces[3*jPt+d];
                }
                RPYTot(rvec, force, uthis);
                for (int d=0; d < 3; d++){
                    # pragma omp atomic update
                    TotalVelocity[3*iPt+d]+= uthis[d];
                }
            } // end jPt loop
        } // end iPt loop
        // Return 2D numpy array
        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { pyPoints.shape()[0] , 3 };
        std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

        // return 2-D NumPy array
        return py::array(py::buffer_info(
        TotalVelocity.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
        ));
    }
    
    npDoub RPYMatrixFreeSpace(npDoub pyPoints, int nThreads){
        /**
        Python wrapper for the RPY kernel evaluations
        **/
        // Copy  py -> cpp
        // allocate std::vector (to pass to the C++ function)
        vec Points(pyPoints.size());

        // copy py::array -> std::vector
        int nPts = Points.size()/3;
        std::memcpy(Points.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        
        vec MRPYDirect = vec(nPts*3*3*nPts); 
        #pragma omp parallel for num_threads(nThreads)
        for (int iPt = 0; iPt < nPts; iPt++){
            for (int jPt = 0; jPt < nPts; jPt++){
                vec MPair(9);
                vec3 rvec;
                for (int iD=0; iD < 3; iD++){
                    rvec[iD]=Points[3*iPt+iD]-Points[3*jPt+iD];
                }
                PairRPYMatrix(rvec,MPair);
                // Copy the pairwise matrix into the big matrix
                for (int iD=0; iD < 3; iD++){
                    for (int jD=0; jD < 3; jD++){
                        MRPYDirect[3*nPts*(3*iPt+iD)+3*jPt+jD]=MPair[3*iD+jD];
                        MRPYDirect[3*nPts*(3*jPt+jD)+3*iPt+iD]=MPair[3*iD+jD];
                    }
                }
            }
        }
        // Return 2D numpy array
        ssize_t              ndim    = 2;
        uint nPtsThr =  Points.size();
        uint stride =  sizeof(double)*nPtsThr;
        std::vector<ssize_t> shape   = { nPtsThr,nPtsThr};
        std::vector<ssize_t> strides = { stride, sizeof(double) };

        // return 2-D NumPy array
        return py::array(py::buffer_info(
        MRPYDirect.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
        ));
    }
    
    npDoub RPYNKerPairs(npInt PyPairPts, npDoub pyPoints, npDoub pyForces, double xi, 
                          double g, double rcut, int nThreads){
        /**
        Python wrapper for the near field kernel evaluations, done in pairs. 
        @param PyPairPts = 2D integer numpy array of pairs of points, where each row represents a
        pair of indices that interact
        @param pyPoints = 2D numpy array of the Chebyshev points 
        @param pyForces = 2D numpy array of the forces at the Chebyshev points
        @param xi =  Ewald parameter
        @param g = strain in the coordinate system
        @param rcut = cutoff distance for near field Ewald
        @param nThreads = number of threads for the calculation
        @return 2D numpy array of the near field velocities 
        **/
        // Copy  py -> cpp
        // allocate std::vector (to pass to the C++ function)
        int Npairs = PyPairPts.shape()[0];
        intvec PairPts(PyPairPts.size());
        vec Points(pyPoints.size());
        vec Forces(pyForces.size());

        // copy py::array -> std::vector
        std::memcpy(PairPts.data(),PyPairPts.data(),PyPairPts.size()*sizeof(int));
        std::memcpy(Points.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec NearFieldVelocity = vec(NEwaldPts*3); 
        // Self terms
        double nearSelfMobility = outFront*Fnear(0.0,xi);
        #pragma omp parallel num_threads(nThreads)
        {
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
            _Dom.calcShifted(rvec,g);
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
        // Return 2D numpy array
        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { pyPoints.shape()[0] , 3 };
        std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

        // return 2-D NumPy array
        return py::array(py::buffer_info(
        NearFieldVelocity.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
        ));
    }

    void RPYNKer(vec3 &rvec, const vec3 &force, double xi, vec3 &unear){
        /**
        Near field velocity for a single pair of points
        @param rvec = displacement vector between the points
        @param force = force at one of the points 
        @param xi = Ewald splitting parameter
        @param unear = 3 array with the near field velocity M_near*f
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
    
    // allocate py::array (to pass the result of the C++ function to Python)
    py::array py_RPYNearKernel(vec3 &rvec,const vec3 &force, double xi){
        /**
        Python wrapper for a single near field evaluation. 
        @param rvec = displacement vector between the points
        @param force = the force at the point
        @param xi = Ewald parameter
        **/

        vec3 unear;
        RPYNKer(rvec,force,xi,unear);

        // allocate py::array (to pass the result of the C++ function to Python)
        auto pyArray = py::array_t<double>(3);
        auto result_buffer = pyArray.request();
        double *result_ptr    = (double *) result_buffer.ptr;
        // copy std::vector -> py::array
        std::memcpy(result_ptr,unear.data(),unear.size()*sizeof(double));

        return pyArray;
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
    
    void PairRPYMatrix(vec3 &rvec, vec &M3){
        /**
        Total RPY kernel M_ij
        **/
        double r = normalize(rvec);
        double F = FtotRPY(r);
        double G = GtotRPY(r);
        for (int iD=0; iD < 3; iD++){
            M3[3*iD+iD] = mu_inv*F;  
            for (int jD = 0; jD < 3; jD++){
                M3[3*iD+jD]+= mu_inv*((G-F)*rvec[iD]*rvec[jD]);  
            }
        }
    }     
    
    private:
    double a, mu, mu_inv, outFront; // hydrodynamic radius
    double sqrtpi = sqrt(M_PI);
    int NEwaldPts;
    DomainC _Dom;


    //================================================
    // NEAR FIELD KERNEL EVALUATION
    //================================================

    //The split RPY near field can be written as M_near = F(r,xi,a)*I+G(r,xi,a)*(I-RR)
    //The next 2 functions are those F and G functions.
    double Fnear(double r, double xi){
        /**
        Part of the RPY near field that multiplies I. 
        @param r = distance between points 
        @param xi = Ewald parameter
        @return value of F, so that M_near = F*(I-RR^T) + G*RR^T
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
        @return value of G, so that M_near = F*(I-RR^T) + G*RR^T
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



    //================================================
    // PLAIN UNSPLIT RPY KERNEL EVALUATION
    //================================================
    //The RPY kernel can be written as M_RPY = F*(I-RR^T) + G*RR^T
    //The next 2 functions are those Ft and Gt functions.
    double FtotRPY(double r){
        /**
        Part of the RPY kernel that multiplies I. 
        @param r = distance between points 
        @return value of F, so that M_RPY = F*(I-RR^T) + G*RR^T
        **/
	    if (r > 2*a){
            return (2.0*a*a+3.0*r*r)/(24*M_PI*r*r*r);
        }
        return (32.0*a-9.0*r)/(192.0*a*a*M_PI);
    }

    double GtotRPY(double r){
        /**
        @param r = distance between points 
        @return value of G, so that M_RPY = F*(I-RR^T) + G*RR^T
        **/
	    if (r > 2*a){
            return (-2.0*a*a+3.0*r*r)/(12*M_PI*r*r*r);
        }
        return (16.0*a-3.0*r)/(96.0*a*a*M_PI);
    }
};

PYBIND11_MODULE(RPYKernelEvaluator, m) {
    py::class_<RPYKernelEvaluator>(m, "RPYKernelEvaluator")
        .def(py::init<double,double, int,vec3>())
        .def("EvaluateRPYNearPairs", &RPYKernelEvaluator::RPYNKerPairs)
        .def("RPYNearKernel", &RPYKernelEvaluator::py_RPYNearKernel)
        .def("RPYVelocityFreeSpace", &RPYKernelEvaluator::RPYVelocityFreeSpace)
        .def("RPYMatrixFreeSpace", &RPYKernelEvaluator::RPYMatrixFreeSpace);
}

