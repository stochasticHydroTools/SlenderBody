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
    
    StericForceEvaluatorC(int nFib, int NCheb, int Nuni, npDoub su, npDoub pyRuni, vec3 DomLengths,int nThr):_Dom(DomLengths){
        /* Constructor. Inputs are the number of fibers, number of Cheb pts per fiber, number of 
        uniform points (typically 1/eps) at which we resample, arclength coordinates (su) of those points, 
        resampling matrix (pyRUni), periodic domain lengths (DomLengths), and number of threads for parallel processing
        */
        _NFib = nFib;
        _nXPerFib = NCheb;
        _nUni = Nuni;
        _RUniform = vec(pyRuni.size());
        std::memcpy(_RUniform.data(),pyRuni.data(),pyRuni.size()*sizeof(double));
        _su = vec(su.size());
        std::memcpy(_su.data(),su.data(),su.size()*sizeof(double));
        _deltasu = _su[1]-_su[0];
        _nThreads = nThr;  
    }  
    
    void SetForceFcnParameters(double dcrit, double delta, double F0, double dcut){
        /*
        The force (per unit area) here is 
        F(r) = F_0 when r < dcrit and F_0*exp((d-r)/delta) when r > dcrit. 
        The truncation distance _dcut is also a parameter. When r > dcut, the force is zero. 
        */
        _F0 = F0;
        _delta = delta;
        _dcrit = dcrit;
        _dcut = dcut;
    }
    
    void initSegmentVariables(double L, int Nsegments, double SegmentCutoff, npDoub pyRMidPoint, npDoub pyREndPoint){
        /*
        Variables needed for segments. 
        All are pretty obvious. L = fiber length. The variable SegmentCutoff is the distance at which two 
        segments are further processed (could have points which contribute to the potential).
        The matrices  pyRMidPoint and pyREndPoint are resampling matrices for the midpoint and endpoint of the segments. 
        */
        _NsegPerFib = Nsegments;
        _L = L;
        _Lseg = _L/_NsegPerFib;
        _RMidpoint = vec(pyRMidPoint.size());
        _REndpoint = vec(pyREndPoint.size());
        _SegmentCutoff = SegmentCutoff;
        std::memcpy(_REndpoint.data(),pyREndPoint.data(),pyREndPoint.size()*sizeof(double));
        std::memcpy(_RMidpoint.data(),pyRMidPoint.data(),pyRMidPoint.size()*sizeof(double));
    }
    
    void initInterpolationVariables(npDoub sCheb, npDoub ValsToCoeffs, int nQuadSegs, double dsQuad){
        /*
        Interpolation and quadrature variables. Chebyshev nodes, matrix that maps the values on the Cheb nodes to the
        coefficients. Then nQuadSegs = number of quadrature points when we integrate over segments, 
        dsQuad = the spacing between them.
        */
        _sC = vec(_nXPerFib);
        std::memcpy(_sC.data(),sCheb.data(),sCheb.size()*sizeof(double));
        _ValsToCoeffs = vec(ValsToCoeffs.size());
        std::memcpy(_ValsToCoeffs.data(),ValsToCoeffs.data(),ValsToCoeffs.size()*sizeof(double));
        _nQuadSegs = nQuadSegs;   
        _dsQuad = dsQuad;         
    }
        

    py::array getUniformPoints(npDoub pyPoints, char type){
        /*
        Uniform points from Chebyshev points pyPoints. The char type has three possibilities:
        'u' = _nUni uniform points (typically 1/eps)
        'e' = endpoints of segments 
        'm' = midpoints of segments
        */
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        int nUni; 
        if (type=='u'){
            nUni = _nUni;
        } else if (type=='e'){
            nUni = _NsegPerFib+1;
        } else if (type=='m'){
            nUni = _NsegPerFib;
        } else {
            throw std::invalid_argument("Uniform points needs type, u (uniform), e (endpoints), m (midpoints)");
        }
        vec AllUniPoints(3*_NFib*nUni);
        #pragma omp parallel for num_threads(_nThreads)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalUniPts(3*nUni);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            if (type=='u'){
                BlasMatrixProduct(nUni,_nXPerFib,3,1.0,0.0,_RUniform,false,FiberPoints,LocalUniPts);
            } else if (type=='e'){
                BlasMatrixProduct(nUni,_nXPerFib,3,1.0,0.0,_REndpoint,false,FiberPoints,LocalUniPts);
            } else if (type=='m'){
                BlasMatrixProduct(nUni,_nXPerFib,3,1.0,0.0,_RMidpoint,false,FiberPoints,LocalUniPts);
            }
            for (int i=0; i < 3*nUni; i++){
                AllUniPoints[3*nUni*iFib+i] = LocalUniPts[i];
            }
        }
        return makePyDoubleArray(AllUniPoints);
    }
    
    py::array ForcesFromGlobalUniformBlobs(npInt pyUnipairs, npDoub pyUniPts, double g){
        /*
        Find steric forces. This implementation corresponds to the BASE CLASS in StericForceEvaluator.py
        There is a list of potentially interacting pairs pyUnipairs that comes in, 
        along with the uniform points and the strain in the domain. This code then loops through the list, 
        eliminates points which are either too close on the same fiber. Then it will give force on 
        pairs that are touching. 
        */
        // Convert numpy to C++ vector
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
            double ds = abs(_su[iuPtMod]-_su[juPtMod]);
            // Donev: Why are you removing close points here? A fiber can interact with itself.
            // What is the problem with not removing these points?
            // Remove points that are close (here too close means on the same fiber and the 
            // arclength between the points is 1.1*dcut). Not setting = dcut to avoid rounding. 
            bool isEligible=true;
            if (iFib==jFib && ds < 1.1*_dcut){
                isEligible = false;
            }
            if (isEligible){
                vec3 rvec;
                for (int d=0; d < 3; d++){
                    rvec[d] = uniPts[3*iUniPt+d]-uniPts[3*jUniPt+d];
                }
                // Periodic shift to find periodic copies that are actually interacting
                _Dom.calcShifted(rvec,g);
                double nr = normalize(rvec);
                // Compute the potential at r 
                double ForceMag = dUdr(nr);
                // Multiplication due to rest length and densities
                vec3 forceij;//, force1, force2, X1, X2;
                for (int d =0; d < 3; d++){
                    forceij[d] = -ForceMag*rvec[d]*_deltasu*_deltasu;
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
        }
        return make1DPyArray(StericForces);
    }
    
    py::array ForceFromCloseSegments(npInt pySegpairs, npDoub pyXCheb, npDoub pyMidpoints, npDoub pyEndpoints, double g){
        /*
        Find steric forces from SEGMENTS. This is a fairly complicated algorithm. The inputs are the pairs 
        of segments which could be interacting (comes from neighbor search), the Chebyshev points of all fibers,
        the midpoints of the segments, the endpoints of the segments, and the strain in the domain. 
        */
        // Convert numpy to C++ vector
        intvec SegmentPairs(pySegpairs.size());
        std::memcpy(SegmentPairs.data(),pySegpairs.data(),pySegpairs.size()*sizeof(int));
        vec ChebPts(pyXCheb.size());
        std::memcpy(ChebPts.data(), pyXCheb.data(),pyXCheb.size()*sizeof(double));
        vec SegMPs(pyMidpoints.size());
        std::memcpy(SegMPs.data(),pyMidpoints.data(),pyMidpoints.size()*sizeof(double));
        vec SegEPs(pyEndpoints.size());
        std::memcpy(SegEPs.data(),pyEndpoints.data(),pyEndpoints.size()*sizeof(double));

        int nPairs = SegmentPairs.size()/2;
        int nGood=0;
        vec StericForces(_NFib*_nXPerFib*3);
        for (int iPair=0; iPair < nPairs; iPair++){
            // STEP 1: Check that the midpoints were not included because of safety factors
            // in the shear cell 
            int iSeg = SegmentPairs[2*iPair];
            int jSeg = SegmentPairs[2*iPair+1];
            vec3 Start1, End1, Start2, End2, dMP;
            // Determine correct periodic copy
            for (int id=0; id < 3; id++){
                dMP[id] = SegMPs[3*iSeg+id]-SegMPs[3*jSeg+id];
            }
            _Dom.calcShifted(dMP,g);
            double ract = sqrt(dot(dMP,dMP)); 
            if (ract < _SegmentCutoff){ 
                // Donev: I am confused why this is not being done in the nList code itself
                // That is, points/pairs should only be added to the list if they are within
                // the cutoff in Euclidean space, to avoid wasting memory traffic on other pairs
                // Why does client code for the neighbor search need to repeat this?
                // ONLY PROCEED IF ract < actual cutoff (remove extra safety for sheared cell)
                vec3 PeriodicShift;
                for (int d =0; d < 3; d++){
                    PeriodicShift[d] = SegMPs[3*iSeg+d]-SegMPs[3*jSeg+d]-dMP[d];
                }
                int iFib = iSeg/_NsegPerFib;
                int iSegMod = iSeg-iFib*_NsegPerFib;
                int jFib = jSeg/_NsegPerFib;
                int jSegMod = jSeg-jFib*_NsegPerFib;
                int StartSeg1 = iFib*(_NsegPerFib+1) + iSegMod;
                int StartSeg2 = jFib*(_NsegPerFib+1) + jSegMod;
                for (int id=0; id < 3; id++){
                    Start1[id] = SegEPs[3*StartSeg1+id];
                    End1[id] = SegEPs[3*StartSeg1+3+id];
                    Start2[id] = SegEPs[3*StartSeg2+id]+PeriodicShift[id];
                    End2[id] = SegEPs[3*StartSeg2+3+id]+PeriodicShift[id];
                }
                vec ClosePts(2);
                // Compute closest distance between segments. This function returns the distance
                // and the points on the segments (ClosePts) where the nearest point of interaction is. 
                double segDist = DistanceBetweenSegments(Start1,End1,Start2,End2,ClosePts);
                //std::cout << "Segment " << iSeg << " and " << jSeg << std::endl;
                //std::cout << segDist << " , " << ClosePts[0] << " , " << ClosePts[1] << std::endl;
                // TODO: Add a step here which uses Newton when the straight segments are close enough. 
                // Now segments are close enough: compute the forces  
                if (segDist < _dcut){
                    // Identify the two fibers (for resampling near the closest point)
                    vec XCheb1(3*_nXPerFib);
                    vec XCheb2(3*_nXPerFib);
                    for (int iPt=0; iPt < _nXPerFib; iPt++){
                        for (int iD=0; iD < 3; iD++){
                            XCheb1[3*iPt+iD] = ChebPts[iFib*3*_nXPerFib+3*iPt+iD];
                            XCheb2[3*iPt+iD] = ChebPts[jFib*3*_nXPerFib+3*iPt+iD];
                        }
                    }
                    // Donev: This is too simplistic of a code
                    // In the worst case when all the fibers are parallel and close by the cost
                    // can be quadratic. Number of points to put down and pieces must depend on geometry in some way (nontrivial)
                    // Put down a quadrature grid around the closest points. The grid has 
                    // spacing _dsQuad and number of points _nQuadSegs. Here I am putting the 
                    // grid around the closest point (e.g, -3, -1, 1, 3 instead of -2, 0, 2)
                    // to avoid putting a quadrature point at the endpoint of the segment.
                    // This prevents double counting segments.  
                    vec Seg1Qpts, Seg2Qpts;
                    for (int q=0; q < _nQuadSegs; q++){
                        double Seg1Pt = (-_nQuadSegs*0.5+0.5+q)*_dsQuad/_Lseg+ClosePts[0];
                        double Seg2Pt = (-_nQuadSegs*0.5+0.5+q)*_dsQuad/_Lseg+ClosePts[1];
                        // If the quadrature point falls on the segment, add it to the list. 
                        // Using push back; might be slow (check later)
                        if (Seg1Pt > 0 && Seg1Pt < 1){
                            Seg1Qpts.push_back(Seg1Pt);
                        } if (Seg2Pt > 0 && Seg2Pt < 1){
                            Seg2Qpts.push_back(Seg2Pt);
                        }
                    }
                    // Now do O(N^2) quadrature over the segment points
                    // For each segment point on the quadrature grid, I compute the 
                    // interpolation row (InterpRow), which is N x 1 and gives me the 
                    // row that samples X at the particular quadrature point
                    // then do the force calculation
                    // Force(a) = -\sum_{q} InterpRow_{a} *ds*ds*dUdr(r)*rhat
                    for (uint iS1Pt=0; iS1Pt < Seg1Qpts.size(); iS1Pt++){
                        double siqp = Seg1Qpts[iS1Pt]*_Lseg+iSegMod*_Lseg;
                        vec InterpRow1(_nXPerFib);
                        InterpolationRow(siqp,InterpRow1);
                        vec X1qp(3);
                        BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow1,false,XCheb1,X1qp); 
                        for (uint jS2Pt=0; jS2Pt < Seg2Qpts.size(); jS2Pt++){
                            double sjqp = Seg2Qpts[jS2Pt]*_Lseg+jSegMod*_Lseg;
                            vec InterpRow2(_nXPerFib);
                            InterpolationRow(sjqp,InterpRow2);
                            vec X2qp(3);
                            BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow2,false,XCheb2,X2qp); 
                            for (int id=0; id < 3; id++){
                                X2qp[id]+=PeriodicShift[id];
                            }
                            vec3 rvec;
                            for (int d=0; d < 3; d++){
                                rvec[d] = X1qp[d]-X2qp[d];
                            }
                            // Compute force and add to both
                            double nr = normalize(rvec);
                            // Compute the potential at r 
                            double ForceMag = dUdr(nr);
                            // Multiplication due to rest length and densities
                            vec3 forceij;//, force1, force2, X1, X2;
                            for (int d =0; d < 3; d++){
                                forceij[d] = -ForceMag*rvec[d]*_dsQuad*_dsQuad;
                            }
                            for (int iPt=0; iPt < _nXPerFib; iPt++){
                                for (int d =0; d < 3; d++){
                                    #pragma omp atomic update
                                    StericForces[3*(iPt+_nXPerFib*iFib)+d] += forceij[d]*InterpRow1[iPt];
                                }
                            }
                            for (int jPt=0; jPt < _nXPerFib; jPt++){
                                for (int d =0; d < 3; d++){
                                    #pragma omp atomic update
                                    StericForces[3*(jPt+_nXPerFib*jFib)+d] -= forceij[d]*InterpRow2[jPt];
                                }
                            }
                        }
                    }
                        
                }
                nGood++;
            }
        }
        std::cout << "Number good pairs " << nGood << std::endl;
        return make1DPyArray(StericForces);
    }
    
    private:

    // Variables for uniform points
    int _nXPerFib, _nUni, _NFib, _nThreads;
    double _F0, _delta, _dcrit, _dcut, _deltasu;
    vec _RUniform, _su;
    DomainC _Dom;
    // Variables for segments
    int _NsegPerFib, _nQuadSegs;            
    double _SegmentCutoff, _L, _Lseg, _dsQuad;
    vec _sC, _REndpoint, _RMidpoint, _ValsToCoeffs;
    
   double dUdr(double nr){
        /*
        The force here is 
        F(r) = F_0 when r < d and F_0*exp((d-r)/delta) when r > d. 
        d is always the size of the fiber
        There are then 2 free parameters: F_0 and delta. 
        We choose F_0*delta = 4*kBT, so this leaves the choice of delta, the 
        radius at which potential decays. We take delta = diameter, so that the 
        potential decays to exp(-4) \approx 0.02 over 4 diameters. So we truncate 
        at d-r = 4*delta. 
        */
        if (nr > _dcut){
            return 0;
        }
        // The function dU/dr = F_0 when r < dcrit and F_0*exp((dcrit-r)/delta) r > dcrit. 
        // The truncation distance is 4*delta
        if (nr < _dcrit){
            return -_F0;
        } 
        return (-_F0*exp((_dcrit-nr)/_delta));
    }
    
    double DistanceBetweenSegments(const vec3 &Start1,const vec3 &End1,const vec3 &Start2,const vec3 &End2, vec &ClosePts){
        vec3 u, v, w;
        for (int d=0; d < 3; d++){
            u[d]=End1[d]-Start1[d];
            v[d]=End2[d]-Start2[d];
            w[d]=Start1[d]-Start2[d];
        }
        double a, b, c, d, e, D, sD, tD, sN, tN, sc, tc, segDist;
        a = dot(u,u);
        b = dot(u,v);
        c = dot(v,v);
        d = dot(u,w);
        e = dot(v,w);
        D = a*c - b*b;
        sD = D;
        tD = D;
    
        double SMALL_NUM = 0.00000001;
    
        // compute the line parameters of the two closest points
        if (D < SMALL_NUM){  // the lines are almost parallel
            sN = 0.0;        // force using point P0 on segment S1
            sD = 1.0;        // to prevent possible division by 0.0 later
            tN = e;
            tD = c;
        } else {// get the closest points on the infinite lines
            sN = (b*e - c*d);
            tN = (a*e - b*d);
            if (sN < 0.0){   // sc < 0 => the s=0 edge is visible       
                sN = 0.0;
                tN = e;
                tD = c;
            } else if (sN > sD){ // sc > 1 => the s=1 edge is visible
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }
            
        if (tN < 0.0){  // tc < 0 => the t=0 edge is visible
            tN = 0.0;
            // recompute sc for this edge
            if (-d < 0.0){
                sN = 0.0;
            } else if (-d > a) {
                sN = sD;
            } else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD){ //  tc > 1 => the t=1 edge is visible
            tN = tD;
            // recompute sc for this edge
            if ((-d + b) < 0.0){
                sN = 0;
            } else if ((-d + b) > a){
                sN = sD;
            } else {
                sN = (-d + b);
                sD = a;
            }
        }
    
        // Finally do the division to get sc and tc
        if(abs(sN) < SMALL_NUM){
            sc = 0.0;
        } else {
            sc = sN / sD;
        }
    
        if (abs(tN) < SMALL_NUM){
            tc = 0.0;
        } else { 
            tc = tN / tD;
        }
    
        // Get the difference of the two closest points
        vec3 dP;
        for (int d=0; d < 3; d++){
            dP[d]=w[d]+sc*u[d]-tc*v[d];
        }
        segDist = normalize(dP);
        ClosePts[0]=sc;
        ClosePts[1]=tc;
        return segDist;
    }
    
    void InterpolationRow(double s, vec &InterpRow){
        // Get the evaluation row for coefficients
        vec EvalCoeffs(_nXPerFib);
        EvalChebRow(s, _L, EvalCoeffs);
        // Multiply by the values -> coefficient matrix
        BlasMatrixProduct(1, _nXPerFib, _nXPerFib,1.0,0.0,EvalCoeffs,false,_ValsToCoeffs,InterpRow); 
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
        .def(py::init<int,int,int, npDoub, npDoub, vec3, int>())
        .def("getUniformPoints", &StericForceEvaluatorC::getUniformPoints)
        .def("SetForceFcnParameters", &StericForceEvaluatorC::SetForceFcnParameters)
        .def("initSegmentVariables",&StericForceEvaluatorC::initSegmentVariables)
        .def("initInterpolationVariables",&StericForceEvaluatorC::initInterpolationVariables)
        .def("ForcesFromGlobalUniformBlobs", &StericForceEvaluatorC::ForcesFromGlobalUniformBlobs)
        .def("ForceFromCloseSegments",&StericForceEvaluatorC::ForceFromCloseSegments);
}


