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

const double SMALL_NUM = 1e-12;
const int MAX_NEWTON_ITS = 10;
const double DISTANCE_EPS = 1e-10;
const double COND_MAX = 1e3;
const double MAX_DESCENT_NORM_OVERL=0.1;


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
    
    void SetForceFcnParameters(double delta, double F0, double dcut){
        /*
        The force (per unit area) here is 
        F_0*exp(r^2/delta^2)
        The truncation distance _dcut is also a parameter. When r > dcut, the force is zero. 
        */
        _F0 = F0;
        _delta = delta;
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
        _NewtonTol = _delta*0.01;
        _RMidpoint = vec(pyRMidPoint.size());
        _REndpoint = vec(pyREndPoint.size());
        _SegmentCutoff = SegmentCutoff;
        std::memcpy(_REndpoint.data(),pyREndPoint.data(),pyREndPoint.size()*sizeof(double));
        std::memcpy(_RMidpoint.data(),pyRMidPoint.data(),pyRMidPoint.size()*sizeof(double));
    }
    
    double getNewtonTol(){
        return _NewtonTol;
    }
    
    void initInterpolationVariables(npDoub sCheb, npDoub ValsToCoeffs, npDoub DiffMat){
        /*
        Interpolation and quadrature variables. Chebyshev nodes, matrix that maps the values on the Cheb nodes to the
        coefficients.
        */
        _sC = vec(_nXPerFib);
        std::memcpy(_sC.data(),sCheb.data(),sCheb.size()*sizeof(double));
        _ValsToCoeffs = vec(ValsToCoeffs.size());
        std::memcpy(_ValsToCoeffs.data(),ValsToCoeffs.data(),ValsToCoeffs.size()*sizeof(double));
        _DiffMat = vec(DiffMat.size());
        std::memcpy(_DiffMat.data(),DiffMat.data(),DiffMat.size()*sizeof(double));         
    }
    
    void initLegQuadVariables(int NPtsPerStd, npDoub xLeg, npDoub wLeg){
        _NPtsPerStd = NPtsPerStd;
         _LegPts = vec(xLeg.size());
        std::memcpy(_LegPts.data(),xLeg.data(),xLeg.size()*sizeof(double));
        _LegWts = vec(wLeg.size());
        std::memcpy(_LegWts.data(),wLeg.data(),wLeg.size()*sizeof(double));
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
        #pragma omp parallel for num_threads(_nThreads)
        for (int iPair=0; iPair < nPairs; iPair++){
            int iUniPt = PairsToRepel[2*iPair];
            int jUniPt = PairsToRepel[2*iPair+1];
            int iuPtMod = iUniPt % _nUni;
            int juPtMod = jUniPt % _nUni;
            int iFib = iUniPt/_nUni;
            int jFib = jUniPt/_nUni;
            double ds = abs(_su[iuPtMod]-_su[juPtMod]);
            // Remove points that are close (here too close means on the same fiber and the 
            // arclength between the points is 1.1*dcut or less). 
            // Not setting = dcut to avoid rounding. 
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
                double wt1 = _deltasu;
                if (iuPtMod==0 || iuPtMod==(_nUni-1)){
                    wt1=0.5*_deltasu;
                }
                double wt2 = _deltasu;
                if (juPtMod==0 || juPtMod==(_nUni-1)){
                    wt2=0.5*_deltasu;
                }
                _Dom.calcShifted(rvec,g);
                double nr = normalize(rvec);
                // Compute the potential at r 
                double ForceMag = dUdrGaussian(nr);
                // Multiplication due to rest length and densities
                vec3 forceij;//, force1, force2, X1, X2;
                for (int d =0; d < 3; d++){
                    forceij[d] = -ForceMag*rvec[d]*wt1*wt2;
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
    
    
    py::array ForceFromIntervals(npInt pyiFibjFib, npDoub pyXCheb, npDoub pyPeriodicShifts, npDoub pyIntervals){
        /*
        Find steric forces from intervals of interaction
        */
        // Convert numpy to C++ vector
        intvec iFibjFib(pyiFibjFib.size());
        std::memcpy(iFibjFib.data(),pyiFibjFib.data(),pyiFibjFib.size()*sizeof(int));
        vec ChebPts(pyXCheb.size());
        std::memcpy(ChebPts.data(), pyXCheb.data(),pyXCheb.size()*sizeof(double));
        vec PeriodicShifts(pyPeriodicShifts.size());
        std::memcpy(PeriodicShifts.data(),pyPeriodicShifts.data(),pyPeriodicShifts.size()*sizeof(double));
        vec Intervals(pyIntervals.size());
        std::memcpy(Intervals.data(),pyIntervals.data(),pyIntervals.size()*sizeof(double));

        int nPairs = iFibjFib.size()/2;
        vec StericForces(_NFib*_nXPerFib*3);
        #pragma omp parallel for num_threads(_nThreads)
        for (int iPair=0; iPair < nPairs; iPair++){
            // Each pair is a pair of segments. This part identifies the start and endpoint
            // of the segments we're considering, as well as what fiber they're on, etc. 
            int iFib = iFibjFib[2*iPair];
            int jFib = iFibjFib[2*iPair+1];
            vec ClosePts(2);
            // Identify the two fibers
            vec XCheb1(3*_nXPerFib);
            vec XCheb2(3*_nXPerFib);
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                for (int iD=0; iD < 3; iD++){
                    XCheb1[3*iPt+iD] = ChebPts[iFib*3*_nXPerFib+3*iPt+iD];
                    XCheb2[3*iPt+iD] = ChebPts[jFib*3*_nXPerFib+3*iPt+iD]+PeriodicShifts[3*iPair+iD];
                }
            }
            vec Seg1Bds(2), Seg2Bds(2);
            Seg1Bds[0] = Intervals[4*iPair];
            Seg1Bds[1] = Intervals[4*iPair+1];
            Seg2Bds[0] = Intervals[4*iPair+2];
            Seg2Bds[1] = Intervals[4*iPair+3];
            int NumS1Pts = int(_NPtsPerStd*(Seg1Bds[1]-Seg1Bds[0])/_delta)+1;
            int NumS2Pts = int(_NPtsPerStd*(Seg2Bds[1]-Seg2Bds[0])/_delta)+1;
            // Gauss-Legendre integration on segment intervals
            vec Seg1Qpts(NumS1Pts), Seg2Qpts(NumS2Pts), Seg1Wts(NumS1Pts), Seg2Wts(NumS2Pts);
            int Pt1StartIndex = int((1.0+NumS1Pts)*0.5*NumS1Pts+SMALL_NUM)-NumS1Pts;
            int Pt2StartIndex = int((1.0+NumS2Pts)*0.5*NumS2Pts+SMALL_NUM)-NumS2Pts;
            for (int q=0; q < NumS1Pts; q++){
                Seg1Qpts[q]=(_LegPts[Pt1StartIndex+q]+1.0)*0.5*(Seg1Bds[1]-Seg1Bds[0])+Seg1Bds[0];
                Seg1Wts[q]=_LegWts[Pt1StartIndex+q]*0.5*(Seg1Bds[1]-Seg1Bds[0]);
            }
            for (int q=0; q < NumS2Pts; q++){
                Seg2Qpts[q]=(_LegPts[Pt2StartIndex+q]+1.0)*0.5*(Seg2Bds[1]-Seg2Bds[0])+Seg2Bds[0];
                Seg2Wts[q]=_LegWts[Pt2StartIndex+q]*0.5*(Seg2Bds[1]-Seg2Bds[0]);
            }
            // Now do O(N^2) quadrature over the segment points
            // For each segment point on the quadrature grid, I compute the 
            // interpolation row (InterpRow), which is N x 1 and gives me the 
            // row that samples X at the particular quadrature point
            // then do the force calculation
            for (int iS1Pt=0; iS1Pt < NumS1Pts; iS1Pt++){
                double siqp = Seg1Qpts[iS1Pt];
                vec InterpRow1(_nXPerFib);
                InterpolationRow(siqp,InterpRow1);
                vec X1qp(3);
                BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow1,false,XCheb1,X1qp); 
                for (int jS2Pt=0; jS2Pt < NumS2Pts; jS2Pt++){
                    double sjqp = Seg2Qpts[jS2Pt];
                    vec InterpRow2(_nXPerFib);
                    InterpolationRow(sjqp,InterpRow2);
                    vec X2qp(3);
                    BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow2,false,XCheb2,X2qp); 
                    vec3 rvec;
                    for (int d=0; d < 3; d++){
                        rvec[d] = X1qp[d]-X2qp[d];
                    }
                    // Compute force and add to both
                    double nr = normalize(rvec);
                    // Compute the potential at r 
                    double ForceMag = dUdrGaussian(nr);
                    // Multiplication due to rest length and densities
                    vec3 forceij;//, force1, force2, X1, X2;
                    for (int d =0; d < 3; d++){
                        forceij[d] = -ForceMag*rvec[d]*Seg1Wts[iS1Pt]*Seg2Wts[jS2Pt];
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
                } // end loop over fib2 pts
            } // end loop over fib1 pts
        } // end loop over pairs
        //std::cout << "Number of pairs " << nPairs << std::endl;
        //std::cout << "Segments under curve " << nUnderCurve << std::endl;
        //std::cout << "Number good pairs " << nGood << std::endl;
        return make1DPyArray(StericForces);
    }
    
    py::array IntervalsOfInteraction(npInt pySegpairs, npDoub pyXCheb, npDoub pyMidpoints, npDoub pyEndpoints, double g){
        /*
        Find intervals of interaction from possible segments. The inputs are the pairs 
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
        vec InteractionIntervals(9*nPairs,-1); 
        // 9 entries in this array: (iFib, start_i, end_i, jFib, start_j, end_j, PeriodicShift (3))
        #pragma omp parallel for num_threads(_nThreads)
        for (int iPair=0; iPair < nPairs; iPair++){
            // Each pair is a pair of segments. This part identifies the start and endpoint
            // of the segments we're considering, as well as what fiber they're on, etc. 
            int iSeg = SegmentPairs[2*iPair];
            int jSeg = SegmentPairs[2*iPair+1];
            vec3 Start1, End1, Start2, End2, dMP;
            // Determine correct periodic copy
            for (int id=0; id < 3; id++){
                dMP[id] = SegMPs[3*iSeg+id]-SegMPs[3*jSeg+id];
            }
            _Dom.calcShifted(dMP,g);
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
            // Now segments are close enough: compute the forces 
            // Identify the two fibers (for resampling near the closest point)
            vec XCheb1(3*_nXPerFib);
            vec XCheb2(3*_nXPerFib);
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                for (int iD=0; iD < 3; iD++){
                    XCheb1[3*iPt+iD] = ChebPts[iFib*3*_nXPerFib+3*iPt+iD];
                    XCheb2[3*iPt+iD] = ChebPts[jFib*3*_nXPerFib+3*iPt+iD]+PeriodicShift[iD];
                }
            }
            vec SegMP1(3), SegMP2(3), CurvMP1(3), CurvMP2(3), curv1(3), curv2(3);
            for (int d=0; d < 3; d++){
                SegMP1[d] = 0.5*(Start1[d]+End1[d]);
                SegMP2[d] = 0.5*(Start2[d]+End2[d]);
            }
            vec InterpRow1(_nXPerFib), InterpRow2(_nXPerFib);
            // Sample at midpoint to estimate curvature
            double sMP1 = 0.5*_Lseg+iSegMod*_Lseg;
            double sMP2 = 0.5*_Lseg+jSegMod*_Lseg;
            InterpolationRow(sMP1,InterpRow1);
            InterpolationRow(sMP2,InterpRow2);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,XCheb1,CurvMP1);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,XCheb2,CurvMP2);
            for (int d=0; d < 3; d++){
                curv1[d] = SegMP1[d]-CurvMP1[d];
                curv2[d] = SegMP2[d]-CurvMP2[d];
            }
            double curvdist1 = sqrt(dot(curv1,curv1));
            double curvdist2 = sqrt(dot(curv2,curv2));
            // If curved pieces of fibers could be interacting, proceed to Newton solve.
            if (segDist < _dcut+curvdist1+curvdist2){
                // Resolve with nonlinear solve to obtain closest points on segments 
                ClosePts[0]=ClosePts[0]*_Lseg+iSegMod*_Lseg;
                ClosePts[1]=ClosePts[1]*_Lseg+jSegMod*_Lseg;
                double curvDist = DistanceBetweenFiberParts(XCheb1,XCheb2, ClosePts);
                if (curvDist < _dcut){
                    double s1star = ClosePts[0];
                    double s2star = ClosePts[1];
                    InterpolationRow(s1star,InterpRow1);
                    InterpolationRow(s2star,InterpRow2);
                    // Compute tangent vectors, take dot product to estimate additional arclength
                    vec DX1(3*_nXPerFib), DX2(3*_nXPerFib), DSqX1(3*_nXPerFib), DSqX2(3*_nXPerFib);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,XCheb1,DX1);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,XCheb2,DX2);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX1,DSqX1);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX2,DSqX2);
                    vec pt1(3), pt2(3), tau1(3), tau2(3), curv1(3), curv2(3);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,XCheb1,pt1);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,XCheb2,pt2);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DX1,tau1);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DX2,tau2);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DSqX1,curv1);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DSqX2,curv2);
                    vec disp(3);
                    for (int d=0; d<3; d++){
                        disp[d]=pt1[d]-pt2[d];
                    }
                    // Approximation of distance 
                    // distance^2 - r^2 = a s1^2 + bs2^2 + 2 e s1 s2 + 2 c s1 + 2 d s2 + f
                    double aa = dot(tau1,tau1)+dot(disp,curv1);
                    double bb = dot(tau2,tau2)-dot(disp,curv2);
                    double cc = dot(disp,tau1); // zero unless @ EP
                    double dd = -dot(disp,tau2); // zero unless @ EP
                    double ee = -dot(tau1,tau2);
                    double ff = dot(disp,disp)-_dcut*_dcut;
                    double s1disc = sqrt(pow(8*dd*ee-8*cc*bb,2)-4*(4*ee*ee-4*aa*bb)*(4*dd*dd-4*bb*ff));
                    double s1plus = ((8*cc*bb-8*dd*ee)+s1disc)/(8*ee*ee-8*aa*bb);
                    double s1minus= ((8*cc*bb-8*dd*ee)-s1disc)/(8*ee*ee-8*aa*bb);
                    double Deltas1 = std::max(abs(s1plus),abs(s1minus));
                    double s2disc = sqrt(pow(8*cc*ee-8*aa*dd,2)-4*(4*ee*ee-4*aa*bb)*(4*cc*cc-4*aa*ff));
                    double s2plus = ((8*dd*aa-8*cc*ee)+s2disc)/(8*ee*ee-8*aa*bb);
                    double s2minus = ((8*dd*aa-8*cc*ee)-s2disc)/(8*ee*ee-8*aa*bb);
                    double Deltas2 = std::max(abs(s2minus),abs(s2plus));
                    InteractionIntervals[9*iPair] = iFib;
                    InteractionIntervals[9*iPair+1] = std::max(s1star-Deltas1,0.0);
                    InteractionIntervals[9*iPair+2] = std::min(s1star+Deltas1,_L);
                    InteractionIntervals[9*iPair+3] = jFib;
                    InteractionIntervals[9*iPair+4] = std::max(s2star-Deltas2,0.0);
                    InteractionIntervals[9*iPair+5] = std::min(s2star+Deltas2,_L);
                    for (int d=0; d < 3; d++){
                        InteractionIntervals[9*iPair+6+d]=PeriodicShift[d];
                    }
                } // end if curv dist < cutoff
            } // end if linear distance < cutoff + safety
        } // end loop over pairs
        //std::cout << "Number of pairs " << nPairs << std::endl;
        //std::cout << "Segments under curve " << nUnderCurve << std::endl;
        //std::cout << "Number good pairs " << nGood << std::endl;
        return makePyByNineArray(InteractionIntervals);
    }
    
    py::array MergeRepeatedIntervals(npDoub pyRepeatedPairs, npInt pynAdditionalIntervals){
        /*
        Merge repeated intervals of interaction. The inputs are RepeatedPairs, which is a
        list of intervals. The first N intervals are those that repeat after entry N. 
        The entries in nAdditionalIntervals, which is an N x 1 array, tell us how many times
        each of the repeated pairs repeats (after entry N). 
        Each row of RepeatedPairs has 9 entries in this array: (iFib, start_i, end_i, jFib, start_j, end_j, PeriodicShift (3))
        */
        // Convert numpy to C++ vector
        vec RepeatedPairs(pyRepeatedPairs.size());
        std::memcpy(RepeatedPairs.data(), pyRepeatedPairs.data(),pyRepeatedPairs.size()*sizeof(double));
        intvec nAdditionalIntervals(pynAdditionalIntervals.size());
        std::memcpy(nAdditionalIntervals.data(),pynAdditionalIntervals.data(),pynAdditionalIntervals.size()*sizeof(int));

        int nPairs = nAdditionalIntervals.size();
        vec NewPairs;
        int StartRepeats=nPairs;
        for (int iPair=0; iPair < nPairs; iPair++){
            // Sort intervals by increasing x1
            int nTotalInts = 1+nAdditionalIntervals[iPair];
            intvec PairInts(nTotalInts,iPair);
            vec Starts1s(nTotalInts,RepeatedPairs[9*iPair+1]);
            for (int iInt=0; iInt < nAdditionalIntervals[iPair]; iInt++){
                PairInts[iInt+1]= StartRepeats+iInt;
                Starts1s[iInt+1] = RepeatedPairs[9*PairInts[iInt+1]+1];
            }
            StartRepeats+=nAdditionalIntervals[iPair];
            // Now we have a vector of the indices and a vector of the s1 locations
            // corresponding to those indices. Now we just need to sort
            
            // THIS SORT STEP IS NOT RIGHT AND IT'S VERY ANNOYING. 
            // There is some memory leak here
            //std::sort(PairInts.begin(), PairInts.end(), compare_indirect_index <decltype(Starts1s)> ( Starts1s ) );
            // Push back the first interval
            vec JoinedIntervals(9);
            for (int iE=0; iE < 9; iE++){
                JoinedIntervals[iE]=RepeatedPairs[PairInts[0]*9+iE];
            }
            //std::cout << JoinedIntervals[1] << std::endl;
            int nJoined=1;
            for (int iInt = 0; iInt < nAdditionalIntervals[iPair]; iInt++){
                int indToCheck = PairInts[iInt+1];
                double s1start = RepeatedPairs[9*indToCheck+1];
                double s1end = RepeatedPairs[9*indToCheck+2];
                double s2start = RepeatedPairs[9*indToCheck+4];
                double s2end = RepeatedPairs[9*indToCheck+5];
                bool DisjointFromAll = true;
                for (int jInt = 0; jInt < nJoined; jInt++){
                    double Rs1start = JoinedIntervals[9*jInt+1];
                    double Rs1end = JoinedIntervals[9*jInt+2];
                    double Rs2start = JoinedIntervals[9*jInt+4];
                    double Rs2end = JoinedIntervals[9*jInt+5];
                    // Determine if intervals are disjoint
                    // We are checking overlap between 
                    // [s1start,s1end] x [Rs1start, Rs1end] AND 
                    // [s2start, s2end] x [Rs2start, Rs2end]
                    if ((Rs1start > s1end || Rs1end < s1start) && 
                        (Rs2start > s2end || Rs2end < s2start)) { 
                    } else {
                        // This merges the two rectangles into one large
                        // rectangle. Because the intervals are always processed
                        // in increasing order of s1, there will be no double
                        // counting
                        DisjointFromAll= false;
                        JoinedIntervals[9*jInt+1] = std::min(s1start,Rs1start);
                        JoinedIntervals[9*jInt+2] = std::max(s1end,Rs1end);
                        JoinedIntervals[9*jInt+4] = std::min(s2start,Rs2start);
                        JoinedIntervals[9*jInt+5] = std::max(s2end,Rs2end);
                    }
                }
                if (DisjointFromAll) { 
                    for (int iE=0; iE < 9; iE++){
                        JoinedIntervals.push_back(RepeatedPairs[indToCheck*9+iE]);
                    }
                    nJoined++;
                }
            } // end loop over intervals for a pair
            NewPairs.insert(NewPairs.end(),JoinedIntervals.begin(),JoinedIntervals.end());
        } // end loop over pairs
        return makePyByNineArray(NewPairs);
    }
    
    private:

    // Variables for uniform points
    int _nXPerFib, _nUni, _NFib, _nThreads;
    double _F0, _delta, _dcut, _deltasu;
    vec _RUniform, _su;
    DomainC _Dom;
    // Variables for segments
    int _NsegPerFib;            
    double _SegmentCutoff, _L, _Lseg, _NewtonTol;
    vec _sC, _REndpoint, _RMidpoint, _ValsToCoeffs, _DiffMat;
    // Variables for Legendre quad
    int _NPtsPerStd;
    vec _LegPts, _LegWts;
    
    double dUdrGaussian(double nr){
        if (nr > _dcut){
            return 0;
        }
        return (-_F0*exp(-nr*nr/(2*_delta*_delta)));
    }
    
    double DistanceBetweenFiberParts(const vec &X1,const vec &X2, vec &ClosePts){
        /* Nonlinear Newton solve to obtain distance between two fibers
        */ 
        // Initial guess for Newton
        double s1star = ClosePts[0];
        double s2star = ClosePts[1];
        vec DX1(3*_nXPerFib), DX2(3*_nXPerFib), DSqX1(3*_nXPerFib), DSqX2(3*_nXPerFib);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,X1,DX1);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,X2,DX2);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX1,DSqX1);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX2,DSqX2);
        vec Gradient(2,100);
        int nIts = 0; 
        vec rts(2);
        rts[0] = s1star;
        rts[1] = s2star;
        vec Jac(4), DescentDirection(2);
        vec dist(3), X1p(3), X2p(3), DX1p(3), DX2p(3), DSqX1p(3), DSqX2p(3),InterpRow1(_nXPerFib), InterpRow2(_nXPerFib);
        double dist_before=_L;
        double distance = _L;
        while (sqrt(dot(Gradient,Gradient)) > _NewtonTol && nIts < MAX_NEWTON_ITS){
            nIts++;
            
            // Compute distance, gradient, and Hessian
            InterpolationRow(rts[0],InterpRow1);
            InterpolationRow(rts[1],InterpRow2);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DX1,DX1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DX2,DX2p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DSqX1,DSqX1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DSqX2,DSqX2p);
            
            for (int d=0; d < 3; d++){
                dist[d]=X1p[d]-X2p[d];
            }
            dist_before = sqrt(dot(dist,dist));
            Gradient[0] = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
            Gradient[1] = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
            Jac[0] = 2*(dot(DX1p,DX1p)+dot(DSqX1p,X1p)-dot(DSqX1p,X2p));
            Jac[2] = -2*dot(DX1p,DX2p);
            Jac[1] = Jac[2];
            Jac[3] = 2*(dot(DX2p,DX2p)+dot(DSqX2p,X2p)-dot(DSqX2p,X1p));
            
            // Eigenvalue decomposition of Jacobian
            vec EigVals(2), EigVecs(4);
            SymmetrizeAndDecomposePositive(2, Jac, 0, EigVecs, EigVals);
            double maxEigInd=0;
            double minEigInd=1;
            if (EigVals[1] > EigVals[0]){
                maxEigInd=1;
                minEigInd=0;
            }
            if (EigVals[maxEigInd] < 0){
                std::cout << "Breaking out of Newton - Hessian is negative definite!" << std::endl;
                break;
            } else if (EigVals[minEigInd] < EigVals[maxEigInd]/COND_MAX){
                EigVals[minEigInd] = EigVals[maxEigInd]/COND_MAX;
            }
            vec TwoByTwoIdentity(4), HessianInv(4); // initialize to identity
            TwoByTwoIdentity[0]=1;
            TwoByTwoIdentity[3]=1;
            ApplyMatrixPowerFromEigDecomp(2, -1.0, EigVecs, EigVals, TwoByTwoIdentity, HessianInv);
            // Define boundary set and project off 
            for (int d=0; d < 2; d++){
                if ((rts[d] < DISTANCE_EPS && Gradient[d] > 0) || (rts[d] > _L-DISTANCE_EPS && Gradient[d] < 0)){
                    // Zero out the entry in the gradient and the row and column of the Hessian
                    for (int HEntry=0; HEntry < 3; HEntry++){
                        HessianInv[HEntry+d]=0;
                    }
                    Gradient[d]=0;
                }
            }
            BlasMatrixProduct(2,2,1,-1.0, 0.0,HessianInv, false, Gradient, DescentDirection);
            double normDD = sqrt(dot(DescentDirection,DescentDirection));
            if (normDD > MAX_DESCENT_NORM_OVERL*_L){
                for (int d=0; d < 2; d++){
                    DescentDirection[d]*=  MAX_DESCENT_NORM_OVERL*_L/normDD;
                }
            }     
            // Armijo back tracking line search
            bool armijo=false;
            double alpha=1;
            double GradDotDescent = dot(DescentDirection,Gradient);
            while (!armijo && alpha > 0){
                if (alpha < _NewtonTol){
                    alpha=0;
                }
                vec rtguess(2);
                for (int d=0; d < 2; d++){
                    rtguess[d]=rts[d]+alpha*DescentDirection[d];
                    // Projection
                    if (rtguess[d] < 0){
                        rtguess[d]=0;
                    } else if (rtguess[d] > _L){
                        rtguess[d]=_L;
                    }
                }
                // Evaluate function
                InterpolationRow(rtguess[0],InterpRow1);
                InterpolationRow(rtguess[1],InterpRow2);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
                for (int d=0; d < 3; d++){
                    dist[d]=X1p[d]-X2p[d];
                }
                distance = sqrt(dot(dist,dist));
                // The Armijo parameter is 1/2
                if ((dist_before - distance) >=-0.5*alpha*GradDotDescent){
                    armijo=true;
                    for (int d=0; d < 2; d++){
                        rts[d]=rtguess[d];
                    }
                } else {
                    alpha=alpha/2;
                }
            } // End Armijo
        } // End Newton
        ClosePts[0] = rts[0];
        ClosePts[1] = rts[1];
        return distance;
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
    
    bool compare(int a, int b, vec data){
        return data[a]>data[b];
    }
    
    npDoub makePyByNineArray(vec &cppvec){
        ssize_t              ndim    = 2;
        std::vector<ssize_t> shape   = { (long) cppvec.size()/9 , 9 };
        std::vector<ssize_t> strides = { sizeof(double)*9 , sizeof(double) };

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
        .def("initLegQuadVariables",&StericForceEvaluatorC::initLegQuadVariables)
        .def("getNewtonTol",&StericForceEvaluatorC::getNewtonTol)
        .def("ForcesFromGlobalUniformBlobs", &StericForceEvaluatorC::ForcesFromGlobalUniformBlobs)
        .def("IntervalsOfInteraction",&StericForceEvaluatorC::IntervalsOfInteraction)
        .def("MergeRepeatedIntervals",&StericForceEvaluatorC::MergeRepeatedIntervals)
        .def("ForceFromIntervals",&StericForceEvaluatorC::ForceFromIntervals);
}


