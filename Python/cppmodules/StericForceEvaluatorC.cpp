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

const double SMALL_NUM = 1e-10;
const int MAX_NEWTON_ITS = 10;



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
    
    void initLegQuadVariables(int nLeg, npDoub xLeg, npDoub wLeg){
        _nLeg = nLeg;
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
                double ForceMag = dUdrGaussian(nr);
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
                // Now segments are close enough: compute the forces 
                // TODO: Add safety factor here based on curvature of segments 
                // Identify the two fibers (for resampling near the closest point)
                vec XCheb1(3*_nXPerFib);
                vec XCheb2(3*_nXPerFib);
                for (int iPt=0; iPt < _nXPerFib; iPt++){
                    for (int iD=0; iD < 3; iD++){
                        XCheb1[3*iPt+iD] = ChebPts[iFib*3*_nXPerFib+3*iPt+iD];
                        XCheb2[3*iPt+iD] = ChebPts[jFib*3*_nXPerFib+3*iPt+iD]+PeriodicShift[iD];
                    }
                }
                // Sample at midpoint to estimate curvature
                if (segDist < _dcut){
                    std::cout << "Segment " << iSeg << " and " << jSeg << std::endl;
                    std::cout << segDist << " , " << ClosePts[0] << " , " << ClosePts[1] << std::endl;
                    // Resolve with nonlinear solve to obtain closest points on segments 
                    double curvDist = DistanceBetweenFiberParts(XCheb1,XCheb2,iSegMod, jSegMod, ClosePts);
                    ClosePts[0]=(ClosePts[0]-iSegMod*_Lseg)/_Lseg;
                    ClosePts[1]=(ClosePts[1]-jSegMod*_Lseg)/_Lseg;
                    // Compute tangent vectors
                    vec DX1(3*_nXPerFib), DX2(3*_nXPerFib);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,XCheb1,DX1);
                    BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,XCheb2,DX2);
                    double s1star = ClosePts[0]*_Lseg+iSegMod*_Lseg;
                    double s2star = ClosePts[1]*_Lseg+jSegMod*_Lseg;
                    std::cout << ClosePts[0] << " , " << ClosePts[1] << std::endl;
                    vec InterpRow1(_nXPerFib), InterpRow2(_nXPerFib);
                    InterpolationRow(s1star,InterpRow1);
                    InterpolationRow(s2star,InterpRow2);
                    vec tau1(3), tau2(3);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DX1,tau1);
                    BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DX2,tau2);
                    double Tau1DotTau2 = dot(tau1,tau2);
                    int dir = 1;
                    if (Tau1DotTau2 < 0){
                        dir = -1;
                    }
                    // Walk along segments to identify box of close points (approximation of elliptical region)
                    vec sSeg1Bds(2,ClosePts[0]), sSeg2Bds(2,ClosePts[1]);
                    for (int Fib1EP=0; Fib1EP < 2; Fib1EP++){
                        int Fib2EP = Fib1EP + ((dir-1)/(-2) % 2);
                        double bddist = segDist;
                        double segincr = _delta/_Lseg; // Stepping one standard deviation at a time
                        bool EP1=false;
                        bool EP2=false;
                        if (abs(ClosePts[0]-Fib1EP) < SMALL_NUM){
                            EP1 = true;
                        } if (abs(ClosePts[1]-Fib2EP) < SMALL_NUM){
                            EP2 = true;    
                        }
                        while (bddist < _dcut && !(EP1 && EP2)){
                            if (!EP1){
                                sSeg1Bds[Fib1EP]+=segincr*(2*Fib1EP-1);
                                if (sSeg1Bds[Fib1EP] > 1){
                                    sSeg1Bds[Fib1EP]=1;
                                    EP1=true;
                                } else if (sSeg1Bds[Fib1EP] < 0){
                                    sSeg1Bds[Fib1EP]=0;
                                    EP1=true;
                                }
                            } if (!EP2){
                                sSeg2Bds[Fib1EP]+=dir*segincr*(2*Fib1EP-1);
                                if (sSeg2Bds[Fib1EP] > 1){
                                    sSeg2Bds[Fib1EP]=1;
                                    EP2=true;
                                } else if (sSeg2Bds[Fib1EP] < 0){
                                    sSeg2Bds[Fib1EP]=0;
                                    EP2=true;
                                }
                            } 
                            double s1Now = sSeg1Bds[Fib1EP]*_Lseg+iSegMod*_Lseg;
                            double s2Now = sSeg2Bds[Fib1EP]*_Lseg+jSegMod*_Lseg;
                            InterpolationRow(s1Now,InterpRow1);
                            InterpolationRow(s2Now,InterpRow2);
                            vec X1qp(3), X2qp(3);
                            BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow1,false,XCheb1,X1qp); 
                            BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow2,false,XCheb2,X2qp); 
                            vec3 rvec;
                            for (int d=0; d < 3; d++){
                                rvec[d] = X1qp[d]-X2qp[d];
                            }
                            // Compute force and add to both
                            bddist = normalize(rvec);
                        }
                    }
                    vec SortSeg2Bds(2);
                    SortSeg2Bds[0]=std::min(sSeg2Bds[0],sSeg2Bds[1]);
                    SortSeg2Bds[1]=std::max(sSeg2Bds[0],sSeg2Bds[1]);
                    // Gauss-Legendre integration on segment intervals
                    vec Seg1Qpts(_nLeg), Seg2Qpts(_nLeg), Seg1Wts(_nLeg), Seg2Wts(_nLeg);
                    for (int q=0; q < _nLeg; q++){
                        Seg1Qpts[q]=(_LegPts[q]+1.0)*0.5*(sSeg1Bds[1]-sSeg1Bds[0])+sSeg1Bds[0];
                        Seg1Qpts[q] = Seg1Qpts[q]*_Lseg+iSegMod*_Lseg;
                        Seg1Wts[q]=_LegWts[q]*0.5*(sSeg1Bds[1]-sSeg1Bds[0])*_Lseg;
                        Seg2Qpts[q]=(_LegPts[q]+1.0)*0.5*(SortSeg2Bds[1]-SortSeg2Bds[0])+SortSeg2Bds[0];
                        Seg2Qpts[q] = Seg2Qpts[q]*_Lseg+jSegMod*_Lseg;
                        Seg2Wts[q]=_LegWts[q]*0.5*(SortSeg2Bds[1]-SortSeg2Bds[0])*_Lseg;
                    }
                    // Now do O(N^2) quadrature over the segment points
                    // For each segment point on the quadrature grid, I compute the 
                    // interpolation row (InterpRow), which is N x 1 and gives me the 
                    // row that samples X at the particular quadrature point
                    // then do the force calculation
                    for (int iS1Pt=0; iS1Pt < _nLeg; iS1Pt++){
                        double siqp = Seg1Qpts[iS1Pt];
                        vec InterpRow1(_nXPerFib);
                        InterpolationRow(siqp,InterpRow1);
                        vec X1qp(3);
                        BlasMatrixProduct(1, _nXPerFib, 3,1.0,0.0,InterpRow1,false,XCheb1,X1qp); 
                        for (int jS2Pt=0; jS2Pt < _nLeg; jS2Pt++){
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
                        }
                    }
                        
                }
                nGood++;
            }
        }
        //std::cout << "Number good pairs " << nGood << std::endl;
        return make1DPyArray(StericForces);
    }
    
    private:

    // Variables for uniform points
    int _nXPerFib, _nUni, _NFib, _nThreads;
    double _F0, _delta, _dcrit, _dcut, _deltasu;
    vec _RUniform, _su;
    DomainC _Dom;
    // Variables for segments
    int _NsegPerFib;            
    double _SegmentCutoff, _L, _Lseg;
    vec _sC, _REndpoint, _RMidpoint, _ValsToCoeffs, _DiffMat;
    // Variables for Legendre quad
    int _nLeg;
    vec _LegPts, _LegWts;
    
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
    
    double dUdrGaussian(double nr){
        if (nr > _dcut){
            return 0;
        }
        return (-_F0*exp(-nr*nr/(2*_delta*_delta)));
    }
    
    double DistanceBetweenFiberParts(const vec &X1,const vec &X2,int iSegMod, int jSegMod, vec &ClosePts){
        /* Nonlinear Newton solve to obtain distance between two fiber sections 
        Section iSegMod [iSegMod*_Lseg, (iSegMod+1)*_Lseg] on fiber 1 
        and Sec jSecMod [jSegMod*_Lseg, (jSegMod+1)*_Lseg] on fiber 2
        */ 
        // Initial guess for Newton
        double s1star = ClosePts[0]*_Lseg+iSegMod*_Lseg;
        double s2star = ClosePts[1]*_Lseg+jSegMod*_Lseg;
        vec DX1(3*_nXPerFib), DX2(3*_nXPerFib), DSqX1(3*_nXPerFib), DSqX2(3*_nXPerFib);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,X1,DX1);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,X2,DX2);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX1,DSqX1);
        BlasMatrixProduct(_nXPerFib,_nXPerFib,3,1.0,0.0,_DiffMat,false,DX2,DSqX2);
        // Step 1 : Newton solve to find close points on segment interior
        vec Fcn(2,100);
        int nIts = 0; 
        vec rts(2);
        rts[0] = 0;
        rts[1] = 0;
        vec Jac(4), Diff(2);
        vec X1p(3), X2p(3), DX1p(3), DX2p(3), DSqX1p(3), DSqX2p(3),InterpRow1(_nXPerFib), InterpRow2(_nXPerFib);
        while (sqrt(Fcn[0]*Fcn[0]+Fcn[1]*Fcn[1]) > SMALL_NUM && nIts < MAX_NEWTON_ITS){
            nIts++;
            InterpolationRow(rts[0],InterpRow1);
            InterpolationRow(rts[1],InterpRow2);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DX1,DX1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DX2,DX2p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DSqX1,DSqX1p);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DSqX2,DSqX2p);
            
            Fcn[0] = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
            Fcn[1] = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
            Jac[0] = 2*(dot(DX1p,DX1p)+dot(DSqX1p,X1p)-dot(DSqX1p,X2p));
            Jac[2] = -2*dot(DX1p,DX2p);
            Jac[1] = Jac[2];
            Jac[3] = 2*(dot(DX2p,DX2p)+dot(DSqX2p,X2p)-dot(DSqX2p,X1p));

            TwoByTwoMatrixSolve(Jac,Fcn,Diff);
            //std::cout << rts[0] << " , " << rts[1] << std::endl;
            //std::cout << Diff[0] << " , " << Diff[1] << std::endl;
            rts[0]-=Diff[0];
            rts[1]-=Diff[1];
            //std::cout << rts[0] << " , " << rts[1] << std::endl;
        }
        vec3 disp;
        for (int d=0; d<3; d++){
            disp[d]=X1p[d]-X2p[d];
        }
        double distance = normalize(disp);
        double seg1start = iSegMod*_Lseg;
        double seg1end = seg1start+_Lseg;
        double seg2start = jSegMod*_Lseg;
        double seg2end = seg2start+_Lseg;
        // If interior minimum is achieved, return
        if (rts[0] > seg1start && rts[0] < seg1end && rts[1] > seg2start && rts[1] < seg2end){
            ClosePts[0] = rts[0];
            ClosePts[1] = rts[1];
            return distance;
        }
        // Other possible minima
        // Fix the endpoint of segment 1, and check for an interior min on segment 2
        // and the endpoints of segment 2
        vec exteriorDists, exteriorS1, exteriorS2;
        double Fcn1D, DFcn;
        for (int iSeg1EP=0; iSeg1EP<2; iSeg1EP++){
            double s1boundary=seg1start+_Lseg*iSeg1EP;
            Fcn1D = 100;
            double s2rt = s2star;
            InterpolationRow(s1boundary,InterpRow1);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
            int nIts=0;
            while (abs(Fcn1D) > SMALL_NUM && nIts < MAX_NEWTON_ITS){
                InterpolationRow(s2rt,InterpRow2);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DX2,DX2p);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,DSqX2,DSqX2p);
                Fcn1D = 2*dot(X2p,DX2p)-2*dot(DX2p,X1p);
                DFcn = 2*(dot(DX2p,DX2p)+dot(DSqX2p,X2p)-dot(DSqX2p,X1p));
                s2rt = s2rt - Fcn1D/DFcn;
            }
            double distance = normalize(disp);
            if (s2rt > seg2start && s2rt < seg2end){
                // interior maximum
                for (int d=0; d<3; d++){
                    disp[d]=X1p[d]-X2p[d];
                }
                distance = normalize(disp);
                exteriorDists.push_back(distance);
                exteriorS1.push_back(s1boundary); 
                exteriorS2.push_back(s2rt);
            }
            // Endpoint check
            for (int iSeg2EP=0; iSeg2EP < 2; iSeg2EP++){
                double s2boundary=seg2start+_Lseg*iSeg2EP;  
                InterpolationRow(s2boundary,InterpRow2);  
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
                for (int d=0; d<3; d++){
                    disp[d]=X1p[d]-X2p[d];
                }
                distance = normalize(disp);
                exteriorDists.push_back(distance);
                exteriorS1.push_back(s1boundary); 
                exteriorS2.push_back(s2boundary);
            }
        } // end loop over s1 boundary pts
        // Fix the endpoint of segment 2, and check for an interior min on segment 1
        // and the endpoints of segment 2
        for (int iSeg2EP=0; iSeg2EP<2; iSeg2EP++){
            double s2boundary=seg2start+_Lseg*iSeg2EP;
            Fcn1D = 100;
            double s1rt = s1star;
            InterpolationRow(s2boundary,InterpRow2);
            BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow2,false,X2,X2p);
            int nIts=0;
            while (abs(Fcn1D) > SMALL_NUM && nIts < MAX_NEWTON_ITS){
                InterpolationRow(s1rt,InterpRow1);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DX1,DX1p);
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,DSqX1,DSqX1p);
                Fcn1D = 2*dot(X1p,DX1p)-2*dot(DX1p,X2p);
                DFcn = 2*(dot(DX1p,DX1p)+dot(DSqX1p,X1p)-dot(DSqX1p,X2p));
                s1rt = s1rt - Fcn1D/DFcn;
            }
            double distance = normalize(disp);
            if (s1rt > seg1start && s1rt < seg1end){
                // interior maximum
                for (int d=0; d<3; d++){
                    disp[d]=X1p[d]-X2p[d];
                }
                distance = normalize(disp);
                exteriorDists.push_back(distance);
                exteriorS1.push_back(s1rt); 
                exteriorS2.push_back(s2boundary);
            }
            for (int iSeg1EP=0; iSeg1EP < 2; iSeg1EP++){
                double s1boundary=seg1start+_Lseg*iSeg1EP;  
                InterpolationRow(s1boundary,InterpRow1);  
                BlasMatrixProduct(1,_nXPerFib,3,1.0,0.0,InterpRow1,false,X1,X1p);
                for (int d=0; d<3; d++){
                    disp[d]=X1p[d]-X2p[d];
                }
                distance = normalize(disp);
                exteriorDists.push_back(distance);
                exteriorS1.push_back(s1boundary); 
                exteriorS2.push_back(s2boundary);
            }
        } // end loop over s2 boundary pts
        // FIND MINIMUM DIST AND S1 S2 AND RETURN
        std::vector<double>::iterator it = std::min_element(std::begin(exteriorDists), std::end(exteriorDists));
        int MinIndex = std::distance(std::begin(exteriorDists), it);
        ClosePts[0] = exteriorS1[MinIndex];
        ClosePts[1] = exteriorS2[MinIndex];
        return exteriorDists[MinIndex];
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
        .def("initLegQuadVariables",&StericForceEvaluatorC::initLegQuadVariables)
        .def("ForcesFromGlobalUniformBlobs", &StericForceEvaluatorC::ForcesFromGlobalUniformBlobs)
        .def("ForceFromCloseSegments",&StericForceEvaluatorC::ForceFromCloseSegments);
}


