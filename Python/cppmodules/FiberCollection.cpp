#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include "SpecialQuadratures.cpp"
#include "utils.h"
#include "RPYKernels.cpp"

/**
    ManyFiberMethods.cpp
    C++ functions to compute the velocity due to many fibers. 
    This includes: the finite part integral, all integral kernel evaluations, 
    and the routine to correct the velocity from Ewald splitting. 
    4 main public methods
    1) CorrectNonLocalVelocity - correct the velocity from Ewald
    2) FinitePartVelocity - get the velocity due to the finite part integral
    3) SubtractAllRPY - subtract the free space RPY kernel (from the Ewald result)
    4) RodriguesRotations - return the rotated X and Xs
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class FiberCollection {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    FiberCollection(double muin,vec3 Lengths, double epsIn, double Lin, double aRPYFacIn, 
                       int NFibIn,int NChebIn, int NuniIn, double deltain, int nThreads){
        /**
        Initialize variables relating to each fiber
        @param muin = viscosity of fluid
        @param Lengths = 3 array of periodic lengths
        @param epsIn = fiber epsilon
        @param Lin = fiber length
        @param aRPYFacIn = hydrodynamic RPY radius is aRPYFacIn*eps*L
        @param NFibIn = number of fibers
        @param NChebIn = number of Chebyshev points on each fiber
        @param NuniIn = number of uniform points on each fiber 
        @param deltain = fraction of fiber with ellipsoidal tapering
        **/
        _aRPYFac = aRPYFacIn;
        initRPYVars(_aRPYFac*epsIn*Lin, muin, NFibIn*NChebIn, Lengths);
        // Set domain lengths
        initLengths(Lengths[0],Lengths[1],Lengths[2]);
        _epsilon = epsIn;
        _L = Lin;
        _NUniPerFib = NuniIn;
        _NFib = NFibIn;
        _delta = deltain;
        _nOMPThr = nThreads;
        
    }

    void initSpecQuadParams(double rcritIn, double dsCLin, double dInterpIn, 
                          double d2panIn, double upsampdIn, double specDIn){
        /**
        Initialize variables relating to special quadrature
        @param rcritIn = critical Bernstein radius for special quadrature
        @param dsCLin = non-dimensional distance d*=d/(_epsilon*L) at which velocity = CL velocity
        @param dInterpIn = non-dimension distance d*=d/(_epsilon*L) at which we STOP interpolating with CL velocity
        @param d2panIn = non-dimension distance d*=d/(_epsilon*L) below which we need 2 panels for special quad
        @param upsampdIn = critical distance d*=d/L below which we need upsampled quadrature
        @param specDIn = critical distance d*=d/L below which we need special quadrature
        **/
        setRhoCrit(rcritIn);
        _dstarCL = dsCLin;
        _dstarInterp = dInterpIn;
        _dstar2panels = d2panIn;
        _UpsampledQuadDist = upsampdIn;
        _SpecQuadDist = specDIn;
    }

    void initNodesandWeights(npDoub pyNormalNodes, npDoub pyNormalWts, npDoub pyUpNodes, npDoub pyUpWeights, npDoub pyVanderMat){
        /**
        Initialize nodes and weights
        @param pyNormalNodes = Chebyshev nodes on the fiber (1D numpy array)
        @param pyNormalWts = Clenshaw-Curtis weights for direct quadrature on the fiber (1D numpy array)
        @param pyUpNodes  = upsampled Chebyshev nodes on the fiber (1D numpy array)
        @param pyNormalWeights = upsampled Clenshaw-Curtis weights for upsampled direct quadrature on the fiber (1D numpy array)
        @param pyVanderMat = vandermonde matrix (2D numpy array)
        **/

        // allocate std::vector (to pass to the C++ function)
        _NChebPerFib = pyNormalNodes.size();
        _NormalChebNodes = vec(_NChebPerFib);
        _NormalChebWts = vec(_NChebPerFib);
        _NUpsample = pyUpNodes.size();
        _UpsampledNodes = vec(_NUpsample);
        _UpsampledChebWts = vec(_NUpsample);
        vec SpecialVandermonde(pyVanderMat.size());

        // copy py::array -> std::vector
        std::memcpy(_NormalChebNodes.data(),pyNormalNodes.data(),pyNormalNodes.size()*sizeof(double));
        std::memcpy(_NormalChebWts.data(),pyNormalWts.data(),pyNormalWts.size()*sizeof(double));
        std::memcpy(_UpsampledNodes.data(),pyUpNodes.data(),pyUpNodes.size()*sizeof(double));
        std::memcpy(_UpsampledChebWts.data(),pyUpWeights.data(),pyUpWeights.size()*sizeof(double));
        std::memcpy(SpecialVandermonde.data(),pyVanderMat.data(),pyVanderMat.size()*sizeof(double));

        setVandermonde(SpecialVandermonde, _NUpsample);
    }

    void initResamplingMatrices(npDoub pyRsUpsamp, npDoub pyRs2Panel, npDoub pyValtoCoef, npDoub pyCoeftoVal){
        /**
        Initialize resampling matrices 
        @param pyRsUpsample = upsampling matrix from the N point grid to Nupsample point grid (2D numpy aray)
        @param pyRs2Panel = upsampling matrix from the N point grid to 2 panels of Nupsample point grid (2D numpy aray)
        @param pyValtoCoef  = matrix that takes values to Chebyshev series coefficients on the N point Chebyshev grid (2D numpy array)
        @param pyCoeftoVal = matrix that takes coefficients of Chebyshev series to values on N point Cheb grid (2D numpy array)
        **/

        // allocate std::vector (to pass to the C++ function)
        _UpsamplingMatrix = vec(pyRsUpsamp.size());
        _TwoPanMatrix = vec(pyRs2Panel.size());
        _ValtoCoeffMatrix = vec(pyValtoCoef.size());
        _CoefftoValMatrix = vec(pyCoeftoVal.size());

        // copy py::array -> std::vector
        std::memcpy(_UpsamplingMatrix.data(),pyRsUpsamp.data(),pyRsUpsamp.size()*sizeof(double));
        std::memcpy(_TwoPanMatrix.data(),pyRs2Panel.data(),pyRs2Panel.size()*sizeof(double));
        std::memcpy(_ValtoCoeffMatrix.data(),pyValtoCoef.data(),pyValtoCoef.size()*sizeof(double));
        std::memcpy(_CoefftoValMatrix.data(),pyCoeftoVal.data(),pyCoeftoVal.size()*sizeof(double));
    }

    void init_FinitePartMatrix(npDoub pyFPMatrix, npDoub pyDiffMatrix){    
        /**
        Python wrapper to initialize variables for finite part integration in C++. 
        @param pyFPMatrix = matrix that maps g function coefficients to velocity values on the N point Cheb grid (2D numpy array)
        @param pyDiffMatrix = Chebyshev differentiation matrix on the N point Chebyshev grid (2D numpy aray)
        **/

        // allocate std::vector (to pass to the C++ function)
        _FinitePartMatrix = vec(pyFPMatrix.size());
        _DiffMatrix = vec(pyDiffMatrix.size());

        // copy py::array -> std::vector
        std::memcpy(_FinitePartMatrix.data(),pyFPMatrix.data(),pyFPMatrix.size()*sizeof(double));
        std::memcpy(_DiffMatrix.data(),pyDiffMatrix.data(),pyDiffMatrix.size()*sizeof(double));
    }
    
    void initMatricesForPreconditioner(npDoub pyUpsampMat, npDoub pyUpsampledchebPolys, npDoub pyLeastSquaresDownsampler, npDoub pyDpInv,
        npDoub pyweightedUpsampler, npDoub pyD4BC, npDoub &pyCs){
        
        _MatFromNto2N = vec(pyUpsampMat.size());
        _LeastSquaresDownsampler = vec(pyLeastSquaresDownsampler.size());
        _UpsampledChebPolys = vec(pyUpsampledchebPolys.size());
        _DpInv = vec(pyDpInv.size());
        _LocalDragCs = vec(pyCs.size());
        _weightedUpsampler = vec(pyweightedUpsampler.size());
        _D4BC = vec(pyD4BC.size());
        
        // copy py::array -> std::vector
        std::memcpy(_MatFromNto2N.data(),pyUpsampMat.data(),pyUpsampMat.size()*sizeof(double));
        std::memcpy(_LeastSquaresDownsampler.data(),pyLeastSquaresDownsampler.data(),pyLeastSquaresDownsampler.size()*sizeof(double));
        std::memcpy(_UpsampledChebPolys.data(),pyUpsampledchebPolys.data(),pyUpsampledchebPolys.size()*sizeof(double));
        std::memcpy(_DpInv.data(),pyDpInv.data(),pyDpInv.size()*sizeof(double));
        std::memcpy(_weightedUpsampler.data(),pyweightedUpsampler.data(),pyweightedUpsampler.size()*sizeof(double));
        std::memcpy(_D4BC.data(),pyD4BC.data(),pyD4BC.size()*sizeof(double));
        std::memcpy(_LocalDragCs.data(),pyCs.data(),pyCs.size()*sizeof(double));
    }

    //===========================================
    //      PUBLIC METHODS FOR COMPUTATION
    //===========================================
    npDoub RPYFiberKernel(npDoub &pyTargets, npDoub &pyFibPts, npDoub &pyForces){
        /**
        Compute the integral of the RPY kernel along a single fiber with respect to some targets
        @param Targets = the target positions (row stacked vector)
        @param FibPts = fiber points along which we are summing the kernel (row stacked vector)
        @param Forces = forces (NOT FORCE DENSITIES) at the fiber points (row stacked vector)
        **/
        // allocate std::vector (to pass to the C++ function)
        vec Targets(pyTargets.size());
        vec Points(pyFibPts.size());
        vec Forces(pyForces.size());

        // copy py::array -> std::vector
        std::memcpy(Targets.data(),pyTargets.data(),pyTargets.size()*sizeof(double));
        std::memcpy(Points.data(),pyFibPts.data(),pyFibPts.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));

        vec velocities(pyTargets.shape()[0]*3,0.0);
        int Ntarg = Targets.size()/3;
        vec3 uadd;
        for (int iTarg=0; iTarg < Ntarg; iTarg++){
            vec3 targPt = {Targets[3*iTarg],Targets[3*iTarg+1],Targets[3*iTarg+2]};
            OneRPYKernelWithForce(targPt,Points,Forces,uadd);
            for (int d=0; d<3; d++){
                velocities[iTarg*3+d]+=uadd[d];
            }
        }
        return makePyDoubleArray(velocities);
    }
    
    npDoub SubtractAllRPY(npDoub &pyChebFiberPoints, npDoub &pyFibForceDensities, npDoub &pyWeights){
        /**
        Method to subtract the RPY kernel at all targets due to the fiber that target belongs to. 
        First step in correcting the Ewald velocities. 
        @param ChebFiberPoints = points on all fibers (2D numpy array)
        @param FibForceDensities = fiber force densities on all fibers (2D numpy array)
        @param pyWeights = weights for the direct quadrature to subtract (1D numpy array)
        **/
        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebFiberPoints.size());
        vec ForceDensities(pyFibForceDensities.size());
        vec DirUpsampledWeights(pyWeights.size());

        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebFiberPoints.data(),pyChebFiberPoints.size()*sizeof(double));
        std::memcpy(ForceDensities.data(),pyFibForceDensities.data(),pyFibForceDensities.size()*sizeof(double));
        std::memcpy(DirUpsampledWeights.data(),pyWeights.data(),pyWeights.size()*sizeof(double));

        // call pure C++ function
        vec correctionUs(pyChebFiberPoints.shape()[0]*3,0.0);
        int Ndirect = DirUpsampledWeights.size();
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iPt=0; iPt < Ndirect*_NFib; iPt++){
            int fibNum = iPt/Ndirect;
            vec3 uRPYSelf = {0,0,0};
            vec3 target = {ChebPoints[3*iPt], ChebPoints[3*iPt+1], ChebPoints[3*iPt+2]};
            OneRPYKernel(target, ChebPoints, ForceDensities, DirUpsampledWeights,Ndirect*fibNum, Ndirect*(fibNum+1), uRPYSelf);
            for (int d =0; d< 3; d++){
                correctionUs[3*iPt+d]+= uRPYSelf[d];
            }
        }
        return makePyDoubleArray(correctionUs);
    }
    
    npDoub FinitePartVelocity(npDoub pyChebPoints, npDoub pyForceDens, npDoub pyXs){
        /**
        Compute the finite part velocity on all fibers 
        @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
        @param pyForceDens = Npts * 3 numpy array (2D) of force densities on ALL fibers
        @param pyXs = Npts * 3 numpy array (2D) of tangent vectors on ALL fibers
        @return velocities = Npts * 3 numpy array (1D, row stacked "C" order) of finite part velocities
        **/

        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec FDens(pyForceDens.size());
        vec Xs(pyXs.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(FDens.data(),pyForceDens.data(),pyForceDens.size()*sizeof(double));
        std::memcpy(Xs.data(),pyXs.data(),pyXs.size()*sizeof(double));

        // call pure C++ function
        vec uFP(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_NChebPerFib); // determine how many fiber we are working with
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = iFib*_NChebPerFib;
            vec Xss(3*_NChebPerFib,0.0);
            vec fprime(3*_NChebPerFib,0.0);
            // Compute Xss, fprime
            MatVec(_NChebPerFib, _NChebPerFib, 3, _DiffMatrix, Xs, start, Xss);
            MatVec(_NChebPerFib, _NChebPerFib, 3, _DiffMatrix, FDens, start, fprime);
            for (int iPt=0; iPt < _NChebPerFib; iPt++){
                int iPtInd = start+iPt;
                double nr, rdotf, oneoverr, ds, oneoverds;
                vec3 rvec;
                double Xsdotf = Xs[3*iPtInd]*FDens[3*iPtInd]+Xs[3*iPtInd+1]*FDens[3*iPtInd+1]+Xs[3*iPtInd+2]*FDens[3*iPtInd+2];
                double Xssdotf = Xss[3*iPt]*FDens[3*iPtInd]+Xss[3*iPt+1]*FDens[3*iPtInd+1]+Xss[3*iPt+2]*FDens[3*iPtInd+2];
                double Xsdotfprime = Xs[3*iPtInd]*fprime[3*iPt]+Xs[3*iPtInd+1]*fprime[3*iPt+1]+Xs[3*iPtInd+2]*fprime[3*iPt+2];
                for (int jPt=0; jPt < _NChebPerFib; jPt++){
                    int jPtInd = start+jPt;
                    for (int d=0; d < 3; d++){
                        rvec[d] = ChebPoints[3*iPtInd+d]-ChebPoints[3*jPtInd+d];
                    }
                    nr = sqrt(dot(rvec,rvec));
                    oneoverr = 1.0/nr;
                    ds = _NormalChebNodes[jPt]-_NormalChebNodes[iPt];
                    oneoverds = 1.0/ds;
                    rdotf = rvec[0]*FDens[3*jPtInd]+rvec[1]*FDens[3*jPtInd+1]+rvec[2]*FDens[3*jPtInd+2];
                    for (int d = 0; d < 3; d++){
                        // Compute the density g from Tornberg's paper and multiply by _FinitePartMatrix to get 
                        // the velocity due to the FP integral.
                        if (iPt==jPt){
                            uFP[3*iPtInd+d] += (0.5*(Xs[3*iPtInd+d]*Xssdotf+Xss[3*iPt+d]*Xsdotf)+
                                fprime[3*iPt+d]+Xs[3*iPtInd+d]*Xsdotfprime) *_FinitePartMatrix[_NChebPerFib*iPt+iPt];
                        } else{
                            uFP[3*iPtInd+d] += ((FDens[3*jPtInd+d] + rvec[d]*rdotf*oneoverr*oneoverr)*oneoverr*std::abs(ds)-\
                                (FDens[3*iPtInd+d]+Xs[3*iPtInd+d]*Xsdotf))*oneoverds*_FinitePartMatrix[_NChebPerFib*iPt+jPt]; 
                        } // end if iPt==jPt
                    } // end d loop
                } // end jPt loop
            } // end iPt loop
        } // end fiber loop
        return make1DPyArray(uFP);
    } // end calcFP velocity

    npDoub RodriguesRotations(npDoub pyXn, npDoub pyXsn, npDoub pyXsStar, npDoub pyAllKAlphas,double dt){
        /*
        Method to update the tangent vectors and positions using Rodriguez rotation and Chebyshev integration
        for many fibers in parallel. 
        @param pyXn = 2D numpy array of current Chebyshev points (time n) on the fiber, 
        @param pyXsn = 2D numpy array of current tangent vectors (time n)
        @param pyXsStar = 2D numpy array of tangent vectors for non-local (Omega) calculations
        @param pyAllKAlphas = fiber velocities using dX/dt = K*alpha 
        @param dt = timestep
        @return TransformedXsAndX = all of the new tangent vectors and positions (2D numpy array)
        */
  
        // allocate std::vector (to pass to the C++ function)
        vec Xn(pyXn.size());
        vec Xsn(pyXsn.size());
        vec XsStar(pyXsStar.size());
        vec AllVels(pyAllKAlphas.size());

        // copy py::array -> std::vector
        std::memcpy(Xn.data(),pyXn.data(),pyXn.size()*sizeof(double));
        std::memcpy(Xsn.data(),pyXsn.data(),pyXsn.size()*sizeof(double));
        std::memcpy(XsStar.data(),pyXsStar.data(),pyXsStar.size()*sizeof(double));
        std::memcpy(AllVels.data(),pyAllKAlphas.data(),pyAllKAlphas.size()*sizeof(double));

        // call pure C++ function
        vec TransformedXsAndX(2*pyXsn.shape()[0]*3,0.0);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < _NFib; iFib++){
            vec DKalpha(3*_NChebPerFib,0);
            int start = iFib*_NChebPerFib;
            // Compute D*(KAlpha)
            MatVec(_NChebPerFib, _NChebPerFib, 3, _DiffMatrix, AllVels, start, DKalpha);

            // Rotate tangent vectors
            for (int iPt=0; iPt < _NChebPerFib; iPt++){
                int ptindex = iPt+start;
                vec3 ThisXsStar, XsToRotate, DKAlphaI;
                for (int d =0; d < 3; d++){
                    ThisXsStar[d] = XsStar[3*ptindex+d];
                    XsToRotate[d] = Xsn[3*ptindex+d];
                    DKAlphaI[d] = DKalpha[3*iPt+d];
                }
                vec3 Omega;
                cross(ThisXsStar,DKAlphaI,Omega); // writes result to Omega
                double nOm = normalize(Omega);
                if (nOm > 1e-6){
                    double theta = nOm*dt;    
                    double OmegaDotXs = dot(Omega,XsToRotate);
                    vec3 OmegaCrossXs;
                    cross(Omega,XsToRotate,OmegaCrossXs);
                    for (int d = 0; d < 3; d++){
                        TransformedXsAndX[3*ptindex+d] = XsToRotate[d]*cos(theta)+OmegaCrossXs[d]*sin(theta)+Omega[d]*OmegaDotXs*(1.0-cos(theta));
                    }
                } else{
                    for (int d = 0; d < 3; d++){
                        TransformedXsAndX[3*ptindex+d] = XsToRotate[d];
                    }
                }
            } // end rotate tangent vectors
            
            // Compute coefficients and integrate
            vec Xshat(3*_NChebPerFib,0);
            vec Xhat(3*_NChebPerFib,0), _deltaX(3*_NChebPerFib,0);
            MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, TransformedXsAndX, start, Xshat);
            IntegrateCoefficients(Xshat, 3, _L, Xhat);
            MatVec(_NChebPerFib, _NChebPerFib, 3, _CoefftoValMatrix, Xhat, 0, _deltaX);
            for (int iPt=0; iPt < _NChebPerFib; iPt++){
                int xptindex = (_NFib+iFib)*_NChebPerFib+iPt;
                for (int d =0; d < 3; d++){
                    TransformedXsAndX[3*xptindex+d]=_deltaX[3*iPt+d]-_deltaX[d]+Xn[3*start+d]+dt*AllVels[3*start+d];
                } 
            } // end update X
        } // end loop over fibers
        return makePyDoubleArray(TransformedXsAndX);
    } // end Rodrigues rotations
    
    
    npDoub CorrectNonLocalVelocity(npDoub pyChebPoints, npDoub pyUniformPoints, npDoub pyForceDens, 
        npDoub pyFinitePartVels, double g,  npInt pyNumbyFib, intvec &allTargetNums){
        /**
        Correct the velocity from Ewald via special quadrature (or upsampled quadrature)
        @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
        @param pyUniformPoints = Npts * 3 numpy array (2D) of uniform points on ALL fibers
        @param pyForceDens = Npts * 3 numpy array (2D) of force densities on ALL fibers
        @param pyFinitePartVels = Npts * 3 numpy array (2D) of velocities due to the finite part integral on 
            ALL fibers (necessary when the fibers are close togther and the centerline velocity is used)
        @param g = strain in coordinate system
        @param pyNumbyFib = Npts (1D) numpy array of Nfib length with the number of targets that require correction for each fiber
        @param TargetNums = python LIST (not array) of the target indices that need correction in sequential order
        @param nThreads = number of threads to use in parallel processing
        @return correctionUs = Npts * 3 numpy array (2D) of corrections to the Ewald velocity. 
        **/

        // allocate std::vector (to pass to the C++ function)
        vec ChebFiberPoints(pyChebPoints.size());
        vec UniformFiberPoints(pyUniformPoints.size());
        vec FibForceDensities(pyForceDens.size());
        vec FinitePartVelocities(pyFinitePartVels.size());
        intvec numTargsbyFib(pyNumbyFib.size());

        // copy py::array -> std::vector
        std::memcpy(ChebFiberPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(UniformFiberPoints.data(),pyUniformPoints.data(),pyUniformPoints.size()*sizeof(double));
        std::memcpy(FibForceDensities.data(),pyForceDens.data(),pyForceDens.size()*sizeof(double));
        std::memcpy(FinitePartVelocities.data(),pyFinitePartVels.data(),pyFinitePartVels.size()*sizeof(double));
        std::memcpy(numTargsbyFib.data(),pyNumbyFib.data(),pyNumbyFib.size()*sizeof(int));

        vec correctionUs(pyChebPoints.shape()[0]*3,0.0);
        
        // Cumulative sum of the number of targets by fiber to figure out where to start
        intvec endTargNum(numTargsbyFib.begin(),numTargsbyFib.end());
        std::partial_sum(endTargNum.begin(), endTargNum.end(),endTargNum.begin());
        #pragma omp parallel for num_threads(_nOMPThr) schedule(dynamic)
        for (int iFib=0; iFib < _NFib; iFib++){
            // Initialize fiber specific quantities
            vec UpsampledPos(3*_NUpsample,0.0), UpsampledForceDs(3*_NUpsample,0.0), PositionCoefficients(3*_NChebPerFib,0.0);
            vec DerivCoefficients(3*_NChebPerFib,0.0), FinitePartCoefficients(3*_NChebPerFib,0.0), forceDCoefficients(3*_NChebPerFib,0.0);
            initFullFiberForSpecial(ChebFiberPoints,FibForceDensities,FinitePartVelocities,iFib, UpsampledPos, 
                UpsampledForceDs, PositionCoefficients, DerivCoefficients,FinitePartCoefficients, forceDCoefficients);
            // 2 Panels
            vec Pan1Pts(3*_NUpsample,0.0), Pan2Pts(3*_NUpsample,0.0), Pan1FDens(3*_NUpsample,0.0), Pan2FDens(3*_NUpsample,0.0);
            vec Pan1Coeffs(3*_NChebPerFib,0.0), Pan2Coeffs(3*_NChebPerFib,0.0), Pan1DCoeffs(3*_NChebPerFib,0.0), Pan2DCoeffs(3*_NChebPerFib,0.0);
            init2PanelsForSpecial(ChebFiberPoints, FibForceDensities, iFib, Pan1Pts, Pan2Pts, Pan1FDens, Pan2FDens, Pan1Coeffs, 
                Pan1DCoeffs, Pan2Coeffs, Pan2DCoeffs);
            // Loop over targets and do corrections
            for (int iT=endTargNum[iFib]-numTargsbyFib[iFib]; iT < endTargNum[iFib]; iT++){
                int ptNumber = allTargetNums[iT]; // target number (from 0 to _NFib*NptsbyFib
                vec3 uRPY={0,0,0}; // RPY subtraction from the velocity
                vec3 uSBT={0,0,0}; // SBT componenent
                vec3 CLpart={0,0,0}; // component coming from the centerline
                double CLwt = 0.0; // weight of centerline (for close points)
                double SBTwt = 1.0; // weight of SBT
                vec3 targetPoint;
                for (int d=0; d< 3; d++){
                    targetPoint[d]= ChebFiberPoints[3*ptNumber+d];
                }
                // First determine what quadrature is necessary
                int qtype = determineQuadratureMethod(UniformFiberPoints, iFib, g, targetPoint);
                if (qtype > 0){ // only do correction if necessary
                    // Subtract RPY kernel
                    OneRPYKernel(targetPoint,ChebFiberPoints,FibForceDensities,_NormalChebWts,iFib*_NChebPerFib,(iFib+1)*_NChebPerFib, uRPY);
                    if (qtype==1){
                        // Correct with upsampling (SBT kernel)
                        OneSBTKernel(targetPoint,UpsampledPos,UpsampledForceDs,_UpsampledChebWts, 0, _NUpsample, uSBT);
                    } else { // special quad
                        // Calculate root
                        complex troot;
                        int sqneeded = calculateRoot(_UpsampledNodes, UpsampledPos, PositionCoefficients,DerivCoefficients,targetPoint, troot);
                        // Estimate distance from fiber (tapprox = 1 if real(troot > 1), -1 if real(troot < -1.0))
                        double tapprox = std::max(std::min(real(troot),1.0),-1.0);
                        vec3 closestFibPoint;
                        double dstar = FindClosestFiberPoint(tapprox,PositionCoefficients,targetPoint,closestFibPoint);
                        if (dstar < _dstarInterp){
                             // Compute weight and value of CL velocity assigned to nearest CL velocity
                            CLwt = std::min((_dstarInterp-dstar)/(_dstarInterp-_dstarCL),1.0); // takes care of very close ones
                            calcCLVelocity(FinitePartCoefficients, DerivCoefficients, forceDCoefficients, tapprox,CLpart);
                        } 
                        SBTwt = 1.0-CLwt;
                        if (sqneeded==1){ // special quadrature
                            // Here is where we differentiate between one panel and two
                            vec w1(_NUpsample), w3(_NUpsample), w5(_NUpsample);
                            if (dstar > _dstar2panels){ // 1 panel of _NUpsample
                                specialWeights(_UpsampledNodes,troot,w1,w3,w5,_L);
                                SBTKernelSplit(targetPoint,UpsampledPos,UpsampledForceDs, w1, w3, w5, uSBT);
                            } else if (dstar > _dstarCL){ // 2 panels of _NUpsample
                                sqneeded = calculateRoot(_UpsampledNodes, Pan1Pts,Pan1Coeffs,Pan1DCoeffs,targetPoint, troot);
                                if (sqneeded){
                                    specialWeights(_UpsampledNodes,troot,w1,w3,w5,_L);
                                    SBTKernelSplit(targetPoint,Pan1Pts,Pan1FDens, w1, w3, w5, uSBT);
                                } else {
                                    OneSBTKernel(targetPoint,Pan1Pts,Pan1FDens,_UpsampledChebWts, 0, _NUpsample, uSBT);
                                }
                                sqneeded = calculateRoot(_UpsampledNodes, Pan2Pts,Pan2Coeffs,Pan2DCoeffs,targetPoint, troot);
                                if (sqneeded){
                                    specialWeights(_UpsampledNodes,troot,w1,w3,w5,_L);
                                    SBTKernelSplit(targetPoint,Pan2Pts,Pan2FDens, w1, w3, w5, uSBT);
                                } else {
                                    OneSBTKernel(targetPoint,Pan2Pts,Pan2FDens,_UpsampledChebWts, 0, _NUpsample, uSBT);
                                }
                                SBTwt*=0.5; // 2 panels, the weights are twice what they should be. Compensate here. 
                            }
                        } else{ // special quad not needed, do full fiber direct
                            OneSBTKernel(targetPoint,UpsampledPos,UpsampledForceDs,_UpsampledChebWts, 0, _NUpsample, uSBT);
                        } 
                    } // end special quad
                    // Add velocity to total velocity
                    for (int d=0; d < 3; d++){
                        #pragma omp atomic update
                        correctionUs[3*ptNumber+d]+= SBTwt*uSBT[d] + CLwt*CLpart[d] -uRPY[d];
                    }
                } // end if correction needed
            } // end loop over targets
        } // end parallel loop over fibers 
        return makePyDoubleArray(correctionUs);
    }
    
    py::array applyPreconditioner(npDoub pyTangents,npDoub pyb, double impcodt){

        // allocate std::vector (to pass to the C++ function)
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec b(pyb.size());
        std::memcpy(b.data(),pyb.data(),pyb.size()*sizeof(double));
                
        int N = _NChebPerFib;
        int nFib = _NFib;
        vec LambasAndAlphas(_NFib*(5*N+1));
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < nFib; iFib++){
	    //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
            vec LocalTangents(3*N);
            vec bFirst(3*N), MinvbFirst(3*N);
            vec bSecond(2*N+1);
            for (int i=0; i < 3*N; i++){
                LocalTangents[i] = Tangents[3*N*iFib+i];
                bFirst[i] = b[3*N*iFib+i];
                MinvbFirst[i] = b[3*N*iFib+i];
            }
            for (int j = 0; j < 2*N+1; j++){
                bSecond[j] = b[3*N*nFib+iFib*(2*N+1)+j];
            }
            
            vec J(6*N*(2*N-2));
            calculateJ(LocalTangents,J);
            vec K(3*N*2*(N-1));
            BlasMatrixProduct(3*N,6*N,2*(N-1),1.0,0.0,_LeastSquaresDownsampler,false,J,K);
            vec Kt(2*(N-1)*3*N);
            BlasMatrixProduct(2*(N-1),6*N,3*N,1.0,0.0,J,true,_weightedUpsampler,Kt); // J^T * weightedUpsampler
            vec KWithI(3*N*(2*N+1));
            vec KTWithI(3*N*(2*N+1));
            std::memcpy(KTWithI.data(),Kt.data(),Kt.size()*sizeof(double));
            for (int i=0; i < 3*N; i++){
                // Add to the columns of K
                for (int j=0; j < 2*(N-1); j++){
                    KWithI[i*(2*N+1)+j]=K[i*2*(N-1)+j];
                }
                KWithI[i*(2*N+1)+2*(N-1)+(i%3)] = 1;
                KTWithI[3*N*(2*(N-1)+(i%3))+i] = _NormalChebWts[i/3];
            }
            
            // Schur complement block B
            vec M(3*N*3*N);
            calcMLocal(LocalTangents,M);
            vec D4BCK(3*N*(2*N+1));
            BlasMatrixProduct(3*N,3*N,2*N+1,1.0,0.0,_D4BC,false,KWithI,D4BCK);
            vec B(3*N*(2*N+1));
            BlasMatrixProduct(3*N,3*N,2*N+1,-impcodt,1.0,M,false,D4BCK,KWithI);
            std::memcpy(B.data(),KWithI.data(),KWithI.size()*sizeof(double));
            
            // Factor M
            int sysDim = 3*N;
            int ipiv [sysDim];
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, sysDim, sysDim, &*M.begin(), sysDim, ipiv);
            // Solve M^-1*bFirst
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, 1, &*M.begin(), sysDim, ipiv, &*MinvbFirst.begin(), 1);
            // Overwrite b2 -> Schur RHS
            BlasMatrixProduct(2*N+1,3*N,1,1.0,1.0,KTWithI,false,MinvbFirst,bSecond);
            
            // Overwrite KWithI -> M^(-1)*B
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, 2*N+1, &*M.begin(), sysDim, ipiv, &*KWithI.begin(), 2*N+1);
            vec SchurComplement((2*N+1)*(2*N+1));
            BlasMatrixProduct(2*N+1, 3*N, 2*N+1,1.0,0.0,KTWithI,false,KWithI,SchurComplement); // Kt*(M^-1*K)
            int ipiv2 [2*N+1];
            LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2*N+1,1,&*SchurComplement.begin(), 2*N+1, ipiv2, &*bSecond.begin(), 1);
            // Back solve to get lambda
            vec Kalpha(3*N);
            BlasMatrixProduct(3*N,2*N+1,1,1.0,0.0,B,false, bSecond, Kalpha);
            for (int i=0; i < 3*N; i++){
                Kalpha[i]-=bFirst[i];
            }
            // Overwrite Kalpha -> lambda
            LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, 1, &*M.begin(), sysDim, ipiv, &*Kalpha.begin(), 1);
            for (int i=0; i < 3*N; i++){
                LambasAndAlphas[3*N*iFib+i] = Kalpha[i];
            }
            for (int j = 0; j < 2*N+1; j++){
                LambasAndAlphas[3*N*nFib+iFib*(2*N+1)+j] = bSecond[j];
            }
        }
        
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);
      
      
    } 
    
    py::array evalBendForces(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Forces(chebPts.size());
        int N = _NChebPerFib;
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*N);
            vec LocalForces(3*N);
            for (int i=0; i < 3*N; i++){
                FiberPoints[i] = chebPts[3*N*iFib+i];
            }
            BlasMatrixProduct(3*N,3*N,1,1.0,0.0,_D4BC,false,FiberPoints,LocalForces);
            for (int i=0; i < 3*N; i++){
                Forces[3*N*iFib+i] = LocalForces[i];
            }
        }
        return make1DPyArray(Forces);
    }
    
    py::array evalLocalVelocities(npDoub pyTangents, npDoub pyForceDensities){
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec ForceDensities(pyForceDensities.size());
        std::memcpy(ForceDensities.data(),pyForceDensities.data(),pyForceDensities.size()*sizeof(double));
        vec LocalVelocities(Tangents.size());
        int N = Tangents.size()/3;
        double viscInv = 1.0/(8.0*M_PI*mu);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iPt=0; iPt < N; iPt++){
            int ptPosition = iPt%_NChebPerFib; 
            double c = _LocalDragCs[ptPosition];
            vec M(9);
            vec f(3);
            vec u(3);
            for (int id = 0; id < 3; id++){
                f[id] = ForceDensities[3*iPt+id];
                for (int jd =0; jd < 3; jd++){
                    double deltaij = 0;
                    if (id==jd){
                        deltaij=1;
                    }
                    double XsXs_ij = Tangents[3*iPt+id]*Tangents[3*iPt+jd];
                    M[3*id+jd] = viscInv*(c*(deltaij+XsXs_ij)+(deltaij-3*XsXs_ij));
                }
            }
            BlasMatrixProduct(3,3,1,1.0,0.0,M,false,f,u);
            for (int id=0; id < 3; id++){
                LocalVelocities[3*iPt+id] = u[id];
            }
        }
        return make1DPyArray(LocalVelocities);
    } 
    
    private:
    
    double _epsilon, _L, _delta, _aRPYFac;
    int _NFib, _NChebPerFib, _NUniPerFib, _NUpsample, _nOMPThr;
    vec _NormalChebNodes, _UpsampledNodes, _NormalChebWts, _UpsampledChebWts;
    vec _LocalDragCs, _UpsampledChebPolys, _DpInv, _MatFromNto2N, _LeastSquaresDownsampler, _weightedUpsampler, _D4BC;
    double _dstarCL,_dstarInterp, _dstar2panels;
    double _UpsampledQuadDist, _SpecQuadDist;
    vec _FinitePartMatrix, _DiffMatrix;
    vec _UpsamplingMatrix, _TwoPanMatrix, _ValtoCoeffMatrix,_CoefftoValMatrix;
    
    void deAliasIntegral(const vec &f, vec &Integrals2N){
        /*
        */
        int N = _NChebPerFib;
        vec fTimesTUpsampled(2*N*(N-1));
        for (int iPt=0; iPt < 2*N; iPt++){
            for (int iPoly=0; iPoly < N-1; iPoly++){
                fTimesTUpsampled[iPt*(N-1)+iPoly]=f[iPt]*_UpsampledChebPolys[iPt*(N-1)+iPoly];
            }
        }
        BlasMatrixProduct(2*N, 2*N, N-1, 1.0, 0.0, _DpInv, false, fTimesTUpsampled, Integrals2N);
    }

    void calculateJ(const vec &Tangents,vec &J){
        int N = _NChebPerFib;
        vec XsUpsampled(6*N);
        BlasMatrixProduct(2*N,N,3,1.0,0.0,_MatFromNto2N,false,Tangents,XsUpsampled);
        vec n1x(2*N), n1y(2*N), n2x(2*N), n2y(2*N), n2z(2*N); // n1z(N) is zero
        for (int iPt = 0; iPt < 2*N; iPt++){
            double x = XsUpsampled[3*iPt];
            double y = XsUpsampled[3*iPt+1];
            double z = XsUpsampled[3*iPt+2];
            double theta = atan2(y,x);
            double phi = atan2(z,sqrt(x*x+y*y));
            if (abs(abs(phi)-M_PI*0.5) < 1e-12){
                theta = 0;
            } 
            n1x[iPt] = -sin(theta);
            n1y[iPt] = cos(theta);
            n2x[iPt] = -cos(theta)*sin(phi);
            n2y[iPt] = -sin(theta)*sin(phi);
            n2z[iPt] = cos(phi);
        }
        // Compute matrix J
        vec n1xTk(2*N*(N-1)), n1yTk(2*N*(N-1)), n2xTk(2*N*(N-1)), n2yTk(2*N*(N-1)), n2zTk(2*N*(N-1));
        deAliasIntegral(n1x, n1xTk); // 2N x N-1 matrix
        deAliasIntegral(n1y, n1yTk); 
        deAliasIntegral(n2x, n2xTk);
        deAliasIntegral(n2y, n2yTk);
        deAliasIntegral(n2z, n2zTk);
        int nCols = 2*N-2;
        for (int iPt=0; iPt < 2*N; iPt++){
            for (int iPoly=0; iPoly < N-1; iPoly++){
                J[(3*iPt)*nCols+iPoly] = n1xTk[iPt*(N-1)+iPoly];
                J[(3*iPt+1)*nCols+iPoly] = n1yTk[iPt*(N-1)+iPoly]; 
                J[(3*iPt)*nCols+iPoly+(N-1)] = n2xTk[iPt*(N-1)+iPoly];
                J[(3*iPt+1)*nCols+iPoly+(N-1)]= n2yTk[iPt*(N-1)+iPoly]; 
                J[(3*iPt+2)*nCols+iPoly+(N-1)] = n2zTk[iPt*(N-1)+iPoly];  
            }
        }
    }

    void calcMLocal(const vec &Tangents,vec &M){
        /*
        Calculates the local drag matrix M. 
        Inputs: Xs = fiber tangent vectors as a 3N array, c = N vector of local drag coefficients, 
        mu = viscosity, N = number of points 
        */
        double viscInv = 1.0/(8.0*M_PI*mu);
        for (int iPt = 0; iPt < _NChebPerFib; iPt++){
            for (int id = 0; id < 3; id++){
                for (int jd =0; jd < 3; jd++){
                    double deltaij = 0;
                    if (id==jd){
                        deltaij=1;
                    }
                    double XsXs_ij = Tangents[3*iPt+id]*Tangents[3*iPt+jd];
                    M[3*_NChebPerFib*(3*iPt+id)+3*iPt+jd] = viscInv*(_LocalDragCs[iPt]*(deltaij+XsXs_ij)+(deltaij-3*XsXs_ij));
                }
            }
        }
    }     
    
    void OneRPYKernelWithForce(const vec3 &targ, const vec &sourcePts, const vec &Forces, vec3 &utarg){
        /**
        Compute the RPY kernel at a specific target due to a fiber. 
        @param targ = the target position (3 array)
        @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
        @param Forces = forces (NOT FORCE DENSITIES) at the fiber points (row stacked vector)
        @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
        **/
        vec3 rvec;
        vec3 force;
        utarg = {0,0,0};
        double r, rdotf, F, G, co2;
        double outFront = 1.0/mu;
        int Nsrc = sourcePts.size()/3;
        for (int iSrc=0; iSrc < Nsrc; iSrc++){
            for (int d=0; d < 3; d++){
                rvec[d]=targ[d]-sourcePts[3*iSrc+d];
                force[d] = Forces[3*iSrc+d];
            }
            r = normalize(rvec);
            rdotf = dot(rvec,force);
            F = FtotRPY(r);
            G = GtotRPY(r);
            co2 = rdotf*G-rdotf*F;
            for (int d=0; d<3; d++){
                utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
            }
        }
    }
    
    void OneRPYKernel(const vec3 &targ, const vec &sourcePts, const vec &ForceDs, const vec &wts, 
                      int first, int last, vec3 &utarg){
        /**
        Compute the RPY kernel at a specific target due to a fiber. This method is more flexible than 
        OneRPYKernelWithForce since it uses force densities and weights to get force.
        @param targ = the target position (3 array)
        @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
        @param ForceDs = forces DENSITIES at the fiber points (row stacked vector)
        @param wts = quadrature weights for the integration
        @param first = where to start kernel the kernel ("row" index in SourcePts)
        @param last = index of sourcePts where we stop adding the kernel ("row" index in sourcePts)
        @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
        **/
        vec3 rvec;
        vec3 force;
        utarg = {0,0,0};
        double r, rdotf, F, G, co2;
        double outFront = 1.0/mu;
        for (int iSrc=first; iSrc < last; iSrc++){
            for (int d=0; d < 3; d++){
                rvec[d]=targ[d]-sourcePts[3*iSrc+d];
                force[d] = ForceDs[3*iSrc+d]*wts[iSrc-first];
            }
            r = normalize(rvec);
            rdotf = dot(rvec,force);
            F = FtotRPY(r);
            G = GtotRPY(r);
            co2 = rdotf*G-rdotf*F;
            for (int d=0; d<3; d++){
                utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
            }
        }
    }
    
        // Method to compute the SBT kernel for a single fiber and target. 
    void OneSBTKernel(const vec3 &targ, const vec &sourcePts, const vec &ForceDs, const vec &wts, 
                      int first, int last, vec3 &utarg){
        /**
        Compute the SBT kernel at a specific target due to a fiber. 
        @param targ = the target position (3 array)
        @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
        @param ForceDs = forces DENSITIES at the fiber points (row stacked vector)
        @param wts = quadrature weights for the integration
        @param first = where to start kernel the kernel (row in SourcePts)
        @param last = index of sourcePts where we stop adding the kernel (row in sourcePts)
        @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
        **/
        vec3 rvec;
        vec3 force;
        double a = _aRPYFac*_epsilon*_L;
        double r, rdotf, F, G, co2;
        double outFront = 1.0/mu;
        for (int iSrc=first; iSrc < last; iSrc++){
            for (int d=0; d < 3; d++){
                rvec[d]=targ[d]-sourcePts[3*iSrc+d];
                force[d] = ForceDs[3*iSrc+d]*wts[iSrc-first];
            }
            r = normalize(rvec);
            rdotf = dot(rvec,force);
            F = (2.0*a*a+3.0*r*r)/(24*M_PI*r*r*r);
            G = (-2.0*a*a+3.0*r*r)/(12*M_PI*r*r*r);
            co2 = rdotf*G-rdotf*F;
            for (int d=0; d<3; d++){
                utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
            }
        }
    }

    void SBTKernelSplit(const vec3 &targpt, const vec &FibPts, const vec &ForceDens, 
                        const vec &w1, const vec &w3, const vec &w5, vec3 &utarg){
        /**
        Method for SBT quadrature when the weights are different for different powers of r in the denominator.
        (as they are in special quadrature)
        @param targpt = the target position (3 array)
        @param FibPts = fiber points along which we are summing the kernel (row stacked vector)
        @param ForceDens = forces DENSITIES at the fiber points (row stacked vector)
        @param w1 = quadrature weights for 1/R power 
        @param w3 = quadrature weights for 1/R^3 power
        @param w5 = quadrature weights for 1/R^5 power
        @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
        **/
        vec3 rvec;
        vec3 forceDensity;
        double r, rdotf, u1, u3, u5;
        double outFront = 1.0/(8.0*M_PI*mu);
        int N = FibPts.size()/3;
        double dco = _aRPYFac*_aRPYFac*2.0/3.0;
        for (int iPt=0; iPt < N; iPt++){
            for (int d=0; d < 3; d++){
                rvec[d] = targpt[d]-FibPts[3*iPt+d];
                forceDensity[d] = ForceDens[3*iPt+d];
            }
            rdotf = dot(rvec,forceDensity);
            r = normalize(rvec);
            for (int d=0; d < 3; d++){
                u1 = forceDensity[d]/r;
                u3 = (r*rvec[d]*rdotf+dco*(_epsilon*_L)*(_epsilon*_L)*forceDensity[d])/(r*r*r);
                u5 = -3.0*dco*(_epsilon*_L)*(_epsilon*_L)*rvec[d]*r*rdotf/pow(r,5);
                utarg[d]+=outFront*(u1*w1[iPt]+u3*w3[iPt]+u5*w5[iPt]);
            }
        }
    }

    int determineQuadratureMethod(const vec &UniformFiberPoints, int iFib, double g, vec3 &targetPoint){
        /**
        Method to find the necessary quadrature type for a given target and fiber. 
        (as they are in special quadrature)
        @param UniformFiberPoints = row stacked vector of ALL uniform points on fibers
        @param iFib = fiber number 
        @param g = strain in the coordinate system 
        @param targetPoint = the target position (3 array) 
        @return 0 if _NChebPerFib quadrature is acceptable (that's what comes out of Ewald), 1
        if upsampled quadrature is needed, 2 if special quadrature is needed. The target point is 
        also modified with a periodic shift. 
        **/
        int qtype=0;
        vec3 rvec;
        vec3 rvecmin={0,0,0};
        double rmin = _UpsampledQuadDist*_L; // cutoff for upsampled quad
        int iPtMin=0;
        for (int iPt=0; iPt < _NUniPerFib; iPt++){
            for (int d=0; d < 3; d++){
                rvec[d] = targetPoint[d]-UniformFiberPoints[3*(iFib*_NUniPerFib+iPt)+d];
            }
            calcShifted(rvec,g); // periodic shift wrt strain 
            double nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
            if (nr < _SpecQuadDist*_L){
                for (int d=0; d < 3; d++){
                    targetPoint[d] = rvec[d] + UniformFiberPoints[3*(iFib*_NUniPerFib+iPt)+d];
                }
                return 2;
            } else if (nr < rmin){
                rmin=nr;
                qtype=1;
                rvecmin = {rvec[0],rvec[1],rvec[2]};
                iPtMin = iPt;
            }
        }
        for (int d=0; d < 3; d++){ // periodic shift in the target point
            targetPoint[d] = rvecmin[d] + UniformFiberPoints[3*(iFib*_NUniPerFib+iPtMin)+d];
        }
        return qtype;
    }

    double FindClosestFiberPoint(double tapprox, const vec &PositionCoefficients, const vec3 &target, vec3 &ClosestPoint){
        /**
        Find the closest point on a fiber to a target. 
        @param tapprox = closest centerline coordinate (on [-1,1] to the target)
        @param PositionCoefficients = Chebyshev coefficients of the centerline position (row stacked vector)
        @param target = the targetPoint (a 3 array)
        @param ClosestPoint = the closest point on the fiber (modified here)
        @return the non-dimensional closest distance d*=distance from fiber /(_epsilon*L). 
        **/
        // Compute closest fiber point
        ClosestPoint = {0,0,0};
        eval_Cheb3Dirs(PositionCoefficients, tapprox, ClosestPoint);
        vec3 dfromFib;
        for (int d=0; d< 3; d++){
            dfromFib[d] = ClosestPoint[d]-target[d];
        }
        double dstar = sqrt(dot(dfromFib,dfromFib)); // distance
        return dstar;
    }
      
    void initFullFiberForSpecial(const vec &ChebFiberPoints, const vec &FibForceDensities,const vec &CenterlineVelocities, int iFib,
        vec &UpsampledPos, vec &UpsampledForceDs, vec &PositionCoefficients, vec &DerivCoefficients, vec &CenterlineVelCoefficients, 
        vec &forceDCoefficients){
        /**
        Initialize the vectors for a particular fiber
        @param ChebFiberPoints = points on all fibers 
        @param FibForceDensities = fiber force densities on all fibers
        @param CenterlineVelocities = finite part velocities of the centerline on all fibers
        @param iFib = fiber number
        @return This is a void method, but it modifies the following arrays: the upsampled positions, upsampled force densities,
        and the coefficients of the fiber position, tangent vectors, finite part (centerline) velocities, and force densities. 
        **/
        int start = _NChebPerFib*iFib;
        // Upsample points and forces to _NUpsample point grid
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, ChebFiberPoints, start, UpsampledPos);
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, FibForceDensities, start, UpsampledForceDs);
        // Compute coefficients for points, tangent vectors, and force densities
        MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, ChebFiberPoints, start, PositionCoefficients);
        DifferentiateCoefficients(PositionCoefficients,3,DerivCoefficients);
        MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, FibForceDensities, start, forceDCoefficients);
        // Centerline velocity coefficients
        MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, CenterlineVelocities, start, CenterlineVelCoefficients);
    }

    void init2PanelsForSpecial(const vec &ChebFiberPoints, const vec &FibForceDensities, int iFib, vec &Pan1Pts, vec &Pan2Pts, 
               vec &Pan1FDens, vec &Pan2FDens, vec &Pan1Coeffs, vec &Pan1DCoeffs, vec &Pan2Coeffs, vec &Pan2DCoeffs){
        /**
        Method to initialize 2 panels for a particular fiber
        @param ChebFiberPoints = points on all fibers 
        @param FibForceDensities = fiber force densities on all fibers
        @param iFib = fiber number
        @return This is a void method, but it modifies the following arrays: the positions, force densities, and coefficients
        of the position and tangent vector on each of the 2 panels. 
        **/
        int start = _NChebPerFib*iFib;
        // Points
        vec TwoPanelsPoints(2*3*_NChebPerFib,0.0);
        MatVec(2*_NChebPerFib, _NChebPerFib, 3, _TwoPanMatrix, ChebFiberPoints, start, TwoPanelsPoints);
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, TwoPanelsPoints, 0, Pan1Pts);
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, TwoPanelsPoints, _NChebPerFib, Pan2Pts);
        // Forces
        vec TwoPanelsForceDs(2*3*_NChebPerFib,0.0);
        MatVec(2*_NChebPerFib, _NChebPerFib, 3, _TwoPanMatrix, FibForceDensities, start, TwoPanelsForceDs);
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, TwoPanelsForceDs, 0, Pan1FDens);
        MatVec(_NUpsample, _NChebPerFib, 3, _UpsamplingMatrix, TwoPanelsForceDs, _NChebPerFib, Pan2FDens);
        // 2 panel coefficients
        MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, TwoPanelsPoints, 0, Pan1Coeffs);
        DifferentiateCoefficients(Pan1Coeffs,3,Pan1DCoeffs);
        MatVec(_NChebPerFib, _NChebPerFib, 3, _ValtoCoeffMatrix, TwoPanelsPoints, _NChebPerFib, Pan2Coeffs);
        DifferentiateCoefficients(Pan2Coeffs,3,Pan2DCoeffs);
    }

    void calcCLVelocity(const vec &FinitePartCoefficients, const vec &DerivCoefficients, 
                        const vec &forceDCoefficients, double tapprox, vec3 &CLpart){
        /**
        Method to calculate the centerline velocity of a fiber at coordinate tapprox in [-1,1]
        @param FinitePartCoefficients = coefficients of the Chebyshev series of the FP velocity
        @param DerivCoefficients = coefficients of the Chebyshev series of the tangent vectors
        @param forceDCoefficients = coefficients of the Chebyshev series of the force densities
        @param tapprox = centerline coordinate on [-1,1]
        @param CLpart = centerline velocity (modified here)
        **/
        eval_Cheb3Dirs(FinitePartCoefficients, tapprox, CLpart); // evaluate finite part velocity and add to CLpart
        // Compute velocity due to local drag
        vec3 Xs = {0,0,0};
        vec3 forceDen = {0,0,0};
        eval_Cheb3Dirs(DerivCoefficients, tapprox, Xs);
        eval_Cheb3Dirs(forceDCoefficients, tapprox, forceDen);
        double Xsdotf = dot(Xs,forceDen);
        double s = (tapprox+1.0)*_L/2.0;
        //std::cout << "Testing new local drag routine with _delta and s=" << _delta << " , " << s << std::endl;
        double c = -log(_epsilon*_epsilon);
        if (_delta < 0.5){
            if (s > 0.5*_L){
                s = _L-s; // reflect
            }
            // Regularize to cylindrical fibers
            double x = 2*s/_L-1;
            double regwt = tanh((x+1)/_delta)-tanh((x-1)/_delta)-1;
            double sNew = regwt*s + (1-regwt*regwt)*_delta*_L*0.5;
            c = log(4.0*sNew*(_L-sNew)/pow(_epsilon*_L,2));
            //std::cout << "s= " << s << " sNew = " << sNew << std::endl;
        }
        //std::cout << "Coefficient " << c << std::endl;
        for (int d =0; d < 3; d++){
            CLpart[d] += 1.0/(8*M_PI*mu)*(c*(forceDen[d]+Xs[d]*Xsdotf)+(forceDen[d]-3*Xs[d]*Xsdotf));
        }
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
