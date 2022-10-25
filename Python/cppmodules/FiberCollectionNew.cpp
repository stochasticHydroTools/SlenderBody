#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include "VectorMethods.cpp"
#include "utils.h"
#include "RPYKernelEvaluator.cpp"

/**
    FiberCollection.cpp
    C++ class that updates arrays of many fiber positions, etc.
    5 main public methods
    1) CorrectNonLocalVelocity - correct the velocity from Ewald
    2) FinitePartVelocity - get the velocity due to the finite part integral
    3) SubtractAllRPY - subtract the free space RPY kernel (from the Ewald result)
    4) RodriguesRotations - return the rotated X and Xs
    5) ApplyPreconditioner 
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class FiberCollectionNew {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    FiberCollectionNew(int nFib,int nXPerFib,int nTauPerFib,double L, double a, double mu,int nThreads){
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
        @param NForDirectQyad =  number of points for upsampled direct quadrature
        @param deltain = fraction of fiber with ellipsoidal tapering
        @param nThreads = # of OpenMP threads for parallel calculations
        **/ 
        _NFib = nFib;
        _nXPerFib = nXPerFib;
        _nTauPerFib = nTauPerFib;
        _nOMPThr = nThreads;
        _a = a;
        _L = L;
        _mu = mu;
    }
    
    void initMobilityMatrices(npDoub pyXNodes,npDoub &pyCs,npDoub pyFPMatrix, npDoub pyDoubFPMatrix, 
        npDoub pyRL2aResampMatrix, npDoub pyRLess2aWts,npDoub pyXDiffMatrix,bool useRPY){    
        /**
        Python wrapper to initialize variables for finite part integration in C++. 
        @param pyFPMatrix = matrix that maps g function coefficients to velocity values on the N point Cheb grid (2D numpy array)
        @param pyDiffMatrix = Chebyshev differentiation matrix on the N point Chebyshev grid (2D numpy aray)
        **/

        // allocate std::vector (to pass to the C++ function)
        _XNodes = vec(_nXPerFib);
        std::memcpy(_XNodes.data(),pyXNodes.data(),pyXNodes.size()*sizeof(double));
        _LocalDragCs = vec(pyCs.size());
        std::memcpy(_LocalDragCs.data(),pyCs.data(),pyCs.size()*sizeof(double));
        
        _FinitePartMatrix = vec(pyFPMatrix.size());
        _XDiffMatrix = vec(pyXDiffMatrix.size());
        std::memcpy(_FinitePartMatrix.data(),pyFPMatrix.data(),pyFPMatrix.size()*sizeof(double));
        std::memcpy(_XDiffMatrix.data(),pyXDiffMatrix.data(),pyXDiffMatrix.size()*sizeof(double));
        
        _stackedXDiffMatrix = vec(9*_XDiffMatrix.size());
        for (int i = 0; i < _nXPerFib; i++){
            for (int j = 0; j < _nXPerFib; j++){
                for (int d = 0; d < 3; d++){
                    _stackedXDiffMatrix[(3*i+d)*3*_nXPerFib+3*j+d] =  _XDiffMatrix[i*_nXPerFib+j];
                }
            }
        }
        
        // RPY mobility
        _DbltFinitePartMatrix = vec(pyDoubFPMatrix.size());
        _RL2aResampMatrix = vec(pyRL2aResampMatrix.size());
        _RLess2aWts = vec(pyRLess2aWts.size());
        std::memcpy(_DbltFinitePartMatrix.data(),pyDoubFPMatrix.data(),pyDoubFPMatrix.size()*sizeof(double));
        std::memcpy(_RL2aResampMatrix.data(),pyRL2aResampMatrix.data(),pyRL2aResampMatrix.size()*sizeof(double));
        std::memcpy(_RLess2aWts.data(),pyRLess2aWts.data(),pyRLess2aWts.size()*sizeof(double));
        _HalfNForSmall = _RL2aResampMatrix.size()/(2*_nXPerFib*_nXPerFib);
        initLocalDragCoeffsRPY();
        
    }
    
    void initResamplingMatrices(int Nuniform, npDoub pyMatfromNtoUniform){    
        /**
        Python wrapper to initialize variables for finite part integration in C++. 
        @param pyFPMatrix = matrix that maps g function coefficients to velocity values on the N point Cheb grid (2D numpy array)
        @param pyDiffMatrix = Chebyshev differentiation matrix on the N point Chebyshev grid (2D numpy aray)
        **/

        // allocate std::vector (to pass to the C++ function)
        _nUni = Nuniform;
        _UnifRSMat = vec(pyMatfromNtoUniform.size());
        std::memcpy(_UnifRSMat.data(),pyMatfromNtoUniform.data(),pyMatfromNtoUniform.size()*sizeof(double));
    }
    
    void initMatricesForPreconditioner(npDoub pyD4BC, npDoub pyD4BCForce, npDoub pyD4BCForceHalf,
        npDoub pyXFromTau, npDoub pyTauFromX,npDoub pyMP,npDoub pyWTildeX, npDoub pyWTildeXInverse){
        /*
        @param pyUpsampMat = upsampling matrix from N to 2N
        @param pyUpsampledchebPolys = Chebyshev polynomials of degree 0, ..., N-2 sampled on the 2N grid
        @param pyLeastSquaresDownsampler = matrix that takes 2N grid points to N grid points using least squares (see paper)
        @param pyDpInv = pseudo-inverse of cheb differentiation matrix on 2N grid
        @param pyweightedUpsampler = upsampling matrix (with weights) for calculating K^*, 
        @param pyD4BC = matrix that calculates the bending force
        @param pyCs = local drag coefficients (1D)
        All of the matrices are passed as 2D numpy arrays which are then converted to matrices in C (row major) order
        */
        _D4BC = vec(pyD4BC.size());
        _D4BCForce = vec(pyD4BCForce.size());
        _BendForceMatHalf = vec(pyD4BCForceHalf.size());
        _XFromTauMat = vec(pyXFromTau.size());
        _WTildeX = vec(pyWTildeX.size());
        _WTildeXInv = vec(pyWTildeXInverse.size());
        _TauFromXMat = vec(pyTauFromX.size());
        _MidpointSamp = vec(pyMP.size());
        
         
        std::memcpy(_D4BC.data(),pyD4BC.data(),pyD4BC.size()*sizeof(double));
        std::memcpy(_D4BCForce.data(),pyD4BCForce.data(),pyD4BCForce.size()*sizeof(double));
        std::memcpy(_BendForceMatHalf.data(),pyD4BCForceHalf.data(),pyD4BCForceHalf.size()*sizeof(double));
        std::memcpy(_XFromTauMat.data(),pyXFromTau.data(),pyXFromTau.size()*sizeof(double));
        std::memcpy(_WTildeX.data(),pyWTildeX.data(),pyWTildeX.size()*sizeof(double));
        std::memcpy(_WTildeXInv.data(),pyWTildeXInverse.data(),pyWTildeXInverse.size()*sizeof(double));
        std::memcpy(_TauFromXMat.data(),pyTauFromX.data(),pyTauFromX.size()*sizeof(double));
        std::memcpy(_MidpointSamp.data(),pyMP.data(),pyMP.size()*sizeof(double));
    }

    py::array evalBendForces(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Forces(chebPts.size());
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_D4BC,false,FiberPoints,LocalForces);
            for (int i=0; i < 3*_nXPerFib; i++){
                Forces[3*_nXPerFib*iFib+i] = LocalForces[i];
            }
        }
        return make1DPyArray(Forces);
    }
    
    py::array getUniformPoints(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec AllUniPoints(3*_NFib*_nUni);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalUniPts(3*_nUni);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(3*_nUni,3*_nXPerFib,1,1.0,0.0,_UnifRSMat,false,FiberPoints,LocalUniPts);
            for (int i=0; i < 3*_nUni; i++){
                AllUniPoints[3*_nUni*iFib+i] = LocalUniPts[i];
            }
        }
        return makePyDoubleArray(AllUniPoints);
    }
    
    py::array evalLocalVelocities(npDoub pyPositions, npDoub pyForceDensities, bool ExactRPY){
        /*
        Evaluate the local velocities M_local*f on the fiber.
        @param pyTangents = tangent vectors
        @param pyForceDensities = force densites on the fiber
        @return 1D numpy array of local velocities
        */
        vec Positions(pyPositions.size());
        std::memcpy(Positions.data(),pyPositions.data(),pyPositions.size()*sizeof(double));
        vec ForceDensities(pyForceDensities.size());
        std::memcpy(ForceDensities.data(),pyForceDensities.data(),pyForceDensities.size()*sizeof(double));
        vec AllLocalVelocities(Positions.size());
        
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < _NFib; iFib++){
            int start = 3*iFib*_nXPerFib;
            vec LocalPts(3*_nXPerFib);
            vec LocalForceDens(3*_nXPerFib);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                LocalPts[iPt]=Positions[start+iPt];
                LocalForceDens[iPt]=ForceDensities[start+iPt];
            }
            vec MLoc(9*_nXPerFib*_nXPerFib);
            calcMLocal(LocalPts, ExactRPY, MLoc);
            vec LocalVel(3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,MLoc,false,LocalForceDens,LocalVel);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                AllLocalVelocities[start+iPt]=LocalVel[iPt];
            }
        }
        return make1DPyArray(AllLocalVelocities);
    }
    
    npDoub FinitePartVelocity(npDoub pyChebPoints, npDoub pyForceDens, bool exactRPY){
        /**
        Compute the finite part velocity on all fibers 
        @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
        @param pyForceDens = Npts * 3 numpy array (2D) of force densities on ALL fibers
        @return velocities = Npts * 3 numpy array (1D, row stacked "C" order) of finite part velocities
        **/

        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec FDens(pyForceDens.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(FDens.data(),pyForceDens.data(),pyForceDens.size()*sizeof(double));

        vec AlluFP(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_nXPerFib); // determine how many fiber we are working with
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*_nXPerFib;
            vec LocalPts(3*_nXPerFib);
            vec LocalForceDens(3*_nXPerFib);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForceDens[iPt]=FDens[start+iPt];
            }
            vec MFP(9*_nXPerFib*_nXPerFib);
            addMFP(LocalPts, exactRPY, MFP);
            if (exactRPY){
                addMFPDoublet(LocalPts,MFP);
                addNumericalRLess2a(LocalPts,MFP);
            }
            vec LocalFPVel(3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,MFP,false,LocalForceDens,LocalFPVel);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                AlluFP[start+iPt]=LocalFPVel[iPt];
            }
        } // end fiber loop
        return make1DPyArray(AlluFP);
    } // end calcFP velocity
    
    py::array applyPreconditioner(npDoub pyPoints, npDoub pyTangents,npDoub pyb, 
        double impcodt, bool implicitFP, bool exactRPY, double svdtol){
        /*
        Apply preconditioner to obtain lambda and alpha. 
        @param pyPoints = chebyshev fiber pts as an Npts x 3 2D numpy array
        @param pyTangents = tangent vectors as an Npts x 3 2D numpy array (or row-stacked 1D, it gets converted anyway)
        @param pyb = RHS vector b 
        @param impcodt = implicit factor * dt (usually dt/2)
        @return a vector (lambda,alpha) with the values on all fibers in a collection
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec b(pyb.size());
        std::memcpy(b.data(),pyb.data(),pyb.size()*sizeof(double));
                
        vec LambasAndAlphas(6*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            vec bFib(3*_nXPerFib), MinvbFib(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
                bFib[i] = b[3*_nXPerFib*iFib+i];
                MinvbFib[i] = b[3*_nXPerFib*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            
            // K and K*
            vec K(9*_nXPerFib*_nXPerFib);
            calcK(LocalTangents,K);
            vec Kt(9*_nXPerFib*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,K,true,_WTildeX,Kt); // K*=K^T*Wtilde
           
            // M and Schur complement block B
            vec M(3*_nXPerFib*3*_nXPerFib);
            calcMLocal(LocalPoints, exactRPY, M);
            if (implicitFP){
                addMFP(LocalPoints, exactRPY, M);
                if (exactRPY){
                    addMFPDoublet(LocalPoints,M);
                    addNumericalRLess2a(LocalPoints,M);
                }
            }
            
            // Overwrite K ->  K-impcoeff*dt*M*D4BC*K;
            vec D4BCK(9*_nXPerFib*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,_D4BC,false,K,D4BCK);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,-impcodt,1.0,M,false,D4BCK,K);
            vec KWithImp(9*_nXPerFib*_nXPerFib);
            std::memcpy(KWithImp.data(),K.data(),K.size()*sizeof(double));
            
            // Factor M
            int sysDim = 3*_nXPerFib;
            int ipiv [sysDim];
            LAPACKESafeCall(LAPACKE_dgetrf(LAPACK_ROW_MAJOR, sysDim, sysDim, &*M.begin(), sysDim, ipiv));
            // Solve M^-1*b first
            LAPACKESafeCall(LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, 1, &*M.begin(), sysDim, ipiv, &*MinvbFib.begin(), 1));
            // Overwrite b2 -> Schur RHS
            vec RHS(sysDim);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1.0,1.0,0.0,Kt,false,MinvbFib,RHS);
            
            // Overwrite KWithImp-> M^(-1)*KWithImp
            vec SchurComplement(9*_nXPerFib*_nXPerFib);
            LAPACKESafeCall(LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, sysDim, &*M.begin(), sysDim, ipiv, &*KWithImp.begin(),sysDim));
            BlasMatrixProduct(sysDim, sysDim,sysDim, 1.0,0.0,Kt,false,KWithImp,SchurComplement); // Kt*(M^-1*B)
            vec alphas(3*_nXPerFib);
            SolveWithPseudoInverse(sysDim, SchurComplement, RHS, alphas, svdtol);
            
            // Back solve to get lambda
            vec Kalpha(sysDim);
            BlasMatrixProduct(sysDim,sysDim,1,1.0,0.0,K,false, alphas, Kalpha);
            for (int i=0; i < sysDim; i++){
                Kalpha[i]-=bFib[i];
            }
            // Overwrite Kalpha -> lambda
            LAPACKESafeCall(LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N',sysDim, 1, &*M.begin(), sysDim, ipiv, &*Kalpha.begin(), 1));
            for (int i=0; i < sysDim; i++){
                LambasAndAlphas[sysDim*iFib+i] = Kalpha[i]; // lambda
                LambasAndAlphas[sysDim*(_NFib+iFib)+i] = alphas[i];
            }
        }
        
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);      
    }
    
    py::array applyThermalPreconditioner(npDoub pyPoints, npDoub pyTangents, npDoub pyMidpoints, npDoub pyU0, npDoub pyExForceDen,
        npDoub pyBendMatX0, npDoub pyRandVec1, npDoub pyRandVec2, double dt, double impco, bool ModifiedBE, bool implicitFP, 
        bool exactRPY, double kbT, double svdtol, double eigValueThreshold){
        /*
        Apply preconditioner to obtain lambda and alpha. 
        @param pyPoints = chebyshev fiber pts as an Npts x 3 2D numpy array
        @param pyTangents = tangent vectors as an Npts x 3 2D numpy array (or row-stacked 1D, it gets converted anyway)
        @param impcodt = implicit factor * dt (usually dt/2)
        @return a vector (lambda,alpha) with the values on all fibers in a collection
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec Midpoints(pyMidpoints.size());
        std::memcpy(Midpoints.data(),pyMidpoints.data(),pyMidpoints.size()*sizeof(double));
        vec RandVec1(pyRandVec1.size());
        std::memcpy(RandVec1.data(),pyRandVec1.data(),pyRandVec1.size()*sizeof(double));
        vec RandVec2(pyRandVec2.size());
        std::memcpy(RandVec2.data(),pyRandVec2.data(),pyRandVec2.size()*sizeof(double));
        vec U0(pyU0.size());
        std::memcpy(U0.data(),pyU0.data(),pyU0.size()*sizeof(double));
        vec ExForceDensity(pyExForceDen.size());
        std::memcpy(ExForceDensity.data(),pyExForceDen.data(),pyExForceDen.size()*sizeof(double));
        vec BendMatX0(pyBendMatX0.size());
        std::memcpy(BendMatX0.data(),pyBendMatX0.data(),pyBendMatX0.size()*sizeof(double));

        vec LambasAndAlphas(6*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            vec LocalRand1(3*_nXPerFib);
            vec LocalRand2(3*_nXPerFib);
            vec LocalU0(3*_nXPerFib);
            vec LocalExForceDen(3*_nXPerFib);
            vec LocalExForce(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
                LocalRand1[i] = RandVec1[3*_nXPerFib*iFib+i];
                LocalRand2[i] = RandVec2[3*_nXPerFib*iFib+i];
                LocalU0[i] = U0[3*_nXPerFib*iFib+i];
                LocalExForceDen[i] = ExForceDensity[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_WTildeX,true,LocalExForceDen,LocalExForce);
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            vec3 ThisMidpoint; 
            for (int d=0; d < 3; d++){
                ThisMidpoint[d] = Midpoints[3*iFib+d];
            }
            
            // M and Schur complement block B
            vec M(3*_nXPerFib*3*_nXPerFib);
            calcMLocal(LocalPoints, exactRPY, M);
            if (implicitFP){
                addMFP(LocalPoints, exactRPY, M);
                if (exactRPY){
                    addMFPDoublet(LocalPoints,M);
                    addNumericalRLess2a(LocalPoints,M);
                }
            }
            // Symmetrize mobility and apply M^(1/2) 
            vec MWsym(3*_nXPerFib*3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,M,false,_WTildeXInv,MWsym);
            vec EigVecs(3*_nXPerFib*3*_nXPerFib), EigVals(3*_nXPerFib);
            SymmetrizeAndDecomposePositive(3*_nXPerFib, MWsym, eigValueThreshold, EigVecs, EigVals);
            vec MWHalfeta(3*_nXPerFib);
            vec MWMinusHalfeta(3*_nXPerFib);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 0.5, EigVecs, EigVals, LocalRand1, MWHalfeta);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib,-0.5, EigVecs, EigVals, LocalRand1, MWMinusHalfeta);
            
            // Compute rotation rates Omega_tilde
            vec KInverse(9*_nXPerFib*_nXPerFib);
            calcKInverse(LocalTangents, KInverse);
            vec OmegaTilde(3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,sqrt(2*kbT/dt), 0.0,KInverse,false,MWHalfeta,OmegaTilde);
            
            vec3 MidpointTilde;
            vec XTilde(3*_nXPerFib);
            vec TauTilde(3*_nTauPerFib);
            OneFibRotationUpdate(LocalTangents,ThisMidpoint,OmegaTilde,TauTilde,MidpointTilde,XTilde,0.5*dt,1e-10);
            // Compute tilde variables
            vec Ktilde(9*_nXPerFib*_nXPerFib);
            calcK(TauTilde,Ktilde);
            vec Mtilde(3*_nXPerFib*3*_nXPerFib);
            calcMLocal(XTilde, exactRPY, Mtilde);
            if (implicitFP){
                addMFP(XTilde, exactRPY, Mtilde);
                if (exactRPY){
                    addMFPDoublet(XTilde,Mtilde);
                    addNumericalRLess2a(XTilde,Mtilde);
                }
            }
            // Symmetrize mobility tilde
            vec MWsymTilde(3*_nXPerFib*3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,Mtilde,false,_WTildeXInv,MWsymTilde);
            vec EigVecsTilde(3*_nXPerFib*3*_nXPerFib), EigValsTilde(3*_nXPerFib);
            SymmetrizeAndDecomposePositive(3*_nXPerFib, MWsymTilde, eigValueThreshold, EigVecsTilde, EigValsTilde);
            
            // RFD term
            vec URFDPlus(3*_nXPerFib);
            vec URFDMinus(3*_nXPerFib);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecsTilde, EigValsTilde, MWMinusHalfeta, URFDPlus); // Mtilde*U_B
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecs, EigVals, MWMinusHalfeta, URFDMinus); // M*U_B
            vec URFD(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                URFD[i] = sqrt(2.0*kbT/dt)*(URFDPlus[i]-URFDMinus[i]);
            }
            
            // Add the thermal flow and RFD to U0
            vec AddBERandomVel(3*_nXPerFib);
            vec BendHalfEta(3*_nXPerFib);
            if (abs(impco-1.0) < 1e-10 && ModifiedBE){
                BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,sqrt(dt/2),0.0,_BendForceMatHalf,false,LocalRand2,BendHalfEta); 
                ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecsTilde, EigValsTilde, BendHalfEta, AddBERandomVel);          
            }    
            vec PenaltyTerm(3*_nXPerFib);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecsTilde, EigValsTilde, BendMatX0, PenaltyTerm);  
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalU0[i]+=URFD[i]-PenaltyTerm[i]+sqrt(2.0*kbT/dt)*(MWHalfeta[i]+AddBERandomVel[i]);
            }
            
            int sysDim = 3*_nXPerFib;
            // Saddle point solve with Ktilde and Mtilde
            vec D4BCK(sysDim*sysDim);
            BlasMatrixProduct(sysDim,sysDim,sysDim,1.0,0.0,_D4BCForce,false,Ktilde,D4BCK);
            vec M_D4BCK(sysDim*sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, D4BCK, M_D4BCK); 
            vec KWithImp(sysDim*sysDim);
            for (int i = 0; i < sysDim*sysDim; i++){
                KWithImp[i] = Ktilde[i]-impco*dt*M_D4BCK[i];
            }

            // Solve M^-1*b first
            vec NewLocalU0(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, -1.0, EigVecsTilde, EigValsTilde, LocalU0, NewLocalU0);  
            // Schur RHS
            vec RHS(sysDim);
            vec ForceN(sysDim);
            BlasMatrixProduct(sysDim,sysDim,1.0,1.0,0.0,_D4BCForce,false,LocalPoints,ForceN);
            for (int i=0; i < sysDim; i++){
                ForceN[i]+=NewLocalU0[i]+LocalExForce[i];
            }
            BlasMatrixProduct(sysDim,sysDim,1.0,1.0,1.0,Ktilde,true,ForceN,RHS);
            
            // Overwrite KWithImp-> M^(-1)*KWithImp
            vec SchurComplement(sysDim*sysDim);
            vec KWithImpNew(sysDim*sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, -1.0, EigVecsTilde, EigValsTilde, KWithImp, KWithImpNew);  
            BlasMatrixProduct(sysDim, sysDim,sysDim, 1.0,0.0,Ktilde,true,KWithImpNew,SchurComplement); // K'*(M^-1*B)
            vec alphas(3*_nXPerFib);
            SolveWithPseudoInverse(sysDim, SchurComplement, RHS, alphas, svdtol);
            
            // Back solve to get lambda
            vec Kalpha(sysDim);
            BlasMatrixProduct(sysDim,sysDim,1,1.0,0.0,Ktilde,false, alphas, Kalpha);
            // Overwrite Kalpha -> lambda
            vec Lambda(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, -1.0, EigVecsTilde, EigValsTilde, Kalpha, Lambda);  
            for (int i=0; i < sysDim; i++){
                LambasAndAlphas[sysDim*iFib+i] = Lambda[i]-ForceN[i]; // lambda
                LambasAndAlphas[sysDim*(_NFib+iFib)+i] = alphas[i];
            }
        }
        
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);      
    }
    

    npDoub RodriguesRotations(npDoub pyXsn, npDoub pyMidpoints, npDoub pyAllAlphas,double dt, double nOmTol){
        /*
        Method to update the tangent vectors and positions using Rodriguez rotation and Chebyshev integration
        for many fibers in parallel. 
        @param pyXn = 2D numpy array of current Chebyshev points (time n) on the fiber, 
        @param pyXsn = 2D numpy array of current tangent vectors (time n)
        @param pyAllAlphas = rotation rates and midpoint velocities
        @param dt = timestep
        @return TransformedXsAndX = all of the new tangent vectors and positions (2D numpy array)
        */
  
        // allocate std::vector (to pass to the C++ function)
        vec Midpoints(pyMidpoints.size());
        vec Xsn(pyXsn.size());
        vec AllAlphas(pyAllAlphas.size());

        // copy py::array -> std::vector
        std::memcpy(Midpoints.data(),pyMidpoints.data(),pyMidpoints.size()*sizeof(double));
        std::memcpy(Xsn.data(),pyXsn.data(),pyXsn.size()*sizeof(double));
        std::memcpy(AllAlphas.data(),pyAllAlphas.data(),pyAllAlphas.size()*sizeof(double));

        // call pure C++ function
        vec TransformedXsAndX(6*_NFib*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < _NFib; iFib++){
            // Rotate tangent vectors
            vec LocalAlphas(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalAlphas[i] = AllAlphas[3*_nXPerFib*iFib+i];
            }
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Xsn[3*_nTauPerFib*iFib+i];
            }
            vec3 ThisMidpoint; 
            for (int d=0; d < 3; d++){
                ThisMidpoint[d] = Midpoints[3*iFib+d];
            }
            vec RotatedTaus(3*_nTauPerFib);
            vec3 NewMidpoint;
            vec NewX(3*_nXPerFib);
            OneFibRotationUpdate(LocalTangents, ThisMidpoint,LocalAlphas, RotatedTaus,NewMidpoint,NewX,dt,nOmTol);
            for (int iPt=0; iPt < _nTauPerFib; iPt++){
                int Tauindex = iPt+iFib*_nTauPerFib;
                for (int d = 0; d< 3; d++){
                    TransformedXsAndX[3*Tauindex+d] = RotatedTaus[3*iPt+d];
                }
            }
            // Add the midpoint
            for (int d=0; d< 3; d++){
                TransformedXsAndX[3*_nTauPerFib*_NFib+3*iFib+d]=NewMidpoint[d];
            } 
            int startXs = 3*_nXPerFib*_NFib;
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                int Xindex = iPt+iFib*_nXPerFib;
                for (int d=0; d < 3; d++){
                    TransformedXsAndX[startXs+3*Xindex+d] = NewX[3*iPt+d];
                }
            }
        }
        return makePyDoubleArray(TransformedXsAndX);
    } // end Rodrigues rotations
    
        
    private:
    
    // Basic force
    int _nOMPThr, _NFib, _nXPerFib, _nTauPerFib;
    int _nUni;
    // Variables for K and elastic force
    vec _D4BC, _D4BCForce,_BendForceMatHalf, _XFromTauMat, _TauFromXMat, _MidpointSamp, _WTildeX, _WTildeXInv;
    // Variables for mobility
    vec _XNodes, _LocalDragCs, _FinitePartMatrix, _DbltFinitePartMatrix, _XDiffMatrix;
    vec _RL2aResampMatrix, _RLess2aWts, _stackedXDiffMatrix;
    vec _LocalStokeslet, _LocalDoublet, _LocalRLess2aI, _LocalRLess2aTau;
    vec _UnifRSMat;
    int _HalfNForSmall;
    double _L, _a, _mu;
    
    void initLocalDragCoeffsRPY(){
        _LocalStokeslet = vec(_nXPerFib);
        _LocalDoublet = vec(_nXPerFib);
        _LocalRLess2aI = vec(_nXPerFib);
        _LocalRLess2aTau = vec(_nXPerFib);
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            double s = _XNodes[iPt];
            _LocalStokeslet[iPt] = log(s*(_L-s)/(4.0*_a*_a));
            _LocalDoublet[iPt] = 1.0/(4.0*_a*_a)-1.0/(2.0*s*s)-1.0/(2.0*(_L-s)*(_L-s)); 
            _LocalRLess2aI[iPt] = 23.0/6.0;
            _LocalRLess2aTau[iPt] = 1.0/2.0;
            if (s < 2*_a){ 
                _LocalStokeslet[iPt] = log((_L-s)/(2.0*_a));
                _LocalDoublet[iPt] = 1.0/(8.0*_a*_a)-1.0/(2.0*(_L-s)*(_L-s));
                _LocalRLess2aI[iPt] = 23.0/12.0+4.0*s/(3.0*_a)-3.0*s*s/(16.0*_a*_a);
                _LocalRLess2aTau[iPt] = 1.0/4.0+s*s/(16.0*_a*_a);
            } else if (s > _L-2*_a){
                _LocalStokeslet[iPt] = log(s/(2.0*_a));
                _LocalDoublet[iPt] = 1.0/(8.0*_a*_a)-1.0/(2.0*s*s);
                _LocalRLess2aI[iPt] = 23.0/12.0+4.0*(_L-s)/(3.0*_a)-3.0*(_L-s)*(_L-s)/(16.0*_a*_a);
                _LocalRLess2aTau[iPt] = 1.0/4.0+(_L-s)*(_L-s)/(16.0*_a*_a);
            }
        }
    }
    
    void calcMLocal(const vec &ChebPoints, bool exactRPY, vec &M){
        /*
        Calculates the local drag matrix M. 
        Inputs: Xs = fiber tangent vectors as a 3N array
        */
        double viscInv = 1.0/(8.0*M_PI*_mu);
        vec Tangents(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,ChebPoints,Tangents); 
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            vec3 Tau;
            for (int id = 0; id < 3; id++){
                Tau[id] = Tangents[3*iPt+id];
            }
            double normTau = normalize(Tau);
            double normTauInv = 1.0/normTau;
            for (int id = 0; id < 3; id++){
                for (int jd =0; jd < 3; jd++){
                    double deltaij = 0;
                    if (id==jd){
                        deltaij=1;
                    }
                    double XsXs_ij =Tau[id]*Tau[jd];
                    if (exactRPY) { 
                        M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] = viscInv*(_LocalStokeslet[iPt]*(deltaij+XsXs_ij)*normTauInv+
                            2*_a*_a/3.0*_LocalDoublet[iPt]*(deltaij-3*XsXs_ij)*pow(normTauInv,3)+
                            (_LocalRLess2aI[iPt]*deltaij+_LocalRLess2aTau[iPt]*XsXs_ij)*normTauInv);
                    } else {
                        //M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] = viscInv*(_LocalDragCs[iPt]*(deltaij+XsXs_ij)/normTau+(deltaij-3*XsXs_ij)/pow(normTau,3));
                        M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] = viscInv*(_LocalDragCs[iPt]*(deltaij+XsXs_ij)+(deltaij-3*XsXs_ij));
                    }
                }
            }
        }
    }
    
    void addMFP(const vec &ChebPoints, bool exactRPY, vec &M){
        /*
        Adds the finite part matrix to the local drag matrix
        Inputs: chebPoints as a 3N array, Tangents as a 3N array
        */
        // Differentiate Xs to get Xss
        vec MFP(3*_nXPerFib*3*_nXPerFib,0.0);
        vec DFPart(3*_nXPerFib*3*_nXPerFib,0.0);
        vec Xs(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,ChebPoints,Xs);
        vec Xss(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,Xs,Xss);
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            vec3 TauHat, XssThis;
            for (int id = 0; id < 3; id++){
                TauHat[id] = Xs[3*iPt+id];
                XssThis[id] = Xss[3*iPt+id];
            }
            double normTau = normalize(TauHat);
            double normTauInv = 1.0/normTau;
            double XsDotXss = dot(TauHat,XssThis);
            for (int jPt=0; jPt < _nXPerFib; jPt++){
                if (iPt==jPt){
                    for (int id=0; id < 3; id++){
                        for (int jd=0; jd < 3; jd++){
                            double deltaij=0;
                            if (id==jd){
                                deltaij = 1;
                            }
                            double DiagPart1 = normTauInv*normTauInv*(0.5*(TauHat[id]*XssThis[jd]+TauHat[jd]*XssThis[id])
                                - 0.5*XsDotXss*deltaij - 1.5*XsDotXss*TauHat[id]*TauHat[jd]);                 
                            MFP[3*_nXPerFib*(3*iPt+id)+3*jPt+jd] +=
                                DiagPart1*_FinitePartMatrix[_nXPerFib*iPt+iPt];
                            DFPart[3*_nXPerFib*(3*iPt+id)+3*jPt+jd]=normTauInv*(deltaij+TauHat[id]*TauHat[jd])
                                *_FinitePartMatrix[_nXPerFib*iPt+iPt];
                        }
                    }
                } else {
                    vec3 rvec;
                    for (int d = 0; d < 3; d++){
                        rvec[d] = ChebPoints[3*iPt+d]-ChebPoints[3*jPt+d];
                    }
                    double r = sqrt(dot(rvec,rvec));
                    double oneoverr = 1.0/r;
                    double ds = _XNodes[jPt]-_XNodes[iPt];
                    double oneoverds = 1.0/ds;
                    for (int id=0; id < 3; id++){
                        for (int jd=0; jd < 3; jd++){
                            double deltaij=0;
                            if (id==jd){
                                deltaij = 1;
                            }
                            // Off diagonal part
                            MFP[3*_nXPerFib*(3*iPt+id)+3*jPt+jd] = (deltaij + rvec[id]*rvec[jd]*oneoverr*oneoverr)
                                *oneoverr*std::abs(ds)*oneoverds*_FinitePartMatrix[_nXPerFib*iPt+jPt];
                            // Diagonal part
                            MFP[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] -= (deltaij + TauHat[id]*TauHat[jd])*oneoverds*normTauInv
                                *_FinitePartMatrix[_nXPerFib*iPt+jPt];
                        }
                    }
                }
            }
        } // end forming matrices
        BlasMatrixProduct(3*_nXPerFib, 3*_nXPerFib, 3*_nXPerFib,1, 1, DFPart, false, _stackedXDiffMatrix, MFP);  // MFP -> DFPart*BigDiff+MFP
        for (uint i = 0; i <M.size(); i++){
            M[i]+=MFP[i];
        }
    } // end method
    
    void addMFPDoublet(const vec &ChebPoints, vec &M){
        /*
        Adds the finite part matrix for the doublet to the local drag matrix for the Stokeslet
        and the finite part matrix for the Stokeslet
        Inputs: chebPoints as a 3N array, Tangents as a 3N array
        */
        // Differentiate Xs to get Xss
        vec MFP(3*_nXPerFib*3*_nXPerFib,0.0);
        vec DFPart(3*_nXPerFib*3*_nXPerFib,0.0);
        vec Xs(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,ChebPoints,Xs);
        vec Xss(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,Xs,Xss);
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            vec3 TauHat, XssThis;
            for (int id = 0; id < 3; id++){
                TauHat[id] = Xs[3*iPt+id];
                XssThis[id] = Xss[3*iPt+id];
            }
            double normTau = normalize(TauHat);
            double normTauInv = 1.0/normTau;
            double XsDotXss = dot(TauHat,XssThis);
            for (int jPt=0; jPt < _nXPerFib; jPt++){
                if (iPt==jPt){
                    for (int id=0; id < 3; id++){
                        for (int jd=0; jd < 3; jd++){
                            double deltaij=0;
                            if (id==jd){
                                deltaij = 1;
                            }
                            MFP[3*_nXPerFib*(3*iPt+id)+3*jPt+jd] +=pow(normTauInv,4)*
                                (-1.5*(TauHat[id]*XssThis[jd]+TauHat[jd]*XssThis[id]) - 1.5*XsDotXss*deltaij
                                 +7.5*XsDotXss*TauHat[id]*TauHat[jd])*_DbltFinitePartMatrix[_nXPerFib*iPt+iPt];
                            DFPart[3*_nXPerFib*(3*iPt+id)+3*jPt+jd]=pow(normTauInv,3)*
                                (deltaij-3*TauHat[id]*TauHat[jd])*_DbltFinitePartMatrix[_nXPerFib*iPt+iPt];
                        }
                    }
                } else {
                    vec3 rvec;
                    for (int d = 0; d < 3; d++){
                        rvec[d] = ChebPoints[3*iPt+d]-ChebPoints[3*jPt+d];
                    }
                    double r = sqrt(dot(rvec,rvec));
                    double oneoverr = 1.0/r;
                    double ds = _XNodes[jPt]-_XNodes[iPt];
                    double oneoverds = 1.0/ds;
                    for (int id=0; id < 3; id++){
                        for (int jd=0; jd < 3; jd++){
                            double deltaij=0;
                            if (id==jd){
                                deltaij = 1;
                            }
                            // Off diagonal part
                            MFP[3*_nXPerFib*(3*iPt+id)+3*jPt+jd] = (deltaij -3*rvec[id]*rvec[jd]*oneoverr*oneoverr)
                                *pow(oneoverr,3)*pow(std::abs(ds),3)*oneoverds*_DbltFinitePartMatrix[_nXPerFib*iPt+jPt];
                            // Diagonal part
                            MFP[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] -= (deltaij -3*TauHat[id]*TauHat[jd])*oneoverds*pow(normTauInv,3)
                                *_DbltFinitePartMatrix[_nXPerFib*iPt+jPt];
                        }
                    }
                }
            }
        } // end forming matrices
        BlasMatrixProduct(3*_nXPerFib, 3*_nXPerFib, 3*_nXPerFib,1, 1, DFPart, false, _stackedXDiffMatrix, MFP);  // MFP -> DFPart*BigDiff+MFP
        for (uint i = 0; i <M.size(); i++){
            M[i]+=2.0*_a*_a/3.0*MFP[i];
        }
    } // end method
    
    void addNumericalRLess2a(const vec &ChebPoints, vec &M){
        vec Tangents(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,ChebPoints,Tangents);
        double viscInv = 1.0/(8.0*M_PI*_mu);
        vec XSamples(_nXPerFib*2*_HalfNForSmall*3,0.0);
        //vec Mpart(3*_nXPerFib*3*_nXPerFib);
        BlasMatrixProduct(_nXPerFib*2*_HalfNForSmall, _nXPerFib, 3, 1.0,0.0,_RL2aResampMatrix, false, ChebPoints, XSamples);
        // Xsamples is an (Ncheb*Nhalf*2 x 3) matrix
        // For each point, the first Nhalf rows are for the [s-2a,s] domain and the next Nhalf are for the [s,s+2a] domain
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int iD=0; iD < 2; iD++){
                vec RowVec(9*_HalfNForSmall);
                for (int jPt = 0; jPt <  _HalfNForSmall; jPt++){
                    vec3 rvec;
                    int jStartIndex = iPt*6*_HalfNForSmall+iD*_HalfNForSmall*3+jPt*3;
                    for (int id=0; id < 3; id++){
                        rvec[id]=ChebPoints[3*iPt+id]-XSamples[jStartIndex+id];
                    }
                    double r = normalize(rvec);
                    double diagterm = viscInv*(4.0/(3.0*_a)*(1.0-9.0*r/(32.0*_a)));
                    double offDiagTerm = viscInv*(1.0/(8.0*_a*_a))*r;
                    for (int id=0; id < 3; id++){
                        for (int jd=0; jd < 3; jd++){
                            double deltaij=0;
                            if (id==jd){
                                deltaij=1;
                            }
                            RowVec[id*(3*_HalfNForSmall)+3*jPt+jd] = diagterm*deltaij;
                            if (r > 1e-12){
                                RowVec[id*(3*_HalfNForSmall)+3*jPt+jd] +=offDiagTerm*rvec[id]*rvec[jd];
                            }
                        }
                    }
                } // end jPt
                // Add the row vector to the grand mobility (multiplying by the correct resampling matrix and 
                // the correct integration weight
                int startMat = (2*iPt+iD)*(_nXPerFib*_HalfNForSmall);
                for (int iR = 0; iR < 3; iR++){
                    for (int jPt=0; jPt < _nXPerFib; jPt++){
                        for (int h = 0; h < _HalfNForSmall; h++){
                            for (int jd=0; jd < 3; jd++){
                                M[(3*iPt+iR)*3*_nXPerFib+3*jPt+jd]+=
                                    RowVec[iR*3*_HalfNForSmall+3*h+jd]*
                                    _RL2aResampMatrix[startMat+h*_nXPerFib+jPt]*_RLess2aWts[(2*iPt+iD)*_HalfNForSmall+h];
                            }
                        }
                    }
                }
            } // end domain
        } // end iPt
        // Subtract diagonal part from M (previously included in local drag terms)
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            vec3 Tau;
            for (int id = 0; id < 3; id++){
                Tau[id]=Tangents[3*iPt+id];
            }
            double normTau = normalize(Tau);
            double normTauInv = 1.0/normTau;
            for (int id = 0; id < 3; id++){
                for (int jd =0; jd < 3; jd++){
                    double deltaij = 0;
                    if (id==jd){
                        deltaij=1;
                    }
                    double XsXs_ij =Tau[id]*Tau[jd];
                    M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd]-= viscInv*(_LocalRLess2aI[iPt]*deltaij+_LocalRLess2aTau[iPt]*XsXs_ij)*normTauInv;
                }
            }
        }
    }
    
    void calcK(const vec &TanVecs, vec &K){
        // First calculate cross product matrix
        vec XMatTimesCPMat(3*_nTauPerFib*3*_nXPerFib);
        vec CPMatrix(9*_nTauPerFib*_nTauPerFib);
        for (int iPt = 0; iPt < _nTauPerFib; iPt++){
            CPMatrix[3*iPt*3*_nTauPerFib+3*iPt+1] = -TanVecs[3*iPt+2];
            CPMatrix[3*iPt*3*_nTauPerFib+3*iPt+2] =  TanVecs[3*iPt+1];
            CPMatrix[(3*iPt+1)*3*_nTauPerFib+3*iPt]= TanVecs[3*iPt+2];
            CPMatrix[(3*iPt+1)*3*_nTauPerFib+3*iPt+2]= -TanVecs[3*iPt];
            CPMatrix[(3*iPt+2)*3*_nTauPerFib+3*iPt]= -TanVecs[3*iPt+1];
            CPMatrix[(3*iPt+2)*3*_nTauPerFib+3*iPt+1]= TanVecs[3*iPt];
        }
        BlasMatrixProduct(3*_nXPerFib, 3*_nTauPerFib, 3*_nTauPerFib,-1.0, 0.0,_XFromTauMat, false, CPMatrix,XMatTimesCPMat);
        // Add the identity part to K
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int iD=0; iD < 3; iD++){
                int rowstart = (3*iPt+iD)*3*_nXPerFib;
                int Xrowstart = (3*iPt+iD)*3*_nTauPerFib;
                // Copy the row entry from XMatTimesCPMat
                for (int iEntry=0; iEntry < 3*_nTauPerFib; iEntry++){
                    K[rowstart+iEntry]=XMatTimesCPMat[Xrowstart+iEntry];
                }
            }
        }  
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int iD=0; iD < 3; iD++){
                int rowstart = (3*iPt+iD)*3*_nXPerFib;
                K[rowstart+3*_nTauPerFib+iD]=1;
            }
        }
    }
    
    void calcKInverse(const vec &TanVecs, vec &KInverse){
        // NOT CORRECT RIGHT NOW!
        // First calculate cross product matrix
        vec CPMatrix(9*_nTauPerFib*_nTauPerFib);
        for (int iPt = 0; iPt < _nTauPerFib; iPt++){
            CPMatrix[3*iPt*3*_nTauPerFib+3*iPt+1] = -TanVecs[3*iPt+2];
            CPMatrix[3*iPt*3*_nTauPerFib+3*iPt+2] =  TanVecs[3*iPt+1];
            CPMatrix[(3*iPt+1)*3*_nTauPerFib+3*iPt]= TanVecs[3*iPt+2];
            CPMatrix[(3*iPt+1)*3*_nTauPerFib+3*iPt+2]= -TanVecs[3*iPt];
            CPMatrix[(3*iPt+2)*3*_nTauPerFib+3*iPt]= -TanVecs[3*iPt+1];
            CPMatrix[(3*iPt+2)*3*_nTauPerFib+3*iPt+1]= TanVecs[3*iPt];
        }
        BlasMatrixProduct(3*_nTauPerFib, 3*_nTauPerFib, 3*_nXPerFib,1.0, 0.0,CPMatrix, false, _TauFromXMat,KInverse);
        // Add the midpoint part for the last three rows
        for (int iD=0; iD < 3; iD++){
            int rowstart=3*_nXPerFib*(3*_nTauPerFib+iD);
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                    KInverse[rowstart+3*iPt+iD]=_MidpointSamp[iPt];
            }
        }
    }

    void OneFibRotationUpdate(const vec &Xsn, const vec3 &Midpoint, const vec &AllAlphas, vec &RotatedTaus,
        vec3 &NewMidpoint, vec &NewX,double dt, double nOmTol){
        for (int iPt=0; iPt < _nTauPerFib; iPt++){
            vec3 XsToRotate, Omega;
            for (int d =0; d < 3; d++){
                XsToRotate[d] = Xsn[3*iPt+d];
                Omega[d] = AllAlphas[3*iPt+d];
            }
            double nOm = normalize(Omega);
            if (nOm > nOmTol){
                double theta = nOm*dt;    
                double OmegaDotXs = dot(Omega,XsToRotate);
                vec3 OmegaCrossXs;
                cross(Omega,XsToRotate,OmegaCrossXs);
                for (int d = 0; d < 3; d++){
                    RotatedTaus[3*iPt+d] = XsToRotate[d]*cos(theta)+OmegaCrossXs[d]*sin(theta)+Omega[d]*OmegaDotXs*(1.0-cos(theta));
                }
            } else{
                for (int d = 0; d < 3; d++){
                    RotatedTaus[3*iPt+d] = XsToRotate[d];
                }
            }
        } // end rotate tangent vectors
        // Add the midpoint
        for (int d=0; d< 3; d++){
            NewMidpoint[d]=Midpoint[d]+dt*AllAlphas[3*_nTauPerFib+d];
        } 
        // Obtain X from Xs and midpoint
        BlasMatrixProduct(3*_nXPerFib, 3*_nTauPerFib, 1,1.0, 0.0,_XFromTauMat, false, RotatedTaus,NewX);
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int d=0; d < 3; d++){
                NewX[3*iPt+d]+=NewMidpoint[d];
            }
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
        // Return a 1D py array
        // allocate py::array (to pass the result of the C++ function to Python)
        auto pyArray = py::array_t<double>(cppvec.size());
        auto result_buffer = pyArray.request();
        double *result_ptr    = (double *) result_buffer.ptr;
        // copy std::vector -> py::array
        std::memcpy(result_ptr,cppvec.data(),cppvec.size()*sizeof(double));
        return pyArray;
    }

};

PYBIND11_MODULE(FiberCollectionNew, m) {
    py::class_<FiberCollectionNew>(m, "FiberCollectionNew")
        .def(py::init<int,int,int,double,double,double,int>())
        .def("initMatricesForPreconditioner", &FiberCollectionNew::initMatricesForPreconditioner)
        .def("initMobilityMatrices", &FiberCollectionNew::initMobilityMatrices)
        .def("initResamplingMatrices", &FiberCollectionNew::initResamplingMatrices)
        .def("evalBendForces",&FiberCollectionNew::evalBendForces)
        .def("getUniformPoints", &FiberCollectionNew::getUniformPoints)
        .def("evalLocalVelocities",&FiberCollectionNew::evalLocalVelocities)
        .def("FinitePartVelocity",&FiberCollectionNew::FinitePartVelocity)
        .def("applyPreconditioner",&FiberCollectionNew::applyPreconditioner)
        .def("applyThermalPreconditioner",&FiberCollectionNew::applyThermalPreconditioner)
        .def("RodriguesRotations",&FiberCollectionNew::RodriguesRotations);
}
