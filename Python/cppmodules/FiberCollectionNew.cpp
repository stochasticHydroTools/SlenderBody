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
#include "SingleFiberMobilityEvaluator.cpp"


/**
    FiberCollection.cpp
    C++ class that updates arrays of many fiber positions, etc.
    2 main public methods :
    1) RodriguesRotations - return the rotated X and Xs
    2) ApplyPreconditioner  - block diagonal solver
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class FiberCollectionNew {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    FiberCollectionNew(int nFib,int nXPerFib,int nTauPerFib,int nThreads,
        double a, double L, double mu,double kbT, double svdtol, double svdrigid, bool quadRPY, bool directRPY, bool oversampleRPY)
        :_MobilityEvaluator(nXPerFib,a,L,mu,quadRPY,directRPY,oversampleRPY){
        /**
        Initialize variables relating to each fiber
        nFib = number of fibers
        nXPerFib = number of collocation points per fiber
        nTauPerFib = number of tangent vectors on each fiber
        nThreads = number of OMP threads,  svdtol = tolerance for SVD of 
        semiflexible fibers, svdrigid = tolerance for SVD of rigid fibers, 
        kbT = thermal energy. Other parameters: see SingleFiberMobilityEvaluator class
        **/ 
        _NFib = nFib;
        _nXPerFib = nXPerFib;
        _nTauPerFib = nTauPerFib;
        _nOMPThr = nThreads;
        _svdtol = svdtol;
        _svdRigid = svdrigid;
        _kbT = kbT;
    }
    
    void initMobilityMatrices(npDoub pyXNodes,npDoub &pyCs,npDoub pyFPMatrix, npDoub pyDoubFPMatrix, 
        npDoub pyRL2aResampMatrix, npDoub pyRLess2aWts,npDoub pyXDiffMatrix,npDoub pyWTildeX, 
        npDoub pyWTildeXInverse, npDoub pyOversampleForDirectQuad, npDoub pyUpsamplingMatrix, double eigValThres){  

        _MobilityEvaluator.initMobilitySubmatrices(pyXNodes,pyCs,pyFPMatrix,pyDoubFPMatrix,pyRL2aResampMatrix,
            pyRLess2aWts,pyXDiffMatrix,pyWTildeXInverse,pyOversampleForDirectQuad,pyUpsamplingMatrix,eigValThres);
        _nUpsample = _MobilityEvaluator.getNupsample();
        _WTildeX = vec(pyWTildeX.size());
        _WTildeXInv = vec(pyWTildeXInverse.size());
        std::memcpy(_WTildeX.data(),pyWTildeX.data(),pyWTildeX.size()*sizeof(double));
        std::memcpy(_WTildeXInv.data(),pyWTildeXInverse.data(),pyWTildeXInverse.size()*sizeof(double));
    }
    
    void initResamplingMatrices(int Nuniform, npDoub pyMatfromNtoUniform){    
        // allocate std::vector (to pass to the C++ function)
        _nUni = Nuniform;
        _UnifRSMat = vec(pyMatfromNtoUniform.size());
        std::memcpy(_UnifRSMat.data(),pyMatfromNtoUniform.data(),pyMatfromNtoUniform.size()*sizeof(double));
    }
    
    void initMatricesForPreconditioner(npDoub pyD4BC, npDoub pyD4BCForce, npDoub pyD4BCForceHalf,
        npDoub pyXFromTau, npDoub pyTauFromX,npDoub pyMP,npDoub pyBendMatX0){

        _D4BC = vec(pyD4BC.size());
        _D4BCForce = vec(pyD4BCForce.size());
        _BendForceMatHalf = vec(pyD4BCForceHalf.size());
        _XFromTauMat = vec(pyXFromTau.size());
        _TauFromXMat = vec(pyTauFromX.size());
        _MidpointSamp = vec(pyMP.size());
        _BendMatX0 = vec(pyBendMatX0.size());  
         
        std::memcpy(_D4BC.data(),pyD4BC.data(),pyD4BC.size()*sizeof(double));
        std::memcpy(_D4BCForce.data(),pyD4BCForce.data(),pyD4BCForce.size()*sizeof(double));
        std::memcpy(_BendForceMatHalf.data(),pyD4BCForceHalf.data(),pyD4BCForceHalf.size()*sizeof(double));
        std::memcpy(_XFromTauMat.data(),pyXFromTau.data(),pyXFromTau.size()*sizeof(double));
        std::memcpy(_TauFromXMat.data(),pyTauFromX.data(),pyTauFromX.size()*sizeof(double));
        std::memcpy(_MidpointSamp.data(),pyMP.data(),pyMP.size()*sizeof(double));
        std::memcpy(_BendMatX0.data(),pyBendMatX0.data(),pyBendMatX0.size()*sizeof(double));
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
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_D4BCForce,false,FiberPoints,LocalForces);
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
    
    py::array getUpsampledPoints(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec AllUpsampPoints(3*_NFib*_nUpsample);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalUpsampPts(3*_nUpsample);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            _MobilityEvaluator.UpsampledPoints(FiberPoints, LocalUpsampPts);
            for (int i=0; i < 3*_nUpsample; i++){
                AllUpsampPoints[3*_nUpsample*iFib+i] = LocalUpsampPts[i];
            }
        }
        return makePyDoubleArray(AllUpsampPoints);
    }
    
    py::array getUpsampledForces(npDoub pyForces){
        vec chebForces(pyForces.size());
        std::memcpy(chebForces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        vec AllUpsampForces(3*_NFib*_nUpsample);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberForces(3*_nXPerFib);
            vec LocalUpsampForces(3*_nUpsample);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberForces[i] = chebForces[3*_nXPerFib*iFib+i];
            }
            _MobilityEvaluator.UpsampledForces(FiberForces, LocalUpsampForces);
            for (int i=0; i < 3*_nUpsample; i++){
                AllUpsampForces[3*_nUpsample*iFib+i] = LocalUpsampForces[i];
            }
        }
        return makePyDoubleArray(AllUpsampForces);
    }
    
    py::array getDownsampledVelocities(npDoub pyUpVelocities){
        vec chebUpVels(pyUpVelocities.size());
        std::memcpy(chebUpVels.data(),pyUpVelocities.data(),pyUpVelocities.size()*sizeof(double));
        vec AllDownVels(3*_NFib*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec FiberUpVels(3*_nUpsample);
            vec LocalDownSampVel(3*_nXPerFib);
            for (int i=0; i < 3*_nUpsample; i++){
                FiberUpVels[i] = chebUpVels[3*_nUpsample*iFib+i];
            }
            _MobilityEvaluator.DownsampledVelocity(FiberUpVels, LocalDownSampVel);
            for (int i=0; i < 3*_nXPerFib; i++){
                AllDownVels[3*_nXPerFib*iFib+i] = LocalDownSampVel[i];
            }
        }
        return makePyDoubleArray(AllDownVels);
    }
       
    py::array evalLocalVelocities(npDoub pyChebPoints, npDoub pyForces, bool includeFP){
        /*
        Evaluate the local velocities M_local*f on the fiber.
        pyPositions = collocation points
        pyForceDensities = force densites on the fiber
        Exact RPY = whether we're doing exact RPY hydro or SBT
        @return 1D numpy array of local velocities
        */
        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec Forces(pyForces.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec AllLocalVelocities(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_nXPerFib); // determine how many fibers we are working with
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*_nXPerFib;
            vec LocalPts(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForces[iPt]=Forces[start+iPt];
            }
            vec MLoc(9*_nXPerFib*_nXPerFib), EigVecs(9*_nXPerFib*_nXPerFib), EigVals(3*_nXPerFib);
            _MobilityEvaluator.MobilityForceMatrix(LocalPts, includeFP, MLoc,EigVecs,EigVals);
            vec LocalVel(3*_nXPerFib);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecs, EigVals, LocalForces, LocalVel);  
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                AllLocalVelocities[start+iPt]=LocalVel[iPt];
            }
        }
        return make1DPyArray(AllLocalVelocities);
    }
    
    py::array SingleFiberRPYSum(int NPerFib, npDoub pyChebPoints, npDoub pyForces){
        /*
        Evaluate the free space RPY kernel M*F on the fiber.
        pyChebPoints = collocation points
        pyForces = forces on the fiber
        @return 1D numpy array of local velocities
        */
        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec Forces(pyForces.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec AllLocalVelocities(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*NPerFib); // determine how many fibers we are working with
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*NPerFib;
            vec LocalPts(3*NPerFib);
            vec LocalForces(3*NPerFib);
            for (int iPt=0; iPt < 3*NPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForces[iPt]=Forces[start+iPt];
            }
            vec ULocal(3*NPerFib);
            _MobilityEvaluator.RPYVelocityFromForces(LocalPts, LocalForces,ULocal);
            for (int iPt=0; iPt < 3*NPerFib; iPt++){
                AllLocalVelocities[start+iPt]=ULocal[iPt];
            }
        }
        return make1DPyArray(AllLocalVelocities);

    }
    
    npDoub FinitePartVelocity(npDoub pyChebPoints, npDoub pyForces){
        /**
        Compute the finite part velocity on all fibers 
        @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
        @param pyForceDens = Npts * 3 numpy array (2D) of forces on ALL fibers
        @return velocities = Npts * 3 numpy array (1D, row stacked "C" order) of finite part velocities
        **/

        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec Forces(pyForces.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec AlluFP(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_nXPerFib); // determine how many fiber we are working with
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*_nXPerFib;
            vec LocalPts(3*_nXPerFib);
            vec LocalForceDens(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForces[iPt]=Forces[start+iPt];
            }
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_WTildeXInv,false,LocalForces,LocalForceDens);
            vec MFP(9*_nXPerFib*_nXPerFib);
            _MobilityEvaluator.MobilityMatrix(LocalPts, false,true, MFP);
            vec LocalFPVel(3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,MFP,false,LocalForceDens,LocalFPVel);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                AlluFP[start+iPt]=LocalFPVel[iPt];
            }
        } // end fiber loop
        return make1DPyArray(AlluFP);
    } // end calcFP velocity
    
    
    py::array ThermalTranslateAndDiffuse(npDoub pyPoints, npDoub pyTangents,npDoub pyRandVec1,bool implicitFP, double dt){

        /*
        Random Brownian translation and rotation, assuming the fiber is rigid
        Inputs: pyPoints = Chebyshev points, pyTangents = tangent vectors, 
        pyRandVec1 = nFib*6 random vector, implicitFP = whether the mobility includes the finite
        part integral (intra-fiber hydro) or not, dt = time step size
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec RandVec1(pyRandVec1.size());
        std::memcpy(RandVec1.data(),pyRandVec1.data(),pyRandVec1.size()*sizeof(double));   
        
        vec AllAlphas(3*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            
            // K and K*
            vec KRig(3*_nXPerFib*6);
            calcKRigid(LocalTangents,KRig);
            // M and Schur complement block B
            int sysDim = 3*_nXPerFib;
            vec MWsym(sysDim*sysDim), EigVecs(sysDim*sysDim), EigVals(sysDim);
            _MobilityEvaluator.MobilityForceMatrix(LocalPoints, implicitFP, MWsym, EigVecs, EigVals);
            
           
            vec MinvK(sysDim*6);
            vec SchurComplement(6*6);
            ApplyMatrixPowerFromEigDecomp(sysDim, -1.0, EigVecs, EigVals, KRig, MinvK);  
            BlasMatrixProduct(6, sysDim,6, 1.0,0.0,KRig,true,MinvK,SchurComplement); // K'*(M^-1*K)
            vec alphas(6);
            vec LocalRand1(6);
            for (int i=0; i < 6; i++){
                LocalRand1[i] = RandVec1[6*iFib+i];
            }
            vec SchurEigVecs(6*6), SchurEigVals(6);
            SymmetrizeAndDecomposePositive(6, SchurComplement,0.0, SchurEigVecs, SchurEigVals); 
            double rigidSVDTol = 2.0*_kbT*dt/(_svdRigid*_svdRigid);
            bool normalize = false;
            bool half = true;
            SolveWithPseudoInverse(6, SchurComplement, LocalRand1, alphas, rigidSVDTol,normalize,half,6); // N*LocalRand1
            int startAlphInd=3*iFib*_nXPerFib;
            for (int iPt = 0; iPt < _nTauPerFib; iPt++){
                for (int d=0; d < 3;d++){
                    AllAlphas[startAlphInd+3*iPt+d]=sqrt(2.0*_kbT/dt)*alphas[d];
                }
            }
            for (int d=0; d < 3;d++){
                AllAlphas[startAlphInd+3*_nTauPerFib+d]=sqrt(2.0*_kbT/dt)*alphas[3+d];
            } 
        }
        // Return 1D numpy array
        return make1DPyArray(AllAlphas);      
    }
    
    py::array applyPreconditioner(npDoub pyPoints, npDoub pyTangents,npDoub pyExForces,npDoub pyURHS,
        double impco, double dt, bool implicitFP,bool rigid){
        /*
        Apply preconditioner to obtain lambda and alpha. 
        pyPoints = chebyshev fiber pts as an Npts x 3 2D numpy array
        pyTangents = tangent vectors as an Npts x 3 2D numpy array (or row-stacked 1D, it gets converted anyway)
        pyExForceDens = external force density on the fibers
        impco = implicit coefficient (1 for backward Euler, 1/2 for Crank-Nicolson)
        dt = time step size
        implicitFP = whether to include intra-fiber hydro in mobility or not
        rigid = whether fibers are rigid (obviously)
        @return a vector (lambda,alpha) with the values on all fibers in a collection
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec ExForces(pyExForces.size());
        std::memcpy(ExForces.data(),pyExForces.data(),pyExForces.size()*sizeof(double));
        vec U_RHS(pyURHS.size());
        std::memcpy(U_RHS.data(),pyURHS.data(),pyURHS.size()*sizeof(double));
                
        vec LambasAndAlphas(6*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            vec MinvbFib(3*_nXPerFib), LocalExForce(3*_nXPerFib), LocalURHS(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
                LocalExForce[i] = ExForces[3*_nXPerFib*iFib+i];
                LocalURHS[i] = U_RHS[3*_nXPerFib*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            
            int FullSize = 3*_nXPerFib;
            vec ForceN(FullSize,0.0);
            for (int i=0; i < FullSize; i++){
                ForceN[i]+=LocalExForce[i];
            }
            int systemSize = 3*_nXPerFib;
            if (rigid){
                systemSize = 6;
            }
            vec alphas(systemSize), Lambda(3*_nXPerFib);
            SolveSaddlePoint(LocalPoints, LocalTangents, ForceN, LocalURHS, impco,dt, rigid, implicitFP, alphas,Lambda);
                       
            for (int i=0; i < FullSize; i++){
                LambasAndAlphas[FullSize*iFib+i] =Lambda[i]; // lambda
                if (rigid) {
                    if (i < FullSize-3){
                        LambasAndAlphas[FullSize*(_NFib+iFib)+i] = alphas[i % 3]; // 0, 1,or 2
                    } else { 
                        LambasAndAlphas[FullSize*(_NFib+iFib)+i] = alphas[(i % 3)+3]; // 0, 1,or 2
                    }
                } else {
                    LambasAndAlphas[FullSize*(_NFib+iFib)+i] = alphas[i];
                }
            }
        }
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);      
    }
    
    py::array KAlphaKTLambda(npDoub pyTangents,npDoub pyAlphas,npDoub pyLambdas,bool rigid){
        /*
        Compute the products K*alpha and K^T*Lambda
        */

        // allocate std::vector (to pass to the C++ function)
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec Alphas(pyAlphas.size());
        std::memcpy(Alphas.data(),pyAlphas.data(),pyAlphas.size()*sizeof(double));
        vec Lambdas(pyLambdas.size());
        std::memcpy(Lambdas.data(),pyLambdas.data(),pyLambdas.size()*sizeof(double));
        
        int systemSize = 3*_nXPerFib;
        if (rigid){
            systemSize = 6;
        }
        vec KAlphaKTLambda(3*_nXPerFib*_NFib+systemSize*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){        
            vec LocalLams(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalLams[i] = Lambdas[3*_nXPerFib*iFib+i];
            }
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            vec LocalAlphas(systemSize);
            for (int i=0; i < systemSize; i++){
                LocalAlphas[i] = Alphas[3*_nXPerFib*iFib+i];
                if (rigid && i > 2){
                    LocalAlphas[i] = Alphas[3*_nXPerFib*iFib+3*(_nXPerFib-1)+(i%3)]; 
                }   
            }
            // Form K and compute products with it
            vec K(3*_nXPerFib*systemSize);
            if (rigid){
                calcKRigid(LocalTangents,K);
            } else {
                calcK(LocalTangents,K);
            }
            vec LocKAlphas(3*_nXPerFib), LocKTLambda(3*_nXPerFib);
            BlasMatrixProduct(3*_nXPerFib,systemSize,1,1.0, 0.0,K,false,LocalAlphas,LocKAlphas);
            BlasMatrixProduct(systemSize,3*_nXPerFib,1,1.0, 0.0,K,true,LocalLams,LocKTLambda);
            for (int i=0; i < 3*_nXPerFib; i++){
                KAlphaKTLambda[3*_nXPerFib*iFib+i] = LocKAlphas[i];
            }
            for (int i=0; i < systemSize; i++){
                KAlphaKTLambda[3*_nXPerFib*_NFib+iFib*systemSize+i] = LocKTLambda[i];
            }
        }
        return make1DPyArray(KAlphaKTLambda);
    }
            
    
    py::array applyThermalPreconditioner(npDoub pyPoints, npDoub pyTangents, npDoub pyMidpoints, npDoub pyU0, npDoub pyExForces,
        npDoub pyRandVec1, npDoub pyRandVec2, double impco, double dt, bool ModifiedBE, bool implicitFP){
        /*
        Apply preconditioner to obtain lambda and alpha for SEMIFLEXIBLE FLUCTUATING FIBERS.  
        pyPoints = chebyshev fiber pts as an Npts x 3 2D numpy array
        pyTangents = tangent vectors as an Npts x 3 2D numpy array (or row-stacked 1D, it gets converted anyway)
        pyMidpoints = midpoints of the fibers
        pyU0 = background velocity
        pyExForceDens = external force density on the fibers
        pyRandVec1 and pyRandVec2 = nFib*3*Nx vectors of random standard Gaussian variables
        impco = implicit coefficient (1 for backward Euler, 1/2 for Crank-Nicolson)
        dt = time step size
        ModifiedBE = whether to add the extra terms to get increased accuracy of backward Euler.
        implicitFP = whether to include intra-fiber hydro in mobility or not
        rigid = whether fibers are rigid (obviously)
        @return a vector (lambda,alpha) with the values on all fibers in a collection
        */

        std::cout << "This needs to be updated to pass force (incl. elastic) and velocity" << std::endl;
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
        vec ExForces(pyExForces.size());
        std::memcpy(ExForces.data(),pyExForces.data(),pyExForces.size()*sizeof(double));
        int sysDim = 3*_nXPerFib;

        vec LambasAndAlphas(6*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
	        vec LocalPoints(sysDim);
            vec LocalTangents(3*_nTauPerFib);
            vec LocalRand1(sysDim);
            vec LocalRand2(sysDim);
            vec LocalU0(sysDim);
            vec LocalExForce(sysDim);
            for (int i=0; i < sysDim; i++){
                LocalPoints[i] = chebPoints[sysDim*iFib+i];
                LocalRand1[i] = RandVec1[sysDim*iFib+i];
                LocalRand2[i] = RandVec2[sysDim*iFib+i];
                LocalU0[i] = U0[sysDim*iFib+i];
                LocalExForce[i] = ExForces[sysDim*iFib+i];
            }
            // Convert force to force density
            vec ForceN(sysDim);
            BlasMatrixProduct(sysDim,sysDim,1.0,1.0,0.0,_D4BCForce,false,LocalPoints,ForceN);
            for (int i=0; i < sysDim; i++){
                ForceN[i]+=LocalExForce[i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            vec3 ThisMidpoint; 
            for (int d=0; d < 3; d++){
                ThisMidpoint[d] = Midpoints[3*iFib+d];
            }
            
            // M and Schur complement block B
            vec MWsym(sysDim*sysDim), EigVecs(sysDim*sysDim), EigVals(sysDim);
            _MobilityEvaluator.MobilityForceMatrix(LocalPoints, implicitFP,MWsym, EigVecs, EigVals);
            vec MWHalfeta(sysDim);
            vec MWMinusHalfeta(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 0.5, EigVecs, EigVals, LocalRand1, MWHalfeta);
            ApplyMatrixPowerFromEigDecomp(sysDim,-0.5, EigVecs, EigVals, LocalRand1, MWMinusHalfeta);
            
            // Compute rotation rates Omega_tilde
            vec KInverse(sysDim*sysDim);
            calcKInverse(LocalTangents, KInverse);
            vec OmegaTilde(sysDim);
            BlasMatrixProduct(sysDim,sysDim,1,sqrt(2.0*_kbT/dt), 0.0,KInverse,false,MWHalfeta,OmegaTilde);
            
            vec3 MidpointTilde;
            vec XTilde(sysDim);
            vec TauTilde(3*_nTauPerFib);
            OneFibRotationUpdate(LocalTangents,ThisMidpoint,OmegaTilde,TauTilde,MidpointTilde,XTilde,0.5*dt,1e-10);
            
            // Compute tilde variables
            vec MWsymTilde(sysDim*sysDim), EigVecsTilde(sysDim*sysDim), EigValsTilde(sysDim);
            _MobilityEvaluator.MobilityForceMatrix(XTilde, implicitFP,MWsymTilde, EigVecsTilde, EigValsTilde);
            
            // RFD term
            vec URFDPlus(sysDim);
            vec URFDMinus(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, MWMinusHalfeta, URFDPlus); // Mtilde*U_B
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecs, EigVals, MWMinusHalfeta, URFDMinus); // M*U_B
            vec URFD(sysDim);
            for (int i=0; i < sysDim; i++){
                URFD[i] = sqrt(2.0*_kbT/dt)*(URFDPlus[i]-URFDMinus[i]);
            }
            
            // Add the thermal flow and RFD to U0
            vec AddBERandomVel(sysDim);
            vec BendHalfEta(sysDim);
            if (abs(impco-1.0) < 1e-10 && ModifiedBE){
                BlasMatrixProduct(sysDim,sysDim,1,sqrt(dt/2),0.0,_BendForceMatHalf,false,LocalRand2,BendHalfEta); 
                ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, BendHalfEta, AddBERandomVel);          
            }    
            vec PenaltyTerm(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, _BendMatX0, PenaltyTerm);  
            for (int i=0; i < sysDim; i++){
                LocalU0[i]+=URFD[i]-PenaltyTerm[i]+sqrt(2.0*_kbT/dt)*(MWHalfeta[i]+AddBERandomVel[i]);
            }
            
            vec alphas(sysDim), Lambda(sysDim);
            bool rigid = false;
            SolveSaddlePoint(XTilde, TauTilde, ForceN, LocalU0, impco,dt, rigid, implicitFP, alphas, Lambda);
            for (int i=0; i < sysDim; i++){
                LambasAndAlphas[sysDim*iFib+i] = Lambda[i]; // lambda
                LambasAndAlphas[sysDim*(_NFib+iFib)+i] = alphas[i];
            }
        }
        
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);      
    }
    

    npDoub RodriguesRotations(npDoub pyXsn, npDoub pyMidpoints, npDoub pyAllAlphas,double dt){
        /*
        Method to update the tangent vectors and positions using Rodriguez rotation and Chebyshev integration
        for many fibers in parallel. 
        pyXsn = 2D numpy array of current tangent vectors (time n)
        pyMidpoints = the midpoints at the current time
        pyAllAlphas = rotation rates and midpoint velocities
        dt = timestep
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
            OneFibRotationUpdate(LocalTangents, ThisMidpoint,LocalAlphas, RotatedTaus,NewMidpoint,NewX,dt,1e-10);
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
    int _nUni, _nUpsample;
    vec _UnifRSMat, _BendMatX0;
    double _svdtol, _svdRigid, _kbT;
    SingleFiberMobilityEvaluator _MobilityEvaluator;
    // Variables for K and elastic force
    vec _D4BC, _D4BCForce,_BendForceMatHalf, _XFromTauMat, _TauFromXMat, _MidpointSamp, _WTildeX, _WTildeXInv;
    
    
    void calcK(const vec &TanVecs, vec &K){
        /*
        Kinematic matrix for semiflexible fibers
        */
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
       
    void calcKRigid(const vec &TanVecs, vec &K){
        /*
        Kinematic matrix for rigid fibers
        */
        // First calculate cross product matrix
        vec XMatTimesCPMat(9*_nXPerFib);
        vec CPMatrix(9*_nTauPerFib);
        for (int iPt = 0; iPt < _nTauPerFib; iPt++){
            CPMatrix[9*iPt+1] = -TanVecs[3*iPt+2];
            CPMatrix[9*iPt+2] =  TanVecs[3*iPt+1];
            CPMatrix[9*iPt+3]= TanVecs[3*iPt+2];
            CPMatrix[9*iPt+5]= -TanVecs[3*iPt];
            CPMatrix[9*iPt+6]= -TanVecs[3*iPt+1];
            CPMatrix[9*iPt+7]= TanVecs[3*iPt];
        }
        BlasMatrixProduct(3*_nXPerFib, 3*_nTauPerFib, 3,-1.0, 0.0,_XFromTauMat, false, CPMatrix,XMatTimesCPMat);
        // Add the identity part to K
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int iD=0; iD < 3; iD++){
                int rowstart = (3*iPt+iD)*6;
                int Xrowstart = (3*iPt+iD)*3;
                // Copy the row entry from XMatTimesCPMat
                for (int iEntry=0; iEntry < 3; iEntry++){
                    K[rowstart+iEntry]=XMatTimesCPMat[Xrowstart+iEntry];
                }
            }
        } 
        for (int iPt=0; iPt < _nXPerFib; iPt++){
            for (int iD=0; iD < 3; iD++){
                int rowstart = (3*iPt+iD)*6;
                K[rowstart+3+iD]=1;
            }
        }
    }
    
    
    void calcKInverse(const vec &TanVecs, vec &KInverse){
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
    
    void SolveSaddlePoint(const vec &LocalPoints, const vec &LocalTangents, vec &ForceN, 
        const vec &U0, double impco, double dt, bool rigid, bool implicitFP, vec &alphas, vec &Lambda){
        // Solve saddle point system 
        // M[X]*(F^n + Lambda) + U0 = (K[X]-impcodt*M[X]*D4BC*K[X]) alpha 
        // K^T Lambda = 0
        int systemSize = 3*_nXPerFib;
        int FullSize = 3*_nXPerFib;
        if (rigid){
            systemSize = 6;
        }
        vec K(3*_nXPerFib*systemSize);
        if (rigid){
            calcKRigid(LocalTangents,K);
        } else {
            calcK(LocalTangents,K);
        }
       
        // M and Schur complement block B
        vec MWsym(FullSize*FullSize), EigVecs(FullSize*FullSize), EigVals(FullSize);
        _MobilityEvaluator.MobilityForceMatrix(LocalPoints, implicitFP, MWsym, EigVecs, EigVals);
       
        // Overwrite K ->  K-impcoeff*dt*MW*D4BC*K;
        vec D4BCK(FullSize*systemSize);
        vec MtimesD4BCK(FullSize*systemSize);
        BlasMatrixProduct(FullSize,FullSize,systemSize,1.0,0.0,_D4BCForce,false,K,D4BCK);
        ApplyMatrixPowerFromEigDecomp(FullSize, 1.0, EigVecs, EigVals, D4BCK, MtimesD4BCK); 
        vec KWithImp(FullSize*systemSize);
        for (int i = 0; i < FullSize*systemSize; i++){
            KWithImp[i] = K[i]-impco*dt*MtimesD4BCK[i];
        }
        
        // Factor M
        // Overwrite b2 -> Schur RHS
        // Solve M^-1*U0, add to force first
        vec NewLocalU0(FullSize);
        ApplyMatrixPowerFromEigDecomp(systemSize, -1.0, EigVecs, EigVals, U0, NewLocalU0);  
        for (int i=0; i < FullSize; i++){
            ForceN[i]+=NewLocalU0[i];
        }
        vec RHS(systemSize);
        BlasMatrixProduct(systemSize,FullSize,1.0,1.0,1.0,K,true,ForceN,RHS);
        
        // Overwrite KWithImp-> M^(-1)*KWithImp
        vec MinvKWithImp(3*_nXPerFib*systemSize);
        ApplyMatrixPowerFromEigDecomp(FullSize, -1.0, EigVecs, EigVals, KWithImp, MinvKWithImp); 
        vec SchurComplement(systemSize*systemSize);
        BlasMatrixProduct(systemSize, FullSize,systemSize, 1.0,0.0,K,true,MinvKWithImp,SchurComplement); // K^T*(M^-1*B)
        double tol = _svdtol;
        bool normalize = true;
        int maxModes = 2*_nTauPerFib+3;
        if (rigid){
            tol = 2.0*_kbT*dt/(_svdRigid*_svdRigid);
            normalize = false;
            maxModes = 6;
        }
        bool half = false;
        SolveWithPseudoInverse(systemSize, SchurComplement, RHS, alphas, tol,normalize,half,maxModes);
        
        // Back solve to get lambda
        vec Kalpha(FullSize);
        BlasMatrixProduct(FullSize,systemSize,1,1.0,0.0,KWithImp,false, alphas, Kalpha);
        // Overwrite Kalpha -> lambda
        ApplyMatrixPowerFromEigDecomp(FullSize, -1.0, EigVecs, EigVals, Kalpha, Lambda);  
        for (int i=0; i < FullSize; i++){
            Lambda[i]-=ForceN[i];
        }
    }

    void OneFibRotationUpdate(const vec &Xsn, const vec3 &Midpoint, const vec &AllAlphas, vec &RotatedTaus,
        vec3 &NewMidpoint, vec &NewX,double dt, double nOmTol){
        // Rotate tangent vectors for a single fiber.
        // Uses Rodrigues rotation formula. 
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
        .def(py::init<int,int,int,int,double,double,double,double,double,double,bool,bool,bool>())
        .def("initMatricesForPreconditioner", &FiberCollectionNew::initMatricesForPreconditioner)
        .def("initMobilityMatrices", &FiberCollectionNew::initMobilityMatrices)
        .def("initResamplingMatrices", &FiberCollectionNew::initResamplingMatrices)
        .def("evalBendForces",&FiberCollectionNew::evalBendForces)
        .def("getUniformPoints", &FiberCollectionNew::getUniformPoints)
        .def("getUpsampledPoints", &FiberCollectionNew::getUpsampledPoints)
        .def("getUpsampledForces", &FiberCollectionNew::getUpsampledForces)
        .def("getDownsampledVelocities", &FiberCollectionNew::getDownsampledVelocities)
        .def("evalLocalVelocities",&FiberCollectionNew::evalLocalVelocities)
        .def("SingleFiberRPYSum",&FiberCollectionNew::SingleFiberRPYSum)
        .def("FinitePartVelocity",&FiberCollectionNew::FinitePartVelocity)
        .def("applyPreconditioner",&FiberCollectionNew::applyPreconditioner)
        .def("KAlphaKTLambda",&FiberCollectionNew::KAlphaKTLambda)
        .def("applyThermalPreconditioner",&FiberCollectionNew::applyThermalPreconditioner)
        .def("RodriguesRotations",&FiberCollectionNew::RodriguesRotations)
        .def("ThermalTranslateAndDiffuse",&FiberCollectionNew::ThermalTranslateAndDiffuse);
}
