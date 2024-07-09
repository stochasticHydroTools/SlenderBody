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
#include "SingleFiberSaddlePointSolver.cpp"


/**
    FiberCollection.cpp
    This is the C++ class that is called from python's fiberCollection.py. 
    Its main purpose is to parallelize calculations with OpenMP and communicate with 
    Python. All of the actual calculations are done in SingleFiberSaddlePointSolver
    and SingleFiberMobilityEvaluator. This class serves to communicate between those classes
    and python. 
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class FiberCollectionC {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    FiberCollectionC(int nFib,int nXPerFib,int nTauPerFib,int nThreads,bool rigid,
        double a, double aFat, double L, double mu,double kbT, double svdtol, double svdrigid, bool quadRPY, bool directRPY, bool oversampleRPY)
        :_MobilityEvaluator(nXPerFib,a,L,mu,quadRPY,directRPY,oversampleRPY), 
         _FatMobilityEvaluator(nXPerFib,aFat,L,mu,quadRPY,directRPY,oversampleRPY){
        /**
        Initialize variables relating to each fiber
        nFib = number of fibers
        nXPerFib = number of collocation points per fiber
        nTauPerFib = number of tangent vectors on each fiber
        nThreads = number of OMP threads,
        rigid = if the fibers are rigid
        a = fiber radius, L = fiber length, mu = fluid viscosity,  svdtol = tolerance for SVD of 
        semiflexible fibers, svdrigid = tolerance for SVD of rigid fibers (this depends on the time step) 
        kbT = thermal energy. Other parameters: see SingleFiberMobilityEvaluator class
        **/ 
        _NFib = nFib;
        _nXPerFib = nXPerFib;
        _nTauPerFib = nTauPerFib;
        _nOMPThr = nThreads;
        _svdtol = svdtol;
        _svdRigid = svdrigid;
        _kbT = kbT;
        _rigid = rigid;
        for (int i=0; i < _NFib; i++){
            _SaddlePointSolvers.push_back(SingleFiberSaddlePointSolver(_nXPerFib,_nTauPerFib, rigid, svdtol, svdrigid));
        }
    }
    
    void initMobilityMatrices(npDoub pyXNodes,npDoub pyRegXNodes,npDoub pyFPMatrix, npDoub pyDoubFPMatrix, 
        npDoub pyRL2aResampMatrix, npDoub pyRLess2aWts,npDoub pyXDiffMatrix,npDoub pyWTildeX, 
        npDoub pyWTildeXInverse, npDoub pyOversampleForDirectQuad, npDoub pyUpsamplingMatrix, double eigValThres){  
        
        _MobilityEvaluator.initMobilitySubmatrices(pyXNodes,pyRegXNodes,pyFPMatrix,pyDoubFPMatrix,pyRL2aResampMatrix,
            pyRLess2aWts,pyXDiffMatrix,pyWTildeXInverse,pyOversampleForDirectQuad,pyUpsamplingMatrix,eigValThres);
        _nUpsample = _MobilityEvaluator.getNupsample();
        _WTildeX = vec(pyWTildeX.size());
        _WTildeXInv = vec(pyWTildeXInverse.size());
        std::memcpy(_WTildeX.data(),pyWTildeX.data(),pyWTildeX.size()*sizeof(double));
        std::memcpy(_WTildeXInv.data(),pyWTildeXInverse.data(),pyWTildeXInverse.size()*sizeof(double));
    }
    
    void initFatMobilityEvaluator(npDoub pyXNodes,npDoub pyRegXNodes,npDoub pyFPMatrix, npDoub pyDoubFPMatrix, 
        npDoub pyRL2aResampMatrix, npDoub pyRLess2aWts,npDoub pyXDiffMatrix,npDoub pyWTildeX,
        npDoub pyWTildeXInverse, npDoub pyOversampleForDirectQuad,npDoub pyUpsamplingMatrix, double eigValThres){  
        
        _FatMobilityEvaluator.initMobilitySubmatrices(pyXNodes,pyRegXNodes,pyFPMatrix,pyDoubFPMatrix,pyRL2aResampMatrix,
            pyRLess2aWts,pyXDiffMatrix,pyWTildeXInverse,pyOversampleForDirectQuad,pyUpsamplingMatrix,eigValThres);
    }
    
    void SetEigValThreshold(double Thresh){
        _MobilityEvaluator.SetEigValThreshold(Thresh);  
        _FatMobilityEvaluator.SetEigValThreshold(Thresh);
    }
    
    void initResamplingMatrices(int Nuniform, npDoub pyMatfromNtoUniform){    
        _nUni = Nuniform;
        _UnifRSMat = vec(pyMatfromNtoUniform.size());
        std::memcpy(_UnifRSMat.data(),pyMatfromNtoUniform.data(),pyMatfromNtoUniform.size()*sizeof(double));
    }
    
    void initMatricesForPreconditioner(npDoub pyD4BCForce, npDoub pyD4BCForceHalf,
        npDoub pyXFromTau, npDoub pyTauFromX,npDoub pyMP,npDoub pyBendMatX0){
        for (int i=0; i < _NFib; i++){
            _SaddlePointSolvers[i].initMatricesForPreconditioner(pyD4BCForce,pyXFromTau, pyTauFromX, pyMP);
        }

        _D4BCForce = vec(pyD4BCForce.size());
        _BendForceMatHalf = vec(pyD4BCForceHalf.size());
        _BendMatX0 = vec(pyBendMatX0.size());  
        _XFromTauMat = vec(pyXFromTau.size());
         
        std::memcpy(_D4BCForce.data(),pyD4BCForce.data(),pyD4BCForce.size()*sizeof(double));
        std::memcpy(_BendForceMatHalf.data(),pyD4BCForceHalf.data(),pyD4BCForceHalf.size()*sizeof(double));
        std::memcpy(_BendMatX0.data(),pyBendMatX0.data(),pyBendMatX0.size()*sizeof(double));
        std::memcpy(_XFromTauMat.data(),pyXFromTau.data(),pyXFromTau.size()*sizeof(double));
    }
    
    
    //===========================================
    //        METHODS FOR DETERMINISTIC FIBERS 
    //===========================================
    py::array evalBendForces(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Forces(chebPts.size());
        int NFibs = chebPts.size()/(3*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < NFibs; iFib++){
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
    
    npDoub evalBendStress(npDoub pyPoints){
        vec chebPts(pyPoints.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Stress(9);
        int NFibs = chebPts.size()/(3*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < NFibs; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_D4BCForce,false,FiberPoints,LocalForces);
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                for (int iD=0; iD < 3; iD++){
                    for (int jD=0; jD < 3; jD++){
                        #pragma omp atomic update
                        Stress[3*iD+jD] -= FiberPoints[3*iPt+iD]*LocalForces[3*iPt+jD];
                    }
                }
            }
        }
        return makePyDoubleArray(Stress);
    }
    
    npDoub evalStressFromForce(npDoub pyPoints, npDoub pyForces){
        vec chebPts(pyPoints.size());
        vec chebForces(pyForces.size());
        std::memcpy(chebPts.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        std::memcpy(chebForces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        vec Stress(9);
        int NFibs = chebPts.size()/(3*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < NFibs; iFib++){
            vec FiberPoints(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                FiberPoints[i] = chebPts[3*_nXPerFib*iFib+i];
                LocalForces[i] = chebForces[3*_nXPerFib*iFib+i];
            }
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                for (int iD=0; iD < 3; iD++){
                    for (int jD=0; jD < 3; jD++){
                        #pragma omp atomic update
                        Stress[3*iD+jD] -= FiberPoints[3*iPt+iD]*LocalForces[3*iPt+jD];
                    }
                }
            }
        }
        return makePyDoubleArray(Stress);
    }
    
    npDoub evalDriftPartofStress(npDoub pyTangents){
        vec chebTans(pyTangents.size());
        std::memcpy(chebTans.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec Stress(9);
        int NFibs = chebTans.size()/(3*_nTauPerFib);
        for (int iFib = 0; iFib < NFibs; iFib++){
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i < 3*_nTauPerFib; i++){
                LocalTangents[i] = chebTans[3*_nTauPerFib*iFib+i];
            }
            _SaddlePointSolvers[iFib].KInverseKProductForStress(LocalTangents,Stress);
        }
        return makePyDoubleArray(Stress);
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
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
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
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
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
    
    py::array getLHalfX(npDoub pyRand2){
        /*
        Multiplies L^(1/2)*eta, which is used in modified backward 
        Euler calculations.
        */
        vec chebRand(pyRand2.size());
        std::memcpy(chebRand.data(),pyRand2.data(),pyRand2.size()*sizeof(double));
        vec AllProducts(3*_NFib*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            vec LocalBERand(3*_nXPerFib);
            vec BendHalfEta(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalBERand[i] = chebRand[3*_nXPerFib*iFib+i];
            }
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,1,1.0,0.0,_BendForceMatHalf,false,LocalBERand,BendHalfEta); 
            for (int i=0; i < 3*_nXPerFib; i++){
                AllProducts[3*_nXPerFib*iFib+i] = BendHalfEta[i];
            }
        }
        return make1DPyArray(AllProducts);
    }
        
    py::array getDownsampledVelocities(npDoub pyUpVelocities){
        vec chebUpVels(pyUpVelocities.size());
        std::memcpy(chebUpVels.data(),pyUpVelocities.data(),pyUpVelocities.size()*sizeof(double));
        vec AllDownVels(3*_NFib*_nXPerFib);
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
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
       
    py::array evalLocalVelocities(npDoub pyChebPoints, npDoub pyForces, bool includeFP, bool Fat){
        /*
        Evaluate the local velocities M_local*f on the fiber.
        */
        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec Forces(pyForces.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec AllLocalVelocities(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_nXPerFib); // determine how many fibers we are working with
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator, _FatMobilityEvaluator)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*_nXPerFib;
            vec LocalPts(3*_nXPerFib);
            vec LocalForces(3*_nXPerFib);
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForces[iPt]=Forces[start+iPt];
            }
            vec MLoc(9*_nXPerFib*_nXPerFib), EigVecs(9*_nXPerFib*_nXPerFib), EigVals(3*_nXPerFib);
            if (Fat){
                _FatMobilityEvaluator.MobilityForceMatrix(LocalPts, includeFP, -1,MLoc,EigVecs,EigVals);
            } else {
                _MobilityEvaluator.MobilityForceMatrix(LocalPts, includeFP, -1,MLoc,EigVecs,EigVals);
            }
            vec LocalVel(3*_nXPerFib);
            ApplyMatrixPowerFromEigDecomp(3*_nXPerFib, 1.0, EigVecs, EigVals, LocalForces, LocalVel);  
            for (int iPt=0; iPt < 3*_nXPerFib; iPt++){
                AllLocalVelocities[start+iPt]=LocalVel[iPt];
            }
        }
        return make1DPyArray(AllLocalVelocities);
    }
    
    py::array SingleFiberRPYSum(int NPerFib, npDoub pyChebPoints, npDoub pyForces, bool Fat){
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
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator, _FatMobilityEvaluator)
        for (int iFib=0; iFib < NFibIn; iFib++){
            int start = 3*iFib*NPerFib;
            vec LocalPts(3*NPerFib);
            vec LocalForces(3*NPerFib);
            for (int iPt=0; iPt < 3*NPerFib; iPt++){
                LocalPts[iPt]=ChebPoints[start+iPt];
                LocalForces[iPt]=Forces[start+iPt];
            }
            vec ULocal(3*NPerFib);
            if (Fat){
                _FatMobilityEvaluator.RPYVelocityFromForces(LocalPts, LocalForces,ULocal);
            } else {
                _MobilityEvaluator.RPYVelocityFromForces(LocalPts, LocalForces,ULocal);
            }
            for (int iPt=0; iPt < 3*NPerFib; iPt++){
                AllLocalVelocities[start+iPt]=ULocal[iPt];
            }
        }
        return make1DPyArray(AllLocalVelocities);

    }
    
    npDoub FinitePartVelocity(npDoub pyChebPoints, npDoub pyForces){
        /**
        Compute the finite part velocity on all fibers. This is the total intra-fiber
        velocity minus the part from local drag. 
        **/

        // allocate std::vector (to pass to the C++ function)
        vec ChebPoints(pyChebPoints.size());
        vec Forces(pyForces.size());
        // copy py::array -> std::vector
        std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
        std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));
        
        vec AlluFP(ChebPoints.size(),0.0);
        int NFibIn = ChebPoints.size()/(3*_nXPerFib); // determine how many fiber we are working with
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
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
    
    
    py::array ThermalTranslateAndDiffuse(npDoub pyPoints, npDoub pyTangents,npDoub pyRandVec1,bool includeFP, double dt){

        /*
        Random Brownian translation and rotation, assuming the fiber is rigid
        Inputs: pyPoints = Chebyshev points, pyTangents = tangent vectors, 
        pyRandVec1 = nFib*6 random vector, includeFP = whether the local mobility includes the finite
        part integral (intra-fiber hydro) or not, dt = time step size
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec RandVec1(pyRandVec1.size());
        std::memcpy(RandVec1.data(),pyRandVec1.data(),pyRandVec1.size()*sizeof(double));   
        
        vec AllAlphas(6*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
        for (int iFib = 0; iFib < _NFib; iFib++){
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            vec LocalRand1(6);
            for (int i=0; i < 6; i++){
                LocalRand1[i] = RandVec1[6*iFib+i];
            }
            vec alphas(6);
            _SaddlePointSolvers[iFib].RigidBodyDisplacements(LocalTangents, LocalPoints, LocalRand1, _MobilityEvaluator,includeFP, _kbT*dt,alphas);
            int startAlphInd=6*iFib;
            for (int d=0; d < 3;d++){
                AllAlphas[startAlphInd+d]=sqrt(2.0*_kbT/dt)*alphas[d];
                AllAlphas[startAlphInd+3+d]=sqrt(2.0*_kbT/dt)*alphas[3+d];
            }
        }
        // Return 1D numpy array
        return make1DPyArray(AllAlphas);      
    }
    
    void FactorizePreconditioner(npDoub pyPoints, npDoub pyTangents,double impco, double dt, bool implicitFP, int NBands){
        /*
        See documentation for this method in fiberCollection.py
        */
        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
      
        int nAlphas = 3*_nXPerFib;
        if (_rigid){
            nAlphas = 6;
        }
        int nFibs = chebPoints.size()/(3*_nXPerFib);   
        vec LambasAndAlphas(3*_nXPerFib*nFibs+nAlphas*nFibs);
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
        for (int iFib = 0; iFib < nFibs; iFib++){
	        vec LocalPoints(3*_nXPerFib);
            vec LocalTangents(3*_nTauPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalPoints[i] = chebPoints[3*_nXPerFib*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            
            _SaddlePointSolvers[iFib].FormSaddlePointMatrices(LocalPoints, LocalTangents, impco, dt, implicitFP, _MobilityEvaluator, NBands);
        }
    }
        
    
    py::array applyPreconditioner(npDoub pyExForces,npDoub pyURHS){
        /*
        Apply preconditioner to obtain lambda and alpha. 
        See documentation for this method in fiberCollection.py
        */

        // allocate std::vector (to pass to the C++ function)
        vec ExForces(pyExForces.size());
        std::memcpy(ExForces.data(),pyExForces.data(),pyExForces.size()*sizeof(double));
        vec U_RHS(pyURHS.size());
        std::memcpy(U_RHS.data(),pyURHS.data(),pyURHS.size()*sizeof(double));
        
        int nAlphas = 3*_nXPerFib;
        if (_rigid){
            nAlphas = 6;
        }
        int nFibs = ExForces.size()/(3*_nXPerFib);   
        vec LambasAndAlphas(3*_nXPerFib*nFibs+nAlphas*nFibs);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < nFibs; iFib++){
            vec LocalExForce(3*_nXPerFib), LocalURHS(3*_nXPerFib);
            for (int i=0; i < 3*_nXPerFib; i++){
                LocalExForce[i] = ExForces[3*_nXPerFib*iFib+i];
                LocalURHS[i] = U_RHS[3*_nXPerFib*iFib+i];
            }
            int FullSize = 3*_nXPerFib;
            vec ForceN(FullSize,0.0);
            for (int i=0; i < FullSize; i++){
                ForceN[i]+=LocalExForce[i];
            }
            vec alphas(nAlphas), Lambda(3*_nXPerFib);
            _SaddlePointSolvers[iFib].SolveSaddlePoint(ForceN, LocalURHS, alphas,Lambda);
            
            for (int i=0; i < FullSize; i++){
                LambasAndAlphas[FullSize*iFib+i] =Lambda[i]; // lambda
            }
            for (int i=0; i < nAlphas; i++){
                LambasAndAlphas[FullSize*nFibs+nAlphas*iFib+i] =alphas[i]; // lambda
            }
        }
        // Return 1D numpy array
        return make1DPyArray(LambasAndAlphas);   
    }
    
    py::array KAlphaKTLambda(npDoub pyTangents,npDoub pyAlphas,npDoub pyLambdas){
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
        if (_rigid){
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
                LocalAlphas[i] = Alphas[systemSize*iFib+i]; 
            }
            vec LocKAlphas(3*_nXPerFib), LocKTLambda(systemSize);
            _SaddlePointSolvers[iFib].KAlpha(LocalAlphas,LocKAlphas);
            _SaddlePointSolvers[iFib].KTLambda(LocalLams,LocKTLambda);
            for (int i=0; i < 3*_nXPerFib; i++){
                KAlphaKTLambda[3*_nXPerFib*iFib+i] = LocKAlphas[i];
            }
            for (int i=0; i < systemSize; i++){
                KAlphaKTLambda[3*_nXPerFib*_NFib+iFib*systemSize+i] = LocKTLambda[i];
            }
        }
        return make1DPyArray(KAlphaKTLambda);
    }


    //=============================================
    // METHODS FOR SEMIFLEXIBLE FLUCTUATING FIBERS 
    //=============================================
                
    py::array MHalfAndMinusHalfEta(npDoub pyPoints, npDoub pyRandVec1, bool FPisLocal, bool FatCorrection){
        /*
        This method computes M[X]^(1/2)*W and M[X}^(-1/2)*W in the case when the mobility is given
        by local drag only. The boolean FPisLocal says whether to include the finite part
        mobility as part of that. 
        */

        // allocate std::vector (to pass to the C++ function)
        vec chebPoints(pyPoints.size());
        std::memcpy(chebPoints.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
        vec RandVec1(pyRandVec1.size());
        std::memcpy(RandVec1.data(),pyRandVec1.data(),pyRandVec1.size()*sizeof(double));
        int sysDim = 3*_nXPerFib;
        
        vec BothHalfAndMinusHalf(2*sysDim*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator, _FatMobilityEvaluator)
        for (int iFib = 0; iFib < _NFib; iFib++){
            //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
	        vec LocalPoints(sysDim);
            vec LocalRand1(sysDim);
            for (int i=0; i < sysDim; i++){
                LocalPoints[i] = chebPoints[sysDim*iFib+i];
                LocalRand1[i] = RandVec1[sysDim*iFib+i];
            }
            
            // M and Schur complement block B
            vec MWsym(sysDim*sysDim), EigVecs(sysDim*sysDim), EigVals(sysDim);
            _MobilityEvaluator.MobilityForceMatrix(LocalPoints, FPisLocal, -1,MWsym, EigVecs, EigVals);
            if (FatCorrection){
                vec MWsymFat(sysDim*sysDim), EigVecsFat(sysDim*sysDim), EigValsFat(sysDim);
                _FatMobilityEvaluator.MobilityForceMatrix(LocalPoints, FPisLocal, -1,MWsymFat, EigVecsFat, EigValsFat);
                // Compute the difference 
                for (uint i=0; i < sysDim*sysDim; i++){
                    MWsym[i]-=MWsymFat[i];
                }
                // Do the eigenvalue decomp
                SymmetrizeAndDecomposePositive(sysDim, MWsym, -100, EigVecs, EigVals); 
            }
            vec MWHalfetaLoc(sysDim);
            vec MWMinusHalfetaLoc(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 0.5, EigVecs, EigVals, LocalRand1, MWHalfetaLoc);
            ApplyMatrixPowerFromEigDecomp(sysDim,-0.5, EigVecs, EigVals, LocalRand1, MWMinusHalfetaLoc);
            for (int i=0; i < sysDim; i++){
                BothHalfAndMinusHalf[sysDim*iFib+i]= MWHalfetaLoc[i];
                BothHalfAndMinusHalf[(_NFib+iFib)*sysDim+i]= MWMinusHalfetaLoc[i];
            }
        }
        return make1DPyArray(BothHalfAndMinusHalf);
    }
    
    py::array TestPInv(npDoub Ap, npDoub bp, int m, int n){
        /*
        This is for de-bugging the solve with pseudo-inverse method
        in VectorMethods.cpp. 
        */

        // allocate std::vector (to pass to the C++ function)
        vec A(Ap.size());
        std::memcpy(A.data(),Ap.data(),Ap.size()*sizeof(double));
        vec b(bp.size());
        std::memcpy(b.data(),bp.data(),bp.size()*sizeof(double));
        int nb = b.size()/m;
        vec answer(n*nb);
        
        SolveWithPseudoInverse(m,n, A, b, answer,_svdtol, false,false, std::max(m,n));
        // Return 2D numpy array
        return make1DPyArray(answer);      
    }
    
    py::array InvertKAndStep(npDoub pyTangents, npDoub pyMidpoints, npDoub pyVelocities, double dt){
        /*
        Takke a "step" by inverting K. Specifically, we are computing
        K^-1*Velocity, and then evolving the fiber by taking a step of size dt
        */

        // allocate std::vector (to pass to the C++ function)
        vec Tangents(pyTangents.size());
        std::memcpy(Tangents.data(),pyTangents.data(),pyTangents.size()*sizeof(double));
        vec Midpoints(pyMidpoints.size());
        std::memcpy(Midpoints.data(),pyMidpoints.data(),pyMidpoints.size()*sizeof(double));
        vec Velocities(pyVelocities.size());
        std::memcpy(Velocities.data(),pyVelocities.data(),pyVelocities.size()*sizeof(double));
        int sysDim = 3*_nXPerFib;

        vec TausMidpointsAndXs(6*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr)
        for (int iFib = 0; iFib < _NFib; iFib++){
            //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
            vec LocalTangents(3*_nTauPerFib);
            vec Velocity(sysDim);
            for (int i=0; i < sysDim; i++){
                Velocity[i] = Velocities[sysDim*iFib+i];
            }
            for (int i=0; i< 3*_nTauPerFib; i++){
                LocalTangents[i] = Tangents[3*_nTauPerFib*iFib+i];
            }
            vec3 ThisMidpoint; 
            for (int d=0; d < 3; d++){
                ThisMidpoint[d] = Midpoints[3*iFib+d];
            }           
            // Compute rotation rates Omega_tilde
            vec OmegaTilde(sysDim);
            _SaddlePointSolvers[iFib].KInverseU(LocalTangents,Velocity,OmegaTilde);
            
            vec3 MidpointTilde;
            vec XTilde(sysDim);
            vec TauTilde(3*_nTauPerFib);
            OneFibRotationUpdate(LocalTangents,ThisMidpoint,OmegaTilde,TauTilde,MidpointTilde,XTilde,dt,_svdtol);
            
            for (int iPt=0; iPt < _nTauPerFib; iPt++){
                int Tauindex = iPt+iFib*_nTauPerFib;
                for (int d = 0; d< 3; d++){
                    TausMidpointsAndXs[3*Tauindex+d] = TauTilde[3*iPt+d];
                }
            }
            // Add the midpoint
            for (int d=0; d< 3; d++){
                TausMidpointsAndXs[3*_nTauPerFib*_NFib+3*iFib+d]=MidpointTilde[d];
            } 
            int startXs = 3*_nXPerFib*_NFib;
            for (int iPt=0; iPt < _nXPerFib; iPt++){
                int Xindex = iPt+iFib*_nXPerFib;
                for (int d=0; d < 3; d++){
                    TausMidpointsAndXs[startXs+3*Xindex+d] = XTilde[3*iPt+d];
                }
            }
        }
        
        // Return 2D numpy array
        return makePyDoubleArray(TausMidpointsAndXs);      
    }
    
    py::array ComputeDriftVelocity(npDoub pyPointsTilde, npDoub pyMHalfEta, npDoub pyMMinusHalfEta, 
        npDoub pyRandVecBE, double dt, bool ModifiedBE, bool FPisLocal){
        /*
        The purpose of this method is to return the velocity that goes on the RHS of the saddle 
        point system. This has three components to it:
        1) The Brownian velocity sqrt(2*kbT/dt)*M^(1/2)*W
        2) The extra term M*L^(1/2)*W that comes in for modified backward Euler
        3) The stochastic drift term. The drift term is done using a combination of M^(1/2)*W, M^(-1/2)*W, and 
        Mtilde. The relevant formulas is Eq. (47) in 
        https://arxiv.org/pdf/2301.11123.pdf
        Note that this method is only used when the hydro is intra-fiber.
        */
        // allocate std::vector (to pass to the C++ function)
        vec chebPointsTilde(pyPointsTilde.size());
        std::memcpy(chebPointsTilde.data(),pyPointsTilde.data(),pyPointsTilde.size()*sizeof(double));
        vec AllMHalfEta(pyMHalfEta.size());
        std::memcpy(AllMHalfEta.data(),pyMHalfEta.data(),pyMHalfEta.size()*sizeof(double));
        vec AllMMinusHalfEta(pyMHalfEta.size());
        std::memcpy(AllMMinusHalfEta.data(),pyMMinusHalfEta.data(),pyMMinusHalfEta.size()*sizeof(double));
        vec RandVecMBE(pyRandVecBE.size());
        std::memcpy(RandVecMBE.data(),pyRandVecBE.data(),pyRandVecBE.size()*sizeof(double));
        int sysDim = 3*_nXPerFib;

        vec U0Drift(3*_nXPerFib*_NFib);
        #pragma omp parallel for num_threads(_nOMPThr) firstprivate(_MobilityEvaluator)
        for (int iFib = 0; iFib < _NFib; iFib++){
            //std::cout << "Doing fiber " << iFib << " on thread " << omp_get_thread_num() << std::endl;
	        vec XTilde(sysDim);
            vec MWHalfEta(sysDim);
            vec MWMinusHalfeta(sysDim);
            vec LocalBERand(sysDim);
            vec LocalU0(sysDim);
            for (int i=0; i < sysDim; i++){
                XTilde[i] = chebPointsTilde[sysDim*iFib+i];
                MWHalfEta[i] = AllMHalfEta[sysDim*iFib+i];
                MWMinusHalfeta[i] = AllMMinusHalfEta[sysDim*iFib+i];
                LocalBERand[i] = RandVecMBE[sysDim*iFib+i];
            }
            
            // Compute tilde variables
            vec MWsymTilde(sysDim*sysDim), EigVecsTilde(sysDim*sysDim), EigValsTilde(sysDim);
            _MobilityEvaluator.MobilityForceMatrix(XTilde, FPisLocal,-1,MWsymTilde, EigVecsTilde, EigValsTilde);
            
            // RFD term
            vec URFDPlus(sysDim);
            vec URFDMinus(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, MWMinusHalfeta, URFDPlus); // Mtilde*U_B
            vec URFD(sysDim);
            for (int i=0; i < sysDim; i++){
                URFD[i] = sqrt(2.0*_kbT/dt)*(URFDPlus[i]-MWHalfEta[i]);
            }
            
            // Add the thermal flow and RFD to U0
            vec AddBERandomVel(sysDim);
            vec BendHalfEta(sysDim);
            if (ModifiedBE){
                BlasMatrixProduct(sysDim,sysDim,1,sqrt(dt/2),0.0,_BendForceMatHalf,false,LocalBERand,BendHalfEta); 
                ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, BendHalfEta, AddBERandomVel);          
            }    
            vec PenaltyTerm(sysDim);
            ApplyMatrixPowerFromEigDecomp(sysDim, 1.0, EigVecsTilde, EigValsTilde, _BendMatX0, PenaltyTerm);  
            for (int i=0; i < sysDim; i++){
                LocalU0[i]+=URFD[i]-PenaltyTerm[i]+sqrt(2.0*_kbT/dt)*AddBERandomVel[i]; // Drift + Backward Euler modification
            }
            for (int i=0; i < sysDim; i++){
                U0Drift[sysDim*iFib+i] = LocalU0[i]; 
            }
        }
        
        // Return 1D numpy array
        return make1DPyArray(U0Drift);      
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
        int numAlphas = 3*_nXPerFib;
        if (_rigid){
            numAlphas = 6;
        }
        #pragma omp parallel for num_threads(_nOMPThr) 
        for (int iFib=0; iFib < _NFib; iFib++){
            // Rotate tangent vectors
            vec LocalAlphas(numAlphas);
            for (int i=0; i < numAlphas; i++){
                LocalAlphas[i] = AllAlphas[numAlphas*iFib+i];
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
            OneFibRotationUpdate(LocalTangents, ThisMidpoint,LocalAlphas, RotatedTaus,NewMidpoint,NewX,dt,_svdtol);
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
    bool _rigid;
    SingleFiberMobilityEvaluator _MobilityEvaluator, _FatMobilityEvaluator;
    std::vector <SingleFiberSaddlePointSolver> _SaddlePointSolvers;
    // Variables for K and elastic force
    vec _D4BC, _D4BCForce,_BendForceMatHalf, _WTildeX, _WTildeXInv,_XFromTauMat;
    
    void OneFibRotationUpdate(const vec &Xsn, const vec3 &Midpoint, const vec &AllAlphas, vec &RotatedTaus,
        vec3 &NewMidpoint, vec &NewX,double dt, double nOmTol){
        // Rotate tangent vectors for a single fiber.
        // Uses Rodrigues rotation formula. 
        for (int iPt=0; iPt < _nTauPerFib; iPt++){
            vec3 XsToRotate, Omega;
            for (int d =0; d < 3; d++){
                XsToRotate[d] = Xsn[3*iPt+d];
                if (_rigid){
                    Omega[d] = AllAlphas[d];
                } else {
                    Omega[d] = AllAlphas[3*iPt+d];
                }
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
        int nOmega = _nTauPerFib;
        if (_rigid){
            nOmega = 1;
        }
        for (int d=0; d< 3; d++){
            NewMidpoint[d]=Midpoint[d]+dt*AllAlphas[3*nOmega+d];
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

PYBIND11_MODULE(FiberCollectionC, m) {
    py::class_<FiberCollectionC>(m, "FiberCollectionC")
        .def(py::init<int,int,int,int,bool,double,double,double,double,double,double,double,bool,bool,bool>())
        .def("initMatricesForPreconditioner", &FiberCollectionC::initMatricesForPreconditioner)
        .def("initMobilityMatrices", &FiberCollectionC::initMobilityMatrices)
        .def("SetEigValThreshold", &FiberCollectionC::SetEigValThreshold)
        .def("initFatMobilityEvaluator",&FiberCollectionC::initFatMobilityEvaluator)
        .def("initResamplingMatrices", &FiberCollectionC::initResamplingMatrices)
        .def("evalBendForces",&FiberCollectionC::evalBendForces)
        .def("evalBendStress",&FiberCollectionC::evalBendStress)
        .def("evalStressFromForce",&FiberCollectionC::evalStressFromForce)
        .def("evalDriftPartofStress",&FiberCollectionC::evalDriftPartofStress)
        .def("getUniformPoints", &FiberCollectionC::getUniformPoints)
        .def("getUpsampledPoints", &FiberCollectionC::getUpsampledPoints)
        .def("getUpsampledForces", &FiberCollectionC::getUpsampledForces)
        .def("getDownsampledVelocities", &FiberCollectionC::getDownsampledVelocities)
        .def("evalLocalVelocities",&FiberCollectionC::evalLocalVelocities)
        .def("SingleFiberRPYSum",&FiberCollectionC::SingleFiberRPYSum)
        .def("FinitePartVelocity",&FiberCollectionC::FinitePartVelocity)
        .def("FactorizePreconditioner", &FiberCollectionC::FactorizePreconditioner)
        .def("applyPreconditioner",&FiberCollectionC::applyPreconditioner)
        .def("KAlphaKTLambda",&FiberCollectionC::KAlphaKTLambda)
        .def("MHalfAndMinusHalfEta", &FiberCollectionC::MHalfAndMinusHalfEta)
        .def("InvertKAndStep", &FiberCollectionC::InvertKAndStep)
        .def("ComputeDriftVelocity", &FiberCollectionC::ComputeDriftVelocity)
        .def("getLHalfX", &FiberCollectionC::getLHalfX)
        .def("RodriguesRotations",&FiberCollectionC::RodriguesRotations)
        .def("ThermalTranslateAndDiffuse",&FiberCollectionC::ThermalTranslateAndDiffuse)
        .def("TestPInv",&FiberCollectionC::TestPInv);
}
