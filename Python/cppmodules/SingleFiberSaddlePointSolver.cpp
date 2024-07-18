#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include "utils.h"
#include "SingleFiberMobilityEvaluator.cpp"

/**
    SingleFiberSaddlePointSolver.cpp
    The purpose of this class is to solve the saddle point system 
    on a single fiber. In addition to this, the object stores the matrices
    K, M (and its eigenvalue decomposition), and the pseudo-inverse of the Schur
    complement matrix. 
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class SingleFiberSaddlePointSolver {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    SingleFiberSaddlePointSolver(int Nx, int Ntau, bool rigid, double svdtol, double svdRigidTol){
        _nXPerFib = Nx;
        _nTauPerFib = Ntau;
        _rigid = rigid;
        _nAlphas = 3*_nXPerFib;
        if (_rigid){
            _nAlphas = 6;
        }
        _K = vec(_nAlphas*3*_nXPerFib);
        _KWithImpPart = vec(_nAlphas*3*_nXPerFib);
        _EigValsM = vec(3*_nXPerFib);
        _EigVecsM = vec(9*_nXPerFib*_nXPerFib);
        _SchurComplement = vec(_nAlphas*_nAlphas);
        _SchurComplementPInv = vec(_nAlphas*_nAlphas);
        
        _maxSVDModes = 2*_nTauPerFib+3;
        _normalizeSingVals = true;
        _SaddlePointTol = svdtol;
        _svdRigidTol = svdRigidTol;
        if (_rigid){
            _maxSVDModes = 6;
        }
    }
    
    void initMatricesForPreconditioner(npDoub pyD4BCForce,npDoub pyXFromTau, npDoub pyTauFromX,npDoub pyMP){

        _D4BCForce = vec(pyD4BCForce.size());
        _XFromTauMat = vec(pyXFromTau.size());
        _TauFromXMat = vec(pyTauFromX.size());
        _MidpointSamp = vec(pyMP.size());
        _Identity = vec(_nAlphas*_nAlphas);
        for (int i=0; i < _nAlphas; i++){
            _Identity[_nAlphas*i+i]=1;
        }    
         
        std::memcpy(_D4BCForce.data(),pyD4BCForce.data(),pyD4BCForce.size()*sizeof(double));
        std::memcpy(_XFromTauMat.data(),pyXFromTau.data(),pyXFromTau.size()*sizeof(double));
        std::memcpy(_TauFromXMat.data(),pyTauFromX.data(),pyTauFromX.size()*sizeof(double));
        std::memcpy(_MidpointSamp.data(),pyMP.data(),pyMP.size()*sizeof(double));
    }
    
    void FormSaddlePointMatrices(const vec &EigVecsM, const vec &EigValsM, const vec &LocalTangents, double impcodt){
        /*
        The purpose of this method is to compute K, the Schur Complement and its psuedo-inverse,
        and the mobility M. They are computed once, and then when we solve a saddle point system
        all we have to do is apply them
        */
        int FullSize = 3*_nXPerFib;
        if (_rigid){
            calcKRigid(LocalTangents,_K);
        } else {
            calcK(LocalTangents,_K);
        }
        for (int j=0; j < FullSize; j++){
            _EigValsM[j]=EigValsM[j];
            if (EigValsM[j] < 0){
                std::cout << "Neg eigenvalue in PC" << std::endl;
            }
        }
         for (int j=0; j < FullSize*FullSize; j++){
            _EigVecsM[j]=EigVecsM[j];
        }    
        // Overwrite K ->  K-impcoeff*dt*MW*D4BC*K;
        vec D4BCK(FullSize*_nAlphas);
        vec MtimesD4BCK(FullSize*_nAlphas);
        BlasMatrixProduct(FullSize,FullSize,_nAlphas,1.0,0.0,_D4BCForce,false,_K,D4BCK);
        ApplyMatrixPowerFromEigDecomp(FullSize, 1.0, _EigVecsM, _EigValsM, D4BCK, MtimesD4BCK); 
        for (int i = 0; i < FullSize*_nAlphas; i++){
            _KWithImpPart[i] = _K[i]-impcodt*MtimesD4BCK[i];
        }
        // Overwrite KWithImp-> M^(-1)*KWithImp
        vec MinvKWithImp(3*_nXPerFib*_nAlphas);
        ApplyMatrixPowerFromEigDecomp(FullSize, -1.0, _EigVecsM, _EigValsM, _KWithImpPart, MinvKWithImp); 
        BlasMatrixProduct(_nAlphas, FullSize,_nAlphas, 1.0,0.0,_K,true,MinvKWithImp,_SchurComplement); // K^T*(M^-1*B)
        bool half = false;
        // Pseudo-inverse against identity to form the pseudo-inverse matrix
        SolveWithPseudoInverse(_nAlphas, _nAlphas,_SchurComplement, _Identity, _SchurComplementPInv, 
            _SaddlePointTol,_normalizeSingVals,half,_maxSVDModes);
    }
        
    void SolveSaddlePoint(vec &ForceN, const vec &U0,vec &alphas, vec &Lambda){
        // Solve saddle point system 
        // M[X]*(F^n + Lambda) + U0 = (K[X]-impcodt*M[X]*D4BC*K[X]) alpha 
        // K^T Lambda = 0
        // Solve M^-1*U0, add to force first
        int FullSize = 3*_nXPerFib;
        vec NewLocalU0(FullSize);
        ApplyMatrixPowerFromEigDecomp(FullSize, -1.0, _EigVecsM, _EigValsM, U0, NewLocalU0);  
        for (int i=0; i < FullSize; i++){
            ForceN[i]+=NewLocalU0[i];
        }
        vec RHS(_nAlphas);
        BlasMatrixProduct(_nAlphas,FullSize,1.0,1.0,1.0,_K,true,ForceN,RHS);

        // Solve for alphas
        BlasMatrixProduct(_nAlphas, _nAlphas,1, 1.0,0.0,_SchurComplementPInv,false,RHS,alphas);
        // Back solve to get lambda
        vec Kalpha(FullSize);
        BlasMatrixProduct(FullSize,_nAlphas,1,1.0,0.0,_KWithImpPart,false, alphas, Kalpha);
        // Overwrite Kalpha -> lambda
        ApplyMatrixPowerFromEigDecomp(FullSize, -1.0, _EigVecsM, _EigValsM, Kalpha, Lambda);  
        for (int i=0; i < FullSize; i++){
            Lambda[i]-=ForceN[i];
        }
    }
    
    void RigidBodyDisplacements(const vec &LocalTangents, const vec &LocalPoints, const vec &RHS, 
        SingleFiberMobilityEvaluator MobEval, bool includeFP, double kbTdt, vec &alphas){
        /*
        These are the displacements N^(1/2)*W when the fibers are treated as rigid. 
        */
        // K and K*
        vec KRig(3*_nXPerFib*6);
        calcKRigid(LocalTangents,KRig);
        // M and Schur complement block B
        int sysDim = 3*_nXPerFib;
        vec MWsym(sysDim*sysDim), EigVecs(sysDim*sysDim), EigVals(sysDim);
        std::cout << "Rigid fiber displacements not really tested -- check" << std::endl;
        MobEval.MobilityForceMatrix(LocalPoints, includeFP, MWsym, EigVecs, EigVals);
        
        vec MinvK(sysDim*6);
        vec SchurComplement(6*6);
        ApplyMatrixPowerFromEigDecomp(sysDim, -1.0, EigVecs, EigVals, KRig, MinvK);  
        BlasMatrixProduct(6, sysDim,6, 1.0,0.0,KRig,true,MinvK,SchurComplement); // K'*(M^-1*K)

        vec SchurEigVecs(6*6), SchurEigVals(6);
        SymmetrizeAndDecomposePositive(6, SchurComplement,0.0, SchurEigVecs, SchurEigVals); 
        bool normalize = false;
        bool half = true;
        SolveWithPseudoInverse(6, 6,SchurComplement, RHS, alphas, _svdRigidTol,normalize,half,6); // N*LocalRand1
    }
    
    void KAlpha(const vec &alphas, vec &Kalphas){
        BlasMatrixProduct(3*_nXPerFib,_nAlphas,1,1.0, 0.0,_K,false,alphas,Kalphas);
    }
    
    void KTLambda(const vec &Lams, vec &KTLambdas){
        BlasMatrixProduct(_nAlphas,3*_nXPerFib,1,1.0, 0.0,_K,true,Lams,KTLambdas);
    }
    
    void KInverseU(const vec &LocalTangents, const vec &Velocity,vec &OmegaTilde){
        /*
        This method inverts K. When the fibers are semiflexible, there is a simple 
        formula (given in Eq. (21) of https://arxiv.org/pdf/2301.11123.pdf) for the 
        inverse of K. However, when the fibers are rigid, the pseudo-inverse K^dagger*K=I_6
        exists, but a simple formula for it does not (to my knowledge). So, we call the numerical
        pseudo-inverse method in this case
        */
        int sysDim = 3*_nXPerFib;
        if (_rigid){
            vec K(_nAlphas*sysDim);
            calcKRigid(LocalTangents,K);
            SolveWithPseudoInverse(sysDim,6, K, Velocity, OmegaTilde,_svdRigidTol, true,false, 6); 
        } else {
            vec KInverse(sysDim*sysDim);
            calcKInverse(LocalTangents, KInverse);
            BlasMatrixProduct(sysDim,sysDim,1,1.0, 0.0,KInverse,false,Velocity,OmegaTilde);
        }
    }
    
    void KInverseKProductForStress(const vec &LocalTangents, vec &Stress){
        int sysDim = 3*_nXPerFib;
        vec K(_nAlphas*sysDim);
        calcK(LocalTangents,K);
        vec KInverse(_nAlphas*sysDim);
        calcKInverse(LocalTangents, KInverse);
        for (int iBlock=0; iBlock < 3; iBlock++){
            for (int iRow=0; iRow < 3; iRow++){
                for (int iPt=0; iPt < _nTauPerFib; iPt++){
                    int pIndex = 3*iPt+iRow;
                    int colIndex = 3*iPt+iBlock;
                    int StressIndex = 3*iBlock+iRow;
                    // Inner product of the pIndex row of Kinverse and the 
                    // ColIndex column of K
                    for (int i=0; i < sysDim; i++){ 
                        Stress[StressIndex]+=KInverse[sysDim*pIndex+i]*K[sysDim*i+colIndex];
                    }
                }
            }
        }
    }  
    
 
    private:
    
    bool _rigid, _normalizeSingVals;
    int _nXPerFib, _nTauPerFib, _nAlphas, _maxSVDModes;
    double _SaddlePointTol, _svdRigidTol;
    vec _D4BCForce ,_XFromTauMat,  _TauFromXMat, _MidpointSamp, _Identity;
    vec _K, _KWithImpPart, _EigValsM, _EigVecsM, _SchurComplement, _SchurComplementPInv;
   
    
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
                     
   
};
