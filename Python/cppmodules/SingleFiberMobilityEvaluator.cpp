#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include "utils.h"
#include "RPYKernelEvaluator.cpp"

/**
    SingleFiberMobilityEvaluator.cpp
    The purpose of this class is to evaluate the mobility matrix
    M[X] on a single fiber. This mobility can be calculated in a number of ways
    based on the booleans RPYQuad, DirectRPY, and OversampledRPY in the constructor. 
    If all are false, it will do SBT. 
    If RPYQuad, then the method will do special quadrature for the RPY kernel. 
    If DirectRPY, then the method will do direct quadrature (just summing over the Nx points)
    If OversampledRPY, the method will do oversampled RPY quadrature. 
**/

namespace py=pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

class SingleFiberMobilityEvaluator {

    public: 
    
    //===========================================
    //        METHODS FOR INITIALIZATION
    //===========================================
    SingleFiberMobilityEvaluator(int Nx,double a, double L, double mu, bool RPYQuad, bool DirectRPY, bool OversampledRPY)
        :_RPYEvaluator(a,mu,Nx,{0,0,0}){
        _nXPerFib = Nx;
        SetQuadType(RPYQuad, DirectRPY, OversampledRPY);
        _a = a;
        _L = L;
        _mu = mu;
    }
    
    void SetQuadType( bool RPYQuad, bool DirectRPY, bool OversampledRPY){
        _exactRPY = RPYQuad;
        _directRPY = DirectRPY;
        _oversampledRPY = OversampledRPY;
        // Check that only one of the inputs is true
        bool goodInput = true;
        if (_exactRPY){
            if (_directRPY || _oversampledRPY){
                goodInput = false;
            }
        }
        if (_directRPY){
            if (_exactRPY || _oversampledRPY){
                goodInput = false;
            }
        }
        if (_oversampledRPY){
            if (_directRPY || _exactRPY){
                goodInput = false;
            }
        } 
        if (!goodInput){
            throw std::runtime_error("Mobility definition is ambiguous - can only specify one of exact RPY, oversampled RPY, or direct RPY");
        }
    }    
    
    void initMobilitySubmatrices(npDoub pyXNodes,npDoub pyRegXNodes,npDoub pyFPMatrix, npDoub pyDoubFPMatrix, 
        npDoub pyRL2aResampMatrix, npDoub pyRLess2aWts,npDoub pyXDiffMatrix, npDoub pyWTildeXInverse,
        npDoub pyOversampleWtsForDirectQuad,npDoub pyUpsamplingMatrix, double eigValThres){    

        // allocate std::vector (to pass to the C++ function)
        _eigValThres = eigValThres;
        _WTildeXInv = vec(pyWTildeXInverse.size());
        std::memcpy(_WTildeXInv.data(),pyWTildeXInverse.data(),pyWTildeXInverse.size()*sizeof(double));
        _XNodes = vec(_nXPerFib);
        std::memcpy(_XNodes.data(),pyXNodes.data(),pyXNodes.size()*sizeof(double));
        _RegNodes = vec(pyRegXNodes.size());
        std::memcpy(_RegNodes.data(),pyRegXNodes.data(),pyRegXNodes.size()*sizeof(double));
        
        _FinitePartMatrix = vec(pyFPMatrix.size());
        _XDiffMatrix = vec(pyXDiffMatrix.size());
        _OverSamplingWtsMatrix = vec(pyOversampleWtsForDirectQuad.size());
        _UpsamplingMatrix = vec(pyUpsamplingMatrix.size());
        std::memcpy(_FinitePartMatrix.data(),pyFPMatrix.data(),pyFPMatrix.size()*sizeof(double));
        std::memcpy(_XDiffMatrix.data(),pyXDiffMatrix.data(),pyXDiffMatrix.size()*sizeof(double));
        std::memcpy(_OverSamplingWtsMatrix.data(),pyOversampleWtsForDirectQuad.data(),pyOversampleWtsForDirectQuad.size()*sizeof(double));
        std::memcpy(_UpsamplingMatrix.data(),pyUpsamplingMatrix.data(),pyUpsamplingMatrix.size()*sizeof(double));
        _Nupsample = _OverSamplingWtsMatrix.size()/(9*_nXPerFib);
        
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
    
    void SetEigValThreshold(double Thres){
        _eigValThres = Thres;
    }
    
    double getEigThreshold(){
        return _eigValThres;
    }
    
    void MobilityForceMatrix(const vec &LocalPoints, bool nonLocalParts, vec &MForce, vec &EigVecs, vec &EigVals){
        /*
        Main method to compute the mobility M[X]
        This is the matrix that acts on FORCES to give velocities
        */
        int NBands = -1;
        if (_directRPY || _oversampledRPY){
            RPYDirectMobility(LocalPoints,MForce,NBands, _oversampledRPY);
            SymmetrizeAndDecomposePositive(3*_nXPerFib, MForce, 0, EigVecs, EigVals);
        } else { // Doing SBT or quadrature 
            vec M(9*_nXPerFib*_nXPerFib);
            MobilityMatrix(LocalPoints, true, nonLocalParts, M);  
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,M,false,_WTildeXInv,MForce);
            SymmetrizeAndDecomposePositive(3*_nXPerFib, MForce, _eigValThres, EigVecs, EigVals);
        }
    }
    
    void MobilityForceMatrix(const vec &LocalPoints, bool nonLocalParts, vec &MForce){
        /*
        Main method to compute the mobility M[X]
        This is the matrix that acts on FORCES to give velocities
        Same as the previous method, but it doesn't symmetrize the matrix (just gives you the raw one)
        */
        int NBands = -1;
        if (_directRPY || _oversampledRPY){
            RPYDirectMobility(LocalPoints,MForce,NBands,_oversampledRPY);
        } else { // Doing SBT or quadrature 
            vec M(9*_nXPerFib*_nXPerFib);
            MobilityMatrix(LocalPoints, true, nonLocalParts, M);  
            BlasMatrixProduct(3*_nXPerFib,3*_nXPerFib,3*_nXPerFib,1.0,0.0,M,false,_WTildeXInv,MForce);
        }
    }
    
    void OversampleMobilityForceMatrix(const vec &LocalPoints, vec &MForce){
        /*
        Main method to compute the mobility M[X]
        This is the matrix that acts on FORCES to give velocities
        Same as the previous method, but it doesn't symmetrize the matrix (just gives you the raw one)
        */
        int NBands = -1;
        RPYDirectMobility(LocalPoints,MForce,NBands,true);
    }
    
    
    void MobilityMatrix(const vec &LocalPoints, bool AddLocal, bool AddNonLocal, vec &M){
        /*
        This is the matrix that acts on force DENSITIES to give velocities. It's only
        included if we use SBT or special RPY quadrature
        */
        if (AddLocal){
            calcMLocal(LocalPoints, M);
        }
        if (AddNonLocal){
            addMFP(LocalPoints, M);
            if (_exactRPY){
                addMFPDoublet(LocalPoints,M);
                addNumericalRLess2a(LocalPoints,M);
            }
        }
    }  
    
    void UpsampledPoints(const vec &LocalPoints, vec &UpsampledPoints){
        int Npts = _nXPerFib;
        if (_directRPY){ 
            UpsampledPoints.resize(3*Npts);
            std::memcpy(UpsampledPoints.data(),LocalPoints.data(),LocalPoints.size()*sizeof(double));
        } else { 
            Npts = _Nupsample;
            UpsampledPoints.resize(3*Npts);
            BlasMatrixProduct(Npts, _nXPerFib, 3,1.0,0.0,_UpsamplingMatrix,false,LocalPoints,UpsampledPoints); 
        }
    }
    
    void UpsampledForces(const vec &LocalForces, vec &UpsampledForces){
        int Npts = _nXPerFib;
        if (_directRPY){ 
            UpsampledForces.resize(3*Npts);
            std::memcpy(UpsampledForces.data(),LocalForces.data(),LocalForces.size()*sizeof(double));
        } else { 
            Npts = _Nupsample;
            UpsampledForces.resize(3*Npts);
            // Multiply by W_up E_up W_tilde^-1
            BlasMatrixProduct(3*Npts, 3*_nXPerFib, 1,1.0,0.0,_OverSamplingWtsMatrix,false,LocalForces,UpsampledForces); 
        }
    }   
    
    void DownsampledVelocity(const vec &UpsampledVelocity, vec &NewVelocity){
        int Npts = _nXPerFib;
        if (_directRPY){ 
            NewVelocity.resize(3*Npts);
            std::memcpy(NewVelocity.data(),UpsampledVelocity.data(),UpsampledVelocity.size()*sizeof(double));
        } else { 
            Npts = _Nupsample;
            NewVelocity.resize(3*Npts);
            // Multiply by( W_up * E_up * WtildeInverse)^T
            BlasMatrixProduct(3*_nXPerFib,3*Npts,1,1.0,0.0,_OverSamplingWtsMatrix,true,UpsampledVelocity,NewVelocity); 
        }
    }
    
    void RPYVelocityFromForces(const vec &Points, const vec &Forces, vec &Velocities){
        int Npts = Points.size()/3;
        for (int iPt = 0; iPt < Npts; iPt++){
            for (int jPt = 0; jPt < Npts; jPt++){
                vec3 u;
                vec3 rvec;
                vec3 Force;
                for (int iD=0; iD < 3; iD++){
                    rvec[iD]=Points[3*iPt+iD]-Points[3*jPt+iD];
                    Force[iD]=Forces[3*jPt+iD];
                }
                _RPYEvaluator.RPYTot(rvec,Force,u);
                // Copy the pairwise matrix into the big matrix
                for (int iD=0; iD < 3; iD++){
                    Velocities[3*iPt+iD]+=u[iD];
                }
            }
        }
    }  
    
    int getNupsample(){
        if (_directRPY){
            return _nXPerFib;
        }
        return _Nupsample;
    }
 
    private:
    
    int _nXPerFib, _Nupsample;
    bool _exactRPY, _directRPY, _oversampledRPY;
    RPYKernelEvaluator _RPYEvaluator;
    // Variables for mobility
    vec _XNodes, _RegNodes, _FinitePartMatrix, _DbltFinitePartMatrix, _XDiffMatrix, _WTildeXInv;
    vec _RL2aResampMatrix, _RLess2aWts, _stackedXDiffMatrix, _OverSamplingWtsMatrix, _UpsamplingMatrix;
    vec _LocalStokeslet, _LocalDoublet, _LocalRLess2aI, _LocalRLess2aTau;
    int _HalfNForSmall;
    double _L, _a, _mu, _eigValThres;
    
    void initLocalDragCoeffsRPY(){
        _LocalStokeslet = vec(_nXPerFib);
        _LocalDoublet = vec(_nXPerFib);
        _LocalRLess2aI = vec(_nXPerFib);
        _LocalRLess2aTau = vec(_nXPerFib);
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            double s = _RegNodes[iPt]; // accounts for regularization
            _LocalStokeslet[iPt] = log(s*(_L-s)/(4.0*_a*_a));
            _LocalDoublet[iPt] = 1.0/(4.0*_a*_a)-1.0/(2.0*s*s)-1.0/(2.0*(_L-s)*(_L-s)); 
            if (s < 2*_a){ 
                _LocalStokeslet[iPt] = log((_L-s)/(2.0*_a));
                _LocalDoublet[iPt] = 1.0/(8.0*_a*_a)-1.0/(2.0*(_L-s)*(_L-s));
            } else if (s > _L-2*_a){
                _LocalStokeslet[iPt] = log(s/(2.0*_a));
                _LocalDoublet[iPt] = 1.0/(8.0*_a*_a)-1.0/(2.0*s*s);
            }
        }
    }
    
    void updateLocalDragCoeffsRPY(const vec &Tangents){
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            vec3 Tau;
            for (int id = 0; id < 3; id++){
                Tau[id] = Tangents[3*iPt+id];
            }
            double nXs = normalize(Tau);
            double s = _RegNodes[iPt]; // accounts for regularization
            if (s < 2*_a) {
                _LocalRLess2aI[iPt] = (128*_a*_a-36*_a*_a*nXs+64*_a*s-9*nXs*s*s)/(48.0*_a*_a);
                _LocalRLess2aTau[iPt] = nXs*(0.25+s*s/(16.0*_a*_a));
            } else if (s > _L-2*_a) {
                double sbar = _L-s;
                _LocalRLess2aI[iPt] = (128*_a*_a-36*_a*_a*nXs+64*_a*sbar-9*nXs*sbar*sbar)/(48.0*_a*_a);
                _LocalRLess2aTau[iPt] = nXs*(0.25+sbar*sbar/(16.0*_a*_a));
            } else {
                _LocalRLess2aI[iPt] = 1.0/6*(32.0-9*nXs);
                _LocalRLess2aTau[iPt] = nXs/2.0;
            }
        }
    }
    
    void calcMLocal(const vec &ChebPoints, vec &M){
        /*
        Calculates the local drag matrix M. 
        Inputs: Xs = fiber tangent vectors as a 3N array
        */
        double viscInv = 1.0/(8.0*M_PI*_mu);
        vec Tangents(3*_nXPerFib,0.0);
        BlasMatrixProduct(_nXPerFib, _nXPerFib, 3,1.0,0.0,_XDiffMatrix,false,ChebPoints,Tangents); 
        updateLocalDragCoeffsRPY(Tangents);
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
                    M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd] = viscInv*(_LocalStokeslet[iPt]*(deltaij+XsXs_ij)*normTauInv+
                        2*_a*_a/3.0*_LocalDoublet[iPt]*(deltaij-3*XsXs_ij)*pow(normTauInv,3)+
                        (_LocalRLess2aI[iPt]*deltaij+_LocalRLess2aTau[iPt]*XsXs_ij));
                }
            }
        }
    }
    
    void addMFP(const vec &ChebPoints,vec &M){
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
        /*
        Adds the parts of the RPY kernel that are done directly (R < 2a) doing
        Gauss-Legendre quad
        */
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
        updateLocalDragCoeffsRPY(Tangents);
        for (int iPt = 0; iPt < _nXPerFib; iPt++){
            vec3 Tau;
            for (int id = 0; id < 3; id++){
                Tau[id]=Tangents[3*iPt+id];
            }
            double normTau = normalize(Tau);
            for (int id = 0; id < 3; id++){
                for (int jd =0; jd < 3; jd++){
                    double deltaij = 0;
                    if (id==jd){
                        deltaij=1;
                    }
                    double XsXs_ij =Tau[id]*Tau[jd];
                    M[3*_nXPerFib*(3*iPt+id)+3*iPt+jd]-= viscInv*(_LocalRLess2aI[iPt]*deltaij+_LocalRLess2aTau[iPt]*XsXs_ij);
                }
            }
        }
    }
    
    void RPYDirectMobility(const vec &LocalPoints, vec &MForce, int NBands, bool oversample){
        /*
        This forms the matrix M when we use direct RPY. The flow of the method is the same, 
        but extra matrices have to be added when we do oversampling
        */
        int Npts = _nXPerFib;
        vec XForRPY(LocalPoints.begin(),LocalPoints.end());
        if (oversample){ 
            Npts = _Nupsample;
            XForRPY.resize(3*Npts);
            BlasMatrixProduct(Npts, _nXPerFib, 3,1.0,0.0,_UpsamplingMatrix,false,LocalPoints,XForRPY); 
        }
        vec MRPYDirect(9*Npts*Npts);
        for (int iPt = 0; iPt < Npts; iPt++){
            int startPt=0;
            int endPt=Npts-1;
            if (NBands > 0){
                startPt = std::max(iPt-NBands,0);
                endPt = std::min(iPt+NBands,Npts-1);
            }
            for (int jPt = startPt; jPt <= endPt; jPt++){
                vec MPair(9);
                vec3 rvec;
                for (int iD=0; iD < 3; iD++){
                    rvec[iD]=XForRPY[3*iPt+iD]-XForRPY[3*jPt+iD];
                }
                _RPYEvaluator.PairRPYMatrix(rvec,MPair);
                // Copy the pairwise matrix into the big matrix
                for (int iD=0; iD < 3; iD++){
                    for (int jD=0; jD < 3; jD++){
                        MRPYDirect[3*Npts*(3*iPt+iD)+3*jPt+jD]=MPair[3*iD+jD];
                        MRPYDirect[3*Npts*(3*jPt+jD)+3*iPt+iD]=MPair[3*iD+jD];
                    }
                }
            }
        }
        if (oversample){ // pre and post multiply by W_up * E_up * WtildeInverse
            vec Mover(9*Npts*_nXPerFib);
            BlasMatrixProduct(3*Npts, 3*Npts, 3*_nXPerFib,1.0,0.0,MRPYDirect,false,_OverSamplingWtsMatrix,Mover); 
            BlasMatrixProduct(3*_nXPerFib,3*Npts,3*_nXPerFib,1.0,0.0,_OverSamplingWtsMatrix,true,Mover,MForce); 
        } else {
            std::memcpy(MForce.data(),MRPYDirect.data(),MRPYDirect.size()*sizeof(double));
        }
    }  
                      
   
};
