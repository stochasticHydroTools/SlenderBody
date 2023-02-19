import numpy as np
from DiscretizedFiber import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial, CellLinkedList
from FiberCollectionC import FiberCollectionC
import copy
import numba as nb
import scipy.sparse as sp
import time
from math import sqrt, exp, pi
from warnings import warn
import BatchedNBodyRPY as gpurpy

# Documentation last updated 01/21/2023

# Definitions (for special quadrature)
verbose = -1;               # debug / timings
svdTolerance = 1e-12;       # tolerance for pseudo-inverse
svdRigid = 0.25; # Exp[||Omega|| dt ] < svdRigid for rigid Brownian fibers

class fiberCollection(object):

    """
    This is a class that operates on a list of fibers together. 
    This python class is basically a wrapper for the C++ class
    FiberCollectionC.cpp, where every function is just calling the 
    corresponding (fast) C++ function. 
    
    At the end of this file, there is a child class for Semiflexible filaments
    that also perform bending fluctuations. That class reimplements some of the 
    methods
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads=1,rigidFibs=False,dt=1e-3):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, turnover time = mean time for each fiber to turnover,
        fibDisc = discretization object that each fiber will get a copy of, 
        nonLocal = are we doing nonlocal hydro? (0 for local drag, 1 for full hydro, 4 for intra-fiber hydro only)
        mu = fluid viscosity
        omega = frequency of background flow oscillations, 
        gam0 = base strain rate of the background flow, 
        Dom = Domain object to initialize the SpatialDatabase objects, 
        kbT = thermal energy, 
        eigValThres = eigenvalue threshold for intra-fiber mobility, 
        nThreads = number of OMP threads for parallel calculations, 
        rigidFibs = whether the fibers are rigid
        """
        self._Nfib = nFibs;
        self._fiberDisc = fibDisc;
        self._NXpf = self._fiberDisc._Nx;
        self._NTaupf = self._fiberDisc._Ntau;
        self._Ndirect = self._fiberDisc._nptsDirect;
        self._FPLoc = self._fiberDisc._FinitePartLocal;
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._nThreads = nThreads;
        self._deathrate = 1/turnovertime; # 1/s
        self.initPointForceVelocityArrays(Dom);
        self._kbT = kbT;
        self._rigid = rigidFibs;
        
        DLens = Dom.getPeriodicLens();
        for iD in range(len(DLens)):
            if (DLens[iD] is None):
                DLens[iD] = 1e99;
        
        # Initialize C++ class 
        if (self._kbT > 0):
            svdRigidTol = self._kbT*dt/(svdRigid*svdRigid);
        else:
            svdRigidTol = svdTolerance;
        print('Rigid tol is %1.2E' %svdRigidTol)
        self._FibColCpp = FiberCollectionC(nFibs,self._NXpf,self._NTaupf,nThreads,self._rigid,\
            self._fiberDisc._a,self._fiberDisc._L,self._mu,self._kbT,svdTolerance,svdRigidTol,self._fiberDisc._RPYSpecialQuad,\
            self._fiberDisc._RPYDirectQuad,self._fiberDisc._RPYOversample);
        self._FibColCpp.initMatricesForPreconditioner(self._fiberDisc._D4BCForce,  \
            self._fiberDisc._D4BCForceHalf,self._fiberDisc._XonNp1MatNoMP,self._fiberDisc._XsFromX,\
            self._fiberDisc._MidpointMat,self._fiberDisc._BendMatX0);
        self._FibColCpp.initResamplingMatrices(self._fiberDisc._nptsUniform,self._fiberDisc._MatfromNtoUniform);
        self._FibColCpp.initMobilityMatrices(self._fiberDisc._sX, fibDisc._leadordercs,\
            self._fiberDisc._FPMatrix.T,self._fiberDisc._DoubletFPMatrix.T,\
            self._fiberDisc._RLess2aResamplingMat,self._fiberDisc._RLess2aWts,\
            self._fiberDisc._DXGrid,self._fiberDisc._stackWTilde_Nx, self._fiberDisc._stackWTildeInverse_Nx,\
            self._fiberDisc._OversamplingWtsMat,self._fiberDisc._EUpsample,eigValThres);

    
    def initFibList(self,fibListIn, Dom,XFileName=None):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
        Inputs: list of Discretized fiber objects (typically empty), Domain object,
        file names for the points if we are initializing from file
        The file names can be either file names or the actual point locations.
        """
        self._fibList = fibListIn;
        self._DomLens = Dom.getLens();
        if (XFileName is None): # initialize straight fibers
            for jFib in range(self._Nfib):
                if self._fibList[jFib] is None:
                    # Initialize memory
                    self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                    # Initialize straight fiber positions at t=0
                    self._fibList[jFib].initFib(Dom.getLens());
        else:
            AllX = np.loadtxt(XFileName)
            SizeX = AllX.shape;
            if (SizeX[0]> self._Nfib*self._NXpf):
                warn('The input file is longer than expected - will use last configuration')
                AllX = AllX[SizeX[0]-self._Nfib*self._NXpf:];
            for jFib in range(self._Nfib): 
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                Xfib = np.reshape(AllX[jFib*self._NXpf:(jFib+1)*self._NXpf,:],(3*self._NXpf,1));
                XsThis, XMPThis = self._fiberDisc.calcXsAndMPFromX(Xfib);
                self._fibList[jFib].initFib(Dom.getLens(),Xs=XsThis,XMP=XMPThis);   
        self.fillPointArrays()
    
    def initPointForceVelocityArrays(self, Dom):
        """
        Method to initialize the memory for lists of points, tangent 
        vectors and lambdas for the fiber collection
        Input: Dom = Domain object 
        """
        # Establish large lists that will be used for the non-local computations
        self._totnumX = self._Nfib*self._NXpf; 
        self._totnumTau = self._Nfib*self._NTaupf; 
        self._totnumDirect = self._Nfib*self._Ndirect;                      
        self._ptsCheb=np.zeros((self._totnumX,3));                     
        self._tanvecs=np.zeros((self._totnumTau,3));
        self._Midpoints = np.zeros((self._Nfib,3));                     
        self._lambdas=np.zeros((self._totnumX*3));   
        self._alphas=np.zeros((self._totnumX*3));                  
        # Initialize the spatial database objects
        self._SpatialCheb = CellLinkedList(self._ptsCheb,Dom,nThr=self._nThreads);
        self._SpatialUni = CellLinkedList(np.zeros((self._fiberDisc._nptsUniform*self._Nfib,3)),Dom,nThr=self._nThreads);
        self._SpatialDirectQuad = CellLinkedList(np.zeros((self._totnumDirect,3)),Dom,nThr=self._nThreads);

    def fillPointArrays(self):
        """
        Copy the X and Xs arguments from self._fibList (list of fiber
        objects) into large (tot#pts x 3) arrays that are stored in memory
        """
        for iFib in range(self._Nfib):
            fib = self._fibList[iFib];
            self._ptsCheb[self._NXpf*iFib:self._NXpf*(iFib+1),:] = np.reshape(fib._X.copy(),(self._NXpf,3));
            self._tanvecs[self._NTaupf*iFib:self._NTaupf*(iFib+1),:] = np.reshape(fib._Xs.copy(),(self._NTaupf,3));
            self._Midpoints[iFib,:]=np.reshape(fib._XMP.copy(),3);
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (k=='_ptsCheb' or k=='_tanvecs' or k=='_lambdas' or k=='_alphas' or k=='_velocities'):
                setattr(result, k, copy.deepcopy(v, memo))
        return result 
       
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self,X,t,exForce,Lamstar,Dom,RPYEval):
        """
        RHS for the block diagonal GMRES system. 
        Inputs: X = Chebyshev point locations,t = system time, 
        exForce = external force (treated explicitly), 
        Lamstar = Lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, 
        Return the nonlocal velocities and forces for the saddle point system. Specifically, if the saddle 
        point solve can be written as  
        B*alpha = M*F+UEx, 
        return F and UEx. Here UEx includes the nonlocal velocity.
        """
        # Compute elastic forces at X
        BendForce = self.evalBendForce(X);
        TotalForce = BendForce+exForce;
        U0 = self.evalU0(X,t);
        UNonLoc = self.nonLocalVelocity(X,Lamstar+exForce+BendForce, Dom, RPYEval);
        return TotalForce, (UNonLoc+U0);

    def calcResidualVelocity(self,BlockDiagAnswer,X,Xs,dtimpco,Lamstar, Dom, RPYEval):
        """
        New RHS (after block diag solve). This is the residual form of the system that gets 
        passed to GMRES (if doing deterministic fibers). 
        Inputs: BlockDiagAnswer = the answer for lambda and alpha from the block diagonal solver,
        dtimpco = delta t * implicit coefficient,
        lamstar = lambdas for the nonlocal calculation, 
        Dom = Domain object, 
        RPYEval = RPY velocity evaluator for the nonlocal terms.
        The RHS for the residual system is  
        [M^NonLocal*(F*(X^n+dt*impco*K*alpha-X^(n+1/2,*)+Lambda - Lambda^*)); 0], 
        where by Lambda and alpha we mean the results from the block diagonal solver 
        """
        Lambdas = BlockDiagAnswer[:3*self._Nfib*self._NXpf];
        alphas = BlockDiagAnswer[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas);
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        FbendCorrection = self.evalBendForce(self._ptsCheb- X+np.reshape(dtimpco*Kalph,(self._Nfib*self._NXpf,3)));
        LamCorrection = Lambdas-Lamstar;
        TotalForce = FbendCorrection+LamCorrection;
        nonLocalVel = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval);
        if (self._rigid):
            Block2Dim = 6*self._Nfib;
        else:
            Block2Dim = 3*self._NXpf*self._Nfib;
        return np.concatenate((nonLocalVel,np.zeros(Block2Dim)));
        
    def CheckResiduals(self,LamAlph,impcodt,X,Xs,Dom,RPYEval,t,UAdd=0,ExForces=0):
        """
        This is just for debugging. It verifies that we solved the system we think we did. 
        """
        LamAlph = np.reshape(LamAlph,len(LamAlph))
        Lambdas = LamAlph[:3*self._Nfib*self._NXpf];
        alphas = LamAlph[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas); 
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        KTLam = KalphKTLam[3*self._Nfib*self._NXpf:];  
        XnLForBend = np.reshape(impcodt*Kalph,(self._Nfib*self._NXpf,3)); # X+dt*K*alpha
        # The bending force from time n is included in ExForces 
        TotalForce = self.evalBendForce(XnLForBend)+Lambdas+ExForces;   
        U0 = self.evalU0(self._ptsCheb,t);
        Local = self.LocalVelocity(X,TotalForce);
        nonLocal = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval);
        res = (Local+nonLocal+U0+UAdd)-Kalph;
        return res;
        
    def Mobility(self,LamAlph,impcodt,X,Xs,Dom,RPYEval):
        """
        Mobility calculation for GMRES
        Inputs: lamalph = the input lambdas and alphas impcodt = delta t * implicit coefficient,
        X = Npts * 3 array of Chebyshev point locations, Xs = Npts * 3 array of 
        tangent vectors,Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms,
        The calculation is 
        [-(M^Local+M^NonLocal)*(impco*dt*F*K*alpha +Lambda); K^T*Lambda ]
        """
        #timeMe = time.time();
        LamAlph = np.reshape(LamAlph,len(LamAlph))
        Lambdas = LamAlph[:3*self._Nfib*self._NXpf];
        alphas = LamAlph[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas);
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        KTLam = KalphKTLam[3*self._Nfib*self._NXpf:];
        XnLForBend = np.reshape(impcodt*Kalph,(self._Nfib*self._NXpf,3)); # dt/2*K*alpha       
        TotalForce = self.evalBendForce(XnLForBend)+Lambdas;
        Local = self.LocalVelocity(X,TotalForce);
        nonLocal = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval);
        FirstBlock = -(Local+nonLocal)+Kalph; # zero if lambda, alpha = 0
        #print('Mobility time %f' %(time.time()-timeMe));
        return np.concatenate((FirstBlock,KTLam));
        
    def FactorizePreconditioner(self,X_nonLoc,Xs_nonLoc,implic_coeff,dt,NBands=-1):
        """
        Block diagonal solver. This solver takes inputs Xs_nonLoc (tangent vectors),, 
        dt (time step size), implic_coeff (implicit coefficient for temporal integrator), 
        X_nonLoc (locations of points), and solves
        the saddle point system. 
        (K-impco*dt*M^L*F*K)*alpha = M^L*ExForces + Lambda)+UNonLoc
        K^T*Lambda = 0
        for Lambda, and alpha. B is K with the implicit part included This corresponds to the matrix solve
        [-M^L K-impco*dt*M^L*F*K; K^T 0]*[Lambda; alpha]=[M^L*ExForces; 0]
        where M^L is the local mobility on each fiber separately (the argument NBands can 
        be used to make the mobility banded). The purpose of this method specifically is to form
        the Schur complement matrix and store its pseudo-inverse so it can be applied in the next method. 
        """
        self._FibColCpp.FactorizePreconditioner(X_nonLoc,Xs_nonLoc,implic_coeff,dt,self._FPLoc,NBands);
       
    def BlockDiagPrecond(self,UNonLoc,ExForces):
        """
        Solve the saddle point system. 
        (K-impco*dt*M^L*F*K)*alpha = M^L*ExForces + Lambda)+UNonLoc
        K^T*Lambda = 0 
        using the precomputed matrices in the previous method.
        """
        UNonLoc = UNonLoc[:3*self._Nfib*self._NXpf];
        LamAlph= self._FibColCpp.applyPreconditioner(ExForces,UNonLoc);
        return LamAlph;
    
    def BrownianUpdate(self,dt,Randoms):
        """
        Random rotational and translational diffusion (assuming a rigid body) over time step dt. 
        This function calls the corresponding python function, which computes 
        alpha = sqrt(2*kbT/dt)*N_rig^(1/2)*randn,
        where N_rig = pinv(K_r^T M^(-1)*K_r)
        Then rotates and translates by alpha*dt = (Omega*dt, U*dt)
        """
        RandAlphas = self._FibColCpp.ThermalTranslateAndDiffuse(self._ptsCheb,self._tanvecs,Randoms, self._FPLoc,dt);
        CppXsMPX = self._FibColCpp.RodriguesRotations(self._tanvecs,self._Midpoints,RandAlphas,dt);   
        self._tanvecs = CppXsMPX[:self._Nfib*self._NTaupf,:];
        self._Midpoints = CppXsMPX[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        self._ptsCheb = CppXsMPX[self._Nfib*self._NXpf:,:];
        
    def nonLocalVelocity(self,X,forces, Dom, RPYEval,subSelf=True):
        """
        Compute the non-local velocity due to the fibers.
        Inputs: Arguments X = fiber positions as Npts x 3 arrays, 
        forces = forces as an 3*npts 1D numpy array
        Dom = Domain object the computation happens on,  
        RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities, 
        subSelf = whether we subtract the self terms. In some instances (e.g., when the
        self mobility is defined with oversampled RPY), the nonlocal mobility includes
        the self mobility and as such we don't need to subtract it. 
        Outputs: nonlocal velocity as a tot#ofpts*3 one-dimensional array.
        """
        # If not doing nonlocal hydro, return nothing
        doFinitePart = (not self._FPLoc);
        if (self._nonLocal==0):
            return np.zeros(3*self._Nfib*self._NXpf);
        elif (self._nonLocal==2 or self._fiberDisc._RPYDirectQuad or self._fiberDisc._RPYOversample):
            doFinitePart=0;
            
        # Calculate finite part velocity
        thist=time.time();
        finitePart = self._FibColCpp.FinitePartVelocity(X, forces);
        if (verbose>=0):
            print('Time to eval finite part %f' %(time.time()-thist));
            thist=time.time();
        if (self._nonLocal==4):
            return doFinitePart*finitePart;
        
        # Add inter-fiber velocity
        Xupsampled = self.getPointsForUpsampledQuad(X);
        Fupsampled = self.getForcesForUpsampledQuad(forces);
        if (verbose>=0):
            print('Upsampling time %f' %(time.time()-thist));
            thist=time.time();
        self._SpatialDirectQuad.updateSpatialStructures(Xupsampled,Dom);
        RPYVelocityUp = RPYEval.calcBlobTotalVel(Xupsampled,Fupsampled,Dom,self._SpatialDirectQuad,self._nThreads);
        if (verbose>=0):
            print('Upsampled Ewald time %f' %(time.time()-thist));
            thist=time.time();
        SelfTerms = self.SubtractSelfTerms(self._Ndirect,Xupsampled,Fupsampled,RPYEval.NeedsGPU());
        if (verbose>=0):
            print('Self time %f' %(time.time()-thist));
            thist=time.time();
        if (subSelf):
            RPYVelocityUp -= SelfTerms;
        RPYVelocity = self.getDownsampledVelocity(RPYVelocityUp);
        if (verbose>=0):
            print('Downsampling time %f' %(time.time()-thist));
        if (np.any(np.isnan(RPYVelocity))):
            raise ValueError('Velocity is nan - stability issue!') 
        return np.reshape(RPYVelocity,self._Nfib*self._NXpf*3)+doFinitePart*finitePart;
         
    def updateLambdaAlpha(self,lamalph):
        """
        Update the lambda and alphas after the solve is complete. 
        Xsarg = legacy argument. To remove. 
        """
        self._lambdas = lamalph[:3*self._Nfib*self._NXpf];
        self._alphas = lamalph[3*self._Nfib*self._NXpf:];
           
    def updateAllFibers(self,dt):
        """
        Update the fiber configurations, assuming self._alphas has been computed above. 
        Inputs: dt = timestep
        """
        CppXsMPX = self._FibColCpp.RodriguesRotations(self._tanvecs,self._Midpoints,self._alphas,dt);   
        self._tanvecs = CppXsMPX[:self._Nfib*self._NTaupf,:];
        self._Midpoints = CppXsMPX[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        self._ptsCheb = CppXsMPX[self._Nfib*self._NXpf:,:];
        return np.amax(self._ptsCheb);
           
    def getX(self):
        return self._ptsCheb;

    def getXs(self):
        return self._tanvecs;
    
    def getLambdas(self):
        return self._lambdas; 
    
    def FiberStress(self,XBend,XLam,Tau_n,Volume):
        """
        Compute fiber stress.
        """
        BendStress = 1/Volume*self._FibColCpp.evalBendStress(XBend);
        LamStress = 1/Volume*self._FibColCpp.evalStressFromForce(XLam,self._lambdas)
        #np.savetxt('LambdasN.txt',self._lambdas)
        #np.savetxt('BendStress.txt',BendStress)
        #np.savetxt('LamStress.txt',LamStress)
        return BendStress, LamStress;
              
    def uniformForce(self,strengths):
        """
        A uniform force density on all fibers with strength strength 
        """
        return np.tile(strengths,self._Nfib*self._NXpf);
    
    def getUniformPoints(self,chebpts):
        """
        Obtain uniform points from a set of Chebyshev points. 
        Inputs: chebpts as a (tot#ofpts x 3) array
        Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
        """
        return self._FibColCpp.getUniformPoints(chebpts);
        
    def getPointsForUpsampledQuad(self,chebpts):
        """
        Obtain upsampled points from a set of Chebyshev points. 
        Inputs: chebpts as a (tot#ofpts x 3) array
        Outputs: upsampled points as a (nPtsUpsample*Nfib x 3) array.
        """
        return self._FibColCpp.getUpsampledPoints(chebpts);
        
    def getForcesForUpsampledQuad(self,chebforces):
        """
        Obtain upsampled forces from a set of Chebyshev points. 
        Inputs: chebforces as a (tot#ofpts x 3) array
        Outputs: upsampled forces as a (nPtsUpsample*Nfib x 3) array.
        """
        return self._FibColCpp.getUpsampledForces(chebforces);
        
    def getDownsampledVelocity(self,upsampledVel):
        """
        Obtain velocity downsampled on the Chebyshev grid from the 
        upsampled velocity
        """
        return self._FibColCpp.getDownsampledVelocities(upsampledVel);
        
    def getg(self,t):
        """
        Get the value of the strain g according to what the background flow dictates.
        Input: time t
        Output: non-dimensional strain g
        """
        if (self._omega > 0):
            g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        else:
            g = self._gam0*t;
        return g;
    
    def getUniformSpatialData(self):
        return self._SpatialUni;
    
    def getFiberDisc(self):
        return self._fiberDisc;
    
    def getSysDimension(self):
        if (self._rigid):
            return 3*self._Nfib*self._NXpf+6*self._Nfib;
        return 6*self._Nfib*self._NXpf;
    
    def getBlockOneSize(self):
        return 3*self._Nfib*self._NXpf;    
    
    def averageTangentVectors(self):
        """
        nFib array, where each entry is the average 1/L*integral_0^L X_s(s) ds
        for each fiber
        """
        avgfibTans = np.zeros((self._Nfib,3));
        wts = np.reshape(self._fiberDisc._w,(self._Npf,1));
        for iFib in range(self._Nfib):
            fiberTans = self._tanvecs[self.getRowInds(iFib),:];
            avgfibTans[iFib,:]=np.sum(wts*fiberTans,axis=0)/self._Lf;
        return avgfibTans;
    
    def initializeLambdaForNewFibers(self,newfibs,ExForces,t,dt,implic_coeff,other=None):
        """
        Solve local problem to initialize values of lambda on the fibers
        Inputs: newfibs = list of replaced fiber indicies, 
        exForceDen = external (gravity/CL) force density on those fibers, 
        t = system time, dt = time step, implic_coeff = implicit coefficient (usually 1/2), 
        other = the other (previous time step) fiber collection
        """
        for iFib in newfibs:
            Xinds = self.getXInds(iFib);
            stackinds = self.getStackInds(iFib);
            X = np.reshape(self._ptsCheb[Xinds,:],3*self._NXpf);
            Xs = np.reshape(self._tanvecs[self.getTauInds(iFib),:],3*self._NTaupf);
            FBend = self.evalBendForce(X);
            FEx = ExForces[stackinds];
            FTotal = FBend+FEx;
            U0 = self.evalU0(self._ptsCheb[Xinds,:],t);
            self._FibColCpp.FactorizePreconditioner(X,Xs,implic_coeff,dt,self._FPLoc,-1);
            lamalph = self._FibColCpp.applyPreconditioner(FTotal,U0);
            self._lambdas[stackinds] = lamalph[:3*self._NXpf];
            if other is not None:
                other._lambdas[stackinds] = lamalph[:3*self._NXpf];    
    
    def FiberBirthAndDeath(self,tstep,other=None):
        """
        Turnover filaments. tstep = time step,
        other = the previous time step fiber collection (for Crank-Nicolson)
        """
        if (tstep==0):
            return [];
        rateDeath = self._deathrate*self._Nfib;
        systime = -np.log(1-np.random.rand())/rateDeath;
        bornFibs=[];
        while (systime < tstep):
            iFib = int(np.random.rand()*self._Nfib); # randomly choose one
            print('Turning over fiber in position %d ' %iFib);
            bornFibs.append(iFib);
            self._fibList[iFib] = DiscretizedFiber(self._fiberDisc);
            # Initialize straight fiber positions at t=0
            self._fibList[iFib].initFib(self._DomLens);
            Xinds = self.getXInds(iFib);
            Tauinds = self.getTauInds(iFib);
            stackinds = self.getStackInds(iFib);
            X, Xs, XMP = self._fibList[iFib].getPositions();
            self._ptsCheb[Xinds,:] = X;
            self._Midpoints[iFib,:] = XMP;
            self._tanvecs[Tauinds,:] = Xs;
            self._lambdas[stackinds] = 0;
            if (other is not None): # if previous fibCollection is supplied reset
                other._ptsCheb[rowinds,:] = X;
                other._Midpoints[iFib,:] = XMP;
                other._tanvecs[Tauinds,:] = Xs;
                other._lambdas[stackinds] = 0;
            # Update time of turnover
            systime += -np.log(1-np.random.rand())/rateDeath;
        return list(set(bornFibs));  
    
    def writeFiberLocations(self,FileName,wora='a'):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        f=open(FileName,wora)
        np.savetxt(f, self._ptsCheb);#, fmt='%1.13f')
        f.close()
        
    ## ====================================================
    ##  "PRIVATE" METHODS NOT ACCESSED OUTSIDE OF THIS CLASS
    ## ====================================================
    def SubtractSelfTerms(self,Ndir,Xupsampled,fupsampled,useGPU=False):
        """
        Subtract the self RPY integrals on each fiber
        Inputs: Ndir = number of points for direct quad,
        Xupsampled = upsampled positions, fupsampled = upsampled forces
        Return the direct RPY velocities
        """
        if (not useGPU):
            U1=self._FibColCpp.SingleFiberRPYSum(Ndir,Xupsampled,fupsampled);
            return np.reshape(U1,(self._Nfib*Ndir,3));
        
        # GPU Version
        hydrodynamicRadius = self._fiberDisc._a;
        selfMobility= 1.0/(6*pi*self._mu*hydrodynamicRadius);
        
        precision = np.float64;
        fupsampledg = np.array(fupsampled,precision);
        Xupsampledg = np.array(Xupsampled,precision);
        
        #It is really important that the result array has the same floating precision as the compiled module, otherwise
        # python will just silently pass by copy and the results will be lost
        MF=np.zeros(self._Nfib*3*Ndir, precision);
        gpurpy.computeMdot(np.reshape(Xupsampledg,self._Nfib*3*Ndir), np.reshape(fupsampledg,self._Nfib*3*Ndir), MF,
                       self._Nfib, Ndir,selfMobility, hydrodynamicRadius)
        if (np.amax(np.abs(MF))==0 and np.amax(np.abs(fupsampledg)) > 0):
        	raise ValueError('You are getting zero velocity with finite force, your UAMMD precision is wrong!') 
        return np.reshape(MF,(self._Nfib*Ndir,3));
        
    def LocalVelocity(self,X,Forces):
        """
        Compute the LOCAL velocity on each fiber. 
        Inputs: X = positions as a (tot#ofpts*3) 1D array
        forceDsAll = the force densities respectively on the fibers as a 1D array
        Outputs: the velocity M_loc*f on all fibers as a 1D array
        """
        return self._FibColCpp.evalLocalVelocities(X,Forces,self._FPLoc);
           
    def evalU0(self,Xin,t):
        """
        Compute the background flow on input array Xin at time t.
        Xin is assumed to be an N x 3 array with each column being
        the x, y, and z locations. 
        This method returns the background flow as a 3N 1d numpy array.
        """
        U0 = np.zeros(Xin.shape);
        U0[:,0]=self._gam0*np.cos(self._omega*t)*Xin[:,1];
        return np.reshape(U0,3*len(Xin[:,0]));
    
    def evalBendForce(self,X_nonLoc):
        """
        Evaluate the bending FORCES
        Inputs: X to evaluate at
        Outputs: the forces -Wtilde*EX_ssss also as a (tot#ofpts) x 3 array
        """
        return self._FibColCpp.evalBendForces(X_nonLoc);
        
    def calcCurvatures(self,X):
        """
        Evaluate fiber curvatures on fibers with locations X. 
        Returns an Nfib array of the mean L^2 curvature by fiber 
        """
        Curvatures=np.zeros(self._Nfib);
        for iFib in range(self._Nfib):
            Curvatures[iFib] = self._fiberDisc.calcFibCurvature(X[iFib*self._NXpf:(iFib+1)*self._NXpf,:]);
        return Curvatures;

    def getXInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 2D arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._NXpf,(iFib+1)*self._NXpf);
    
    def getTauInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 2D arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._NTaupf,(iFib+1)*self._NTaupf);
        
    def getStackInds(self,iFib):
        """
        Method to get the row indices in any (Nfib*Nperfib*3) long 1D arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*3*self._NXpf,(iFib+1)*3*self._NXpf);
        
class SemiflexiblefiberCollection(fiberCollection):

    """
    This class is a child of fiberCollection which implements BENDING FLUCTUATIONS. 
    There are some additional methods required in this case. 
    """

    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, kbT,eigValThres,nThreads=1,rigidFibs=False,dt=1e-3):
        super().__init__(nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads,rigidFibs,dt);
        self._iT = 0;
        SPDMatrix = self._fiberDisc._RPYDirectQuad or self._fiberDisc._RPYOversample;
        if (self._nonLocal==1 and not SPDMatrix):
            raise TypeError('If doing nonlocal hydro with fluctuations, can only do direct quadrature or oversampled')
        
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,t,ExForces,lamstar,Dom,RPYEval=None):
        """
        This method overwrites the one in the abstract class because the nonlocal hydro is handled only
        through GMRES. It is not like in the deterministic case, where first we do a "guess" for the 
        nonlocal hydro. 
        """
        self._iT+=1;
        BendForce = self.evalBendForce(X_nonLoc);
        U0 = self.evalU0(X_nonLoc,t);
        TotalForce = BendForce+ExForces;
        return TotalForce, U0;
        
    def BrownianUpdate(self,dt,Randoms):
        # Do nothing for semiflexible filaments
        warn('The Brownian update is implemented as part of the solve, not a separate split step!') 
       
    def MHalfAndMinusHalfEta(self,X,Ewald,Dom):
        """
        This method computes M[X]^(1/2)*W, which is the typical Brownian velocity. 
        When we are doing nonlocal hydro, M[X]^(1/2)*W is the only thing which is important. 
        However, when we have only local hydro, it becomes useful to also compute M^(-1/2)*W 
        for later use in the drift term. This function will do that. 
        """
        if (self._nonLocal==1):
            Xupsampled = self.getPointsForUpsampledQuad(X);
            MHalfEtaUp = Ewald.calcMOneHalfW(Xupsampled,Dom);
            MHalfEta = np.reshape(self.getDownsampledVelocity(MHalfEtaUp),3*self._Nfib*self._NXpf);
            MMinusHalfEta = 0; # never used
        else:
            nLanczos = 0;
            RandVec1 = np.random.randn(3*self._Nfib*self._NXpf);
            #np.savetxt('RandVec1.txt',RandVec1)
            MHalfAndMinusHalf = self._FibColCpp.MHalfAndMinusHalfEta(X,RandVec1, self._FPLoc); # Replace with PSE 
            MHalfEta = MHalfAndMinusHalf[:3*self._Nfib*self._NXpf];
            MMinusHalfEta = MHalfAndMinusHalf[3*self._Nfib*self._NXpf:];
        return MHalfEta, MMinusHalfEta;#, nLanczos;
    
    def ComputeTotalVelocity(self,X,F,Dom,RPYEval):
        Local = self.LocalVelocity(X,F);
        nonLocal = self.nonLocalVelocity(X,F,Dom,RPYEval);
        return Local+nonLocal;
    
    def StepToMidpoint(self,MHalfEta,dt):
        """
        Step to the midpoint by inverting K. Specifically, we are computing
        K^-1*sqrt(2*kbT/dt)*M^(1/2)*W, and then evolving the fiber to the "midpoint"
        by taking a step of size dt/2
        """
        MidtimeCoordinates = self._FibColCpp.InvertKAndStep(self._tanvecs,self._Midpoints,sqrt(2*self._kbT/dt)*MHalfEta,0.5*dt);
        TauMidtime = MidtimeCoordinates[:self._Nfib*self._NTaupf,:];
        MPMidtime = MidtimeCoordinates[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        XMidtime = MidtimeCoordinates[self._Nfib*self._NXpf:,:];
        return TauMidtime, MPMidtime, XMidtime;
    
    def DriftPlusBrownianVel(self,XMidTime,MHalfEta, MMinusHalfEta,dt,ModifyBE,Dom,RPYEval):
        """
        The purpose of this method is to return the velocity that goes on the RHS of the saddle 
        point system. This has three components to it:
        1) The Brownian velocity sqrt(2*kbT/dt)*M^(1/2)*W
        2) The extra term M*L^(1/2)*W that comes in for modified backward Euler
        3) The stochastic drift term. The drift term is done using an RFD on M when there is 
        nonlocal hydro. Otherwise, it's done using a combination of M^(1/2)*W, M^(-1/2)*W, and 
        Mtilde. The relevant formulas for the drift terms are equations (47) and (C.12) in 
        https://arxiv.org/pdf/2301.11123.pdf
        """
        RandVec2 = np.random.randn(3*self._Nfib*self._NXpf); 
        if (self._rigid):
            ModifyBE = False;
        if (self._nonLocal==1):
            # Generate modified backward Euler step
            LHalfBERand = sqrt(self._kbT)*self._FibColCpp.getLHalfX(RandVec2); #L^1/2*eta_2
            RandVec3 = np.random.randn(3*self._Nfib*self._NXpf); 
            deltaRFD = 1e-5;
            disp = deltaRFD*self._fiberDisc._L;
            RFDCoords = self._FibColCpp.InvertKAndStep(self._tanvecs,self._Midpoints,RandVec3,disp);
            X_RFD = RFDCoords[self._Nfib*self._NXpf:,:];
            VelPlus = self.ComputeTotalVelocity(X_RFD,RandVec3,Dom,RPYEval);
            BEForce = int(ModifyBE)*disp/self._kbT*LHalfBERand; # to cancel later
            # BE force is applied to M and not Mtilde; should give same statistics. 
            VelMinus = self.ComputeTotalVelocity(self._ptsCheb,RandVec3-BEForce,Dom,RPYEval);
            DriftAndMBEVel = self._kbT/disp*(VelPlus-VelMinus);
        else:
            # This includes the velocity for modified backward Euler!
            #np.savetxt('RandVec2.txt',RandVec2)
            DriftAndMBEVel = self._FibColCpp.ComputeDriftVelocity(XMidTime,MHalfEta,MMinusHalfEta,RandVec2,dt,ModifyBE,self._FPLoc); 
        TotalRHSVel = DriftAndMBEVel+sqrt(2*self._kbT/dt)*MHalfEta;
        return TotalRHSVel;
    
    def Mobility(self,LamAlph,impcodt,X,Xs,Dom,RPYEval):
        """
        Mobility calculation for GMRES
        Inputs: lamalph = the input lambdas and alphas impcodt = delta t * implicit coefficient,
        X = Npts * 3 array of Chebyshev point locations, Xs = Npts * 3 array of 
        tangent vectors,Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms, 
        The calculation is 
        [-(M^Local+M^NonLocal)*(impco*dt*F*K*alpha +Lambda); K^T*Lambda ]
        
        This differs from the deterministic class because here we just use the oversampled velocity on the GPU as 
        the ENTIRE velocity (local + nonlocal).
        """
        #timeMe = time.time();
        LamAlph = np.reshape(LamAlph,len(LamAlph))
        Lambdas = LamAlph[:3*self._Nfib*self._NXpf];
        alphas = LamAlph[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas);
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        KTLam = KalphKTLam[3*self._Nfib*self._NXpf:];
        XnLForBend = np.reshape(impcodt*Kalph,(self._Nfib*self._NXpf,3)); # dt/2*K*alpha       
        TotalForce = self.evalBendForce(XnLForBend)+Lambdas;
        AllVel = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval,subSelf=False);
        FirstBlock = -AllVel+Kalph; # zero if lambda, alpha = 0
        #print('Mobility time %f' %(time.time()-timeMe));
        return np.concatenate((FirstBlock,KTLam));
        
    def FiberStress(self,XBend,XLam,Tau_n,Volume):
        """
        Compute fiber stress.
        XLam = X used for the saddle point solve. For Brownian stress, we want to use
        XLam as the argument for the bending and constraint stress, and then tau_n 
        for the extra drift
        """
        #raise NotImplementedError('Stress not implemented for semiflexible filaments')
        BendStress = 1/Volume*self._FibColCpp.evalBendStress(XLam);
        LamStress = 1/Volume*self._FibColCpp.evalStressFromForce(XLam,self._lambdas)
        DriftStress = self._kbT*1/Volume*self._FibColCpp.evalDriftPartofStress(Tau_n);
        #np.savetxt('LambdasN.txt',self._lambdas)
        #np.savetxt('BendStress.txt',BendStress)
        #np.savetxt('LamStress.txt',LamStress)
        return BendStress, LamStress+DriftStress;
    
