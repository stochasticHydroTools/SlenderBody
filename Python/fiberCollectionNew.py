import numpy as np
from DiscretizedFiberNew import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial, CellLinkedList
from FiberCollectionNew import FiberCollectionNew
import copy
import numba as nb
import scipy.sparse as sp
import time
from math import sqrt, exp, pi
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
    FiberCollectionNew.cpp, where every function is just calling the 
    corresponding (fast) C++ function
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads=1,rigidFibs=False):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, turnover time = mean time for each fiber to turnover,
        fibDisc = discretization object that each fiber will get a copy of, 
        nonLocal = are we doing nonlocal hydro? (TEMP: 0 for local drag, >0 for intra-fiber hydro))
        mu = fluid viscosity,omega = frequency of background flow oscillations, 
        gam0 = base strain rate of the background flow, Dom = Domain object to initialize the SpatialDatabase objects, 
        kbT = thermal energy, eigValThres = eigenvalue threshold for intra-fiber mobility, 
        nThreads = number of OMP threads for parallel calculations, rigidFibs = whether the fibers are rigid
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
        self._FibColCpp = FiberCollectionNew(nFibs,self._NXpf,self._NTaupf,nThreads,
            self._fiberDisc._a,self._fiberDisc._L,self._mu,self._kbT,svdTolerance,svdRigid,self._fiberDisc._RPYSpecialQuad,\
            self._fiberDisc._RPYDirectQuad,self._fiberDisc._RPYOversample);
        self._FibColCpp.initMatricesForPreconditioner(self._fiberDisc._D4BC, self._fiberDisc._D4BCForce,  \
            self._fiberDisc._D4BCForceHalf,self._fiberDisc._XonNp1MatNoMP,self._fiberDisc._XsFromX,\
            self._fiberDisc._MidpointMat,self._fiberDisc._BendMatX0);
        self._FibColCpp.initResamplingMatrices(self._fiberDisc._nptsUniform,self._fiberDisc._MatfromNtoUniform);
        self._FibColCpp.initMobilityMatrices(self._fiberDisc._sX, fibDisc._leadordercs,\
            self._fiberDisc._FPMatrix.T,self._fiberDisc._DoubletFPMatrix.T,\
            self._fiberDisc._RLess2aResamplingMat,self._fiberDisc._RLess2aWts,\
            self._fiberDisc._DXGrid,self._fiberDisc._stackWTilde_Nx, self._fiberDisc._stackWTildeInverse_Nx,\
            self._fiberDisc._OversamplingWtsMat,self._fiberDisc._EUpsample,eigValThres);
    
    def initFibList(self,fibListIn, Dom,tanvecfileName=None,midpointfileName=None):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
        Inputs: list of Discretized fiber objects (typically empty), Domain object,
        file names for the points and tangents vectors if we are initializing from file
        The file names can be either file names or the actual point/tanvec locations.
        """
        self._fibList = fibListIn;
        self._DomLens = Dom.getLens();
        if (tanvecfileName is None): # initialize straight fibers
            for jFib in range(self._Nfib):
                if self._fibList[jFib] is None:
                    # Initialize memory
                    self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                    # Initialize straight fiber positions at t=0
                    self._fibList[jFib].initFib(Dom.getLens());
        else:
            AllXs = np.loadtxt(tanvecfileName)
            try:
                AllMPs = np.loadtxt(midpointfileName)
            except:
                AllMPs = self._DomLens*np.random.rand((self._Nfib,3));
            for jFib in range(self._Nfib): 
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                XsThis = AllXs[jFib*self._NTaupf:(jFib+1)*self._NTaupf,:];
                XMPThis = AllMPs[jFib,:];
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
            self._Midpoints[iFib,:]=fib._XMP.copy();
    
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
        evaluator for the nonlocal terms, includeFP = whether to do include the finite part (intra-fiber)
        interactions in the nonlocal velocity 
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
        passed to GMRES. 
        Inputs: BlockDiagAnswer = the answer for lambda and alpha from the block diagonal solver, 
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, dtimpco = delta t * implicit coefficient,
        lamstar = lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, FPimplicit = whether to do the finite part integral implicitly.
        The RHS for the residual system is  
        [M^NonLocal*(F*(X^n+dt*impco*K*alpha-X^(n+1/2,*)+lambda - lambda^*)); 0], where by lambda and alpha
        we mean the results from the block diagonal solver 
        """
        Lambdas = BlockDiagAnswer[:3*self._Nfib*self._NXpf];
        alphas = BlockDiagAnswer[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas,self._rigid);
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        FbendCorrection = self.evalBendForce(self._ptsCheb- X+np.reshape(dtimpco*Kalph,(self._Nfib*self._NXpf,3)));
        LamCorrection = Lambdas-Lamstar;
        TotalForce = FbendCorrection+LamCorrection;
        nonLocalVel = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval);
        return np.concatenate((nonLocalVel,np.zeros(3*self._NXpf*self._Nfib)));
        
    def CheckResiduals(self,LamAlph,impcodt,X,Xs,Dom,RPYEval,t,UAdd=0,ExForces=0):
        LamAlph = np.reshape(LamAlph,len(LamAlph))
        Lambdas = LamAlph[:3*self._Nfib*self._NXpf];
        alphas = LamAlph[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas,self._rigid); 
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
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors,Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms, 
        returnForce = whether we need the force (we do if this collection is part of a collection 
        of species and  the hydro for the species collection is still outstanding) 
        The calculation is 
        [-(M^Local+M^NonLocal)*(impco*dt*F*K*alpha +Lambda); K^T*Lambda ]
        """
        LamAlph = np.reshape(LamAlph,len(LamAlph))
        Lambdas = LamAlph[:3*self._Nfib*self._NXpf];
        alphas = LamAlph[3*self._Nfib*self._NXpf:];
        KalphKTLam = self._FibColCpp.KAlphaKTLambda(Xs,alphas,Lambdas,self._rigid);
        Kalph = KalphKTLam[:3*self._Nfib*self._NXpf];
        KTLam = KalphKTLam[3*self._Nfib*self._NXpf:];
        XnLForBend = np.reshape(impcodt*Kalph,(self._Nfib*self._NXpf,3)); # dt/2*K*alpha       
        TotalForce = self.evalBendForce(XnLForBend)+Lambdas;
        Local = self.LocalVelocity(X,TotalForce);
        nonLocal = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval);
        FirstBlock = -(Local+nonLocal)+Kalph; # zero if lambda, alpha = 0
        return np.concatenate((FirstBlock,KTLam));
       
    def BlockDiagPrecond(self,UNonLoc,Xs_nonLoc,dt,implic_coeff,X_nonLoc,ExForces):
        """
        Block diagonal solver. This solver takes inputs Xs_nonLoc (tangent vectors),, 
        dt (time step size), implic_coeff (implicit coefficient for temporal integrator), 
        X_nonLoc (locations of points), and doFP (whether to include finite part), and solves
        the saddle point system. 
        (K-impco*dt*M^L*F*K)*alpha = M^L*ExForces + Lambda)+UNonLoc
        K^T*Lambda = 0
        for Lambda, and alpha. B is K with the implicit part included This corresponds to the matrix solve
        [-M^L K-impco*dt*M^L*F*K; K^T 0]*[Lambda; alpha]=[M^L*ExForces; 0]
        where M^L is the local mobility on each fiber separately. If doFP=1, this 
        matrix describes the hydrodynamics fiber-by-fiber. Otherwise, it is just
        a local drag matrix
        """
        UNonLoc = UNonLoc[:3*self._Nfib*self._NXpf];
        LamAlph= self._FibColCpp.applyPreconditioner(X_nonLoc,Xs_nonLoc,ExForces,UNonLoc,implic_coeff,dt,self._FPLoc,self._rigid);
        return LamAlph;
    
    def BrownianUpdate(self,dt,Randoms):
        """
        Random rotational and translational diffusion (assuming a rigid body) over time step dt. 
        This function calls the corresponding python function, which computes 
        alpha = sqrt(2*kbT/dt)*N_rig^(1/2)*randn,
        where N_rig = pinv(K_r^T M^(-1)*K_r)
        Then rotates and translates by alpha*dt = (Omega*dt, U*dt)
        """
        RandAlphas = self._FibColCpp.ThermalTranslateAndDiffuse(self._ptsCheb,self._tanvecs,Randoms, (self._nonLocal > 0),dt);
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
        RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities, doFinitePart = whether to 
        include the finite part integral in the calculation (no if it's being done implicitly)
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
    
    def FiberStress(self,XforNL,Lambdas,Dom):
        """
        Compute fiber stress. This method not complete. 
        """
        return NumbaColloc.FiberStressNumba(XforNL,self.evalBendForceDensity(XforNL),Lambdas,\
            1.0*Dom.getLens(),self._Nfib,self._Npf,self._fiberDisc._w);
              
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
        Obtain upsampled points from a set of Chebyshev points. 
        Inputs: chebforces as a (tot#ofpts x 3) array
        Outputs: upsampled forces as a (nPtsUpsample*Nfib x 3) array.
        """
        return self._FibColCpp.getUpsampledForces(chebforces);
        
    def getDownsampledVelocity(self,upsampledVel):
        """
        Obtain values at Chebyshev nodes from upsampled 
        Inputs: upsampled values as a (nPtsUpsample*Nfib x 3) array.
        Outputs: vals at cheb pts as a (tot#ofpts x 3) array
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
        return 6*self._Nfib*self._NXpf;
    
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
            lamalph = self._FibColCpp.applyPreconditioner(X,Xs,FTotal,U0,implic_coeff,dt,self._FPLoc,self._rigid);
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
        Inputs: Xs = positions as a (tot#ofpts*3) 1D array
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
    This class is a child of fiberCollection which implements BENDING FLUCTUATIONS
    """

    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, kbT,eigValThres,nThreads=1):
        super().__init__(nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads);
        self._iT = 0;
        
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,t,ExForces,lamstar,Dom,RPYEval=None):
        """
        """
        self._iT+=1;
        BendForce = self.evalBendForce(X_nonLoc);
        U0 = self.evalU0(X_nonLoc,t);
        TotalForce = BendForce+ExForces;
        return TotalForce, U0;
        
    def BrownianUpdate(self,dt,kbT,Randoms):
        # Do nothing for semiflexible filaments
        raise ValueError('The Brownian update is implemented as part of the solve, not a separate split step!') 
       
    def MHalfAndMinusHalfEta(self,X,Ewald,Dom):
        if (self._nonLocal==1):
            Xupsampled = self.getPointsForUpsampledQuad(X);
            MHalfEtaUp = Ewald.calcMOneHalfW(Xupsampled,Dom);
            MHalfEta = np.reshape(self.getDownsampledVelocity(MHalfEtaUp),3*self._Nfib*self._NXpf);
            MMinusHalfEta = 0; # never used
        else:
            RandVec1 = np.random.randn(3*self._Nfib*self._NXpf);
            #np.savetxt('RandVec1.txt',RandVec1)
            MHalfAndMinusHalf = self._FibColCpp.MHalfAndMinusHalfEta(X,RandVec1, self._FPLoc); # Replace with PSE 
            MHalfEta = MHalfAndMinusHalf[:3*self._Nfib*self._NXpf];
            MMinusHalfEta = MHalfAndMinusHalf[3*self._Nfib*self._NXpf:];
        return MHalfEta, MMinusHalfEta;
    
    def ComputeTotalVelocity(self,X,F,Dom,RPYEval):
        Local = self.LocalVelocity(X,F);
        nonLocal = self.nonLocalVelocity(X,F,Dom,RPYEval);
        return Local+nonLocal;
    
    def StepToMidpoint(self,MHalfEta,dt):
        MidtimeCoordinates = self._FibColCpp.InvertKAndStep(self._tanvecs,self._Midpoints,sqrt(2*self._kbT/dt)*MHalfEta,0.5*dt);
        TauMidtime = MidtimeCoordinates[:self._Nfib*self._NTaupf,:];
        MPMidtime = MidtimeCoordinates[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        XMidtime = MidtimeCoordinates[self._Nfib*self._NXpf:,:];
        return TauMidtime, MPMidtime, XMidtime;
    
    def DriftPlusBrownianVel(self,XMidTime,MHalfEta, MMinusHalfEta,dt,ModifyBE,Dom,RPYEval):
        RandVec2 = np.random.randn(3*self._Nfib*self._NXpf); 
        if (self._nonLocal==1):
            # Generate modified backward Euler step
            LHalfBERand = sqrt(dt/2)*self._FibColCpp.getLHalfX(RandVec2); #L^1/2*eta_2
            RandVec3 = np.random.randn(3*self._Nfib*self._NXpf); 
            deltaRFD = 1e-5;
            disp = deltaRFD*self._fiberDisc._L;
            RFDCoords = self._FibColCpp.InvertKAndStep(self._tanvecs,self._Midpoints,RandVec3,disp);
            X_RFD = RFDCoords[self._Nfib*self._NXpf:,:];
            VelPlus = self.ComputeTotalVelocity(X_RFD,RandVec3,Dom,RPYEval);
            BEForce = disp/self._kbT*LHalfBERand; # to cancel later
            # BE force is applied to M and not Mtilde; should give same statistics. 
            VelMinus = self.ComputeTotalVelocity(self._ptsCheb,RandVec3-BEForce,Dom,RPYEval);
            DriftAndMBEVel = self._kbT/disp*(VelPlus-VelMinus);
        else:
            # This includes the velocity for modified backward Euler!
            #np.savetxt('RandVec2.txt',RandVec2)
            DriftAndMBEVel = self._FibColCpp.ComputeDriftVelocity(XMidTime,MHalfEta,MMinusHalfEta,RandVec2,dt,ModifyBE,self._FPLoc); 
        TotalRHSVel = DriftAndMBEVel+sqrt(2*self._kbT/dt)*MHalfEta;
        return TotalRHSVel;
        
        
    
