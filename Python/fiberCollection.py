import numpy as np
from DiscretizedFiber import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial, CellLinkedList
from FiberCollectionC import FiberCollectionC
import copy
import scipy.sparse as sp
import time
from math import sqrt, exp, pi
from warnings import warn
import BatchedNBodyRPY as gpurpy

# Definitions
verbose = -1;               # debug / timings
svdTolerance = 1e-12;       # tolerance for pseudo-inverse
svdRigid = 0.25; # Exp[||Omega|| dt ] < svdRigid for rigid Brownian fibers

class fiberCollection(object):

    """
    This is a class that operates on a list of fibers together. 
    This python class is basically a wrapper for the C++ class
    FiberCollectionC.cpp, where every function is just calling the 
    corresponding (fast) C++ function. 
        
    An important variable throughout this class is self._nonLocal, which refers to 
    how the hydrodynamics is handled. There are currently 3 options:
    nonLocal = 0: local drag only. This can include intra-fiber hydro depending on if 
    the fiberDiscretization object has FPIsLocal = True;
    nonLocal = 1: full hydrodynamics (all fibers communicate with all fibers)
    nonLocal = 4: only intra-fiber hydrodynamics. Depending on if FPIsLocal = True, 
    then this would be done using the same matrix as local drag OR if FPIsLocal = False, 
    the "finite part" velocity goes on the RHS of the saddle point system. 
    
    Another important function of this class is to keep SpatialDatabase objects inside 
    that can be used to query neighbors. All of these objects are currently "CellLinkedList,"
    objects, which is a parallel CPU/GPU implementation of linked lists. (See SpatialDatabase.py
    for documentation)
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,fibDiscFat=None,nThreads=1,rigidFibs=False,dt=1e-3):
        """
        Constructor for the fiberCollection class. This constructor initializes both the python and 
        corresponding C++ class.
        
        Parameters
        ----------
        nFibs: positive int
            The number of fibers
        turnovertime: double
            Mean time for each fiber to turnover
        fibDisc: FibCollocationDiscretization object
            The discretization object that each fiber will get a copy of, 
        nonLocal: integer
            The type of hydrodynamics (0 for local drag, 1 for full hydro, 4 for intra-fiber hydro only)
        mu: double
            The fluid viscosity
        omega: double
            The frequency of background flow oscillations 
        gam0: double
            The base strain rate of the background flow 
        Dom: Domain object
            Used to initialize the SpatialDatabase objects, 
        kbT: double 
            The thermal energy
        fibDiscFat: FibCollocationDiscretization object
            A discretization object for a "fatter" fiber
        nThreads: int (default is 1)
            The number of OMP threads for parallel calculations 
        rigidFibs: bool (default is False)
            Whether the fibers are rigid
        dt: double, optional
            Time step size (only for setting the tolerance in rigid Schur complement pseudo-inverse)
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
            
        aFat = fibDisc._a;
        if (fibDiscFat is not None):
            if (fibDiscFat._a <= fibDisc._a):
                raise ValueError('You cannot have the fatter discretization be skinnier')    
            if (not fibDisc._RPYSpecialQuad):
                raise ValueError('You cannot use the fat discretization unless you have special quad')  
            aFat = fibDiscFat._a;     
            
        self._FibColCpp = FiberCollectionC(nFibs,self._NXpf,self._NTaupf,nThreads,self._rigid,\
            fibDisc._a,aFat,fibDisc._L,self._mu,self._kbT,svdTolerance,svdRigidTol,fibDisc._RPYSpecialQuad,\
            fibDisc._RPYDirectQuad,fibDisc._RPYOversample);
        self._FibColCpp.initMatricesForPreconditioner(fibDisc._D4BCForce,  \
            fibDisc._D4BCForceHalf,fibDisc._XonNp1MatNoMP,fibDisc._XsFromX,\
            fibDisc._MidpointMat,fibDisc._BendMatX0);
        self._FibColCpp.initResamplingMatrices(fibDisc._nptsUniform,fibDisc._MatfromNtoUniform);
        self._FibColCpp.initMobilityMatrices(fibDisc._sX, fibDisc._sRegularized,\
            fibDisc._FPMatrix.T,fibDisc._DoubletFPMatrix.T,\
            fibDisc._RLess2aResamplingMat,fibDisc._RLess2aWts,\
            fibDisc._DXGrid,fibDisc._stackWTilde_Nx, fibDisc._stackWTildeInverse_Nx,\
            fibDisc._OversamplingWtsMat,fibDisc._EUpsample,fibDisc._EigValThres);
        self._FatCorrection=False;
        if (fibDiscFat is not None and nonLocal):
            self._FatCorrection = True;
            self._FibColCpp.initFatMobilityEvaluator(fibDiscFat._sX, fibDiscFat._sRegularized,\
                fibDiscFat._FPMatrix.T,fibDiscFat._DoubletFPMatrix.T,\
                fibDiscFat._RLess2aResamplingMat,fibDiscFat._RLess2aWts,\
                fibDiscFat._DXGrid,fibDiscFat._stackWTilde_Nx, fibDiscFat._stackWTildeInverse_Nx,\
                fibDiscFat._OversamplingWtsMat,fibDiscFat._EUpsample,fibDiscFat._EigValThres);
            
    
    def initFibList(self,fibListIn, Dom,XFileName=None):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors. This method inserts the fibers by giving them a random
        center position and tangent vector, and therefore allows for possible 
        overlaps in the initial configuration.
        
        Parameters
        ----------
        fibListIn: list
            List of DiscretizedFiber objects. Typically empty, and filled in this method.
        Domain: Domain object
            The periodic domain used to initialize the fibers
        XFileName: string, optional
            If the positions are being read in from a file, this string gives the name
            of the file to read
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
        
    def RSAFibers(self,fibListIn, Dom,StericEval,nDiameters=1):
        """
        Initialize the list of fibers. This method inserts the fibers using
        random sequential addition, meaning the fibers will not overlap. 
        
        Parameters
        ----------
        fibListIn: list
            List of DiscretizedFiber objects. Typically empty, and filled in this method.
        Domain: Domain object
            The periodic domain used to initialize the fibers
        StericEval: StericForceEvaluator object
            Used to compute overlaps between the fibers in random sequential addition
        nDiameters: double, defaults to 1
            The number of diameters apart we want the centerlines of the fibers to be. This
            method guarantees that no pair of fibers will be less than nDiameters apart. 
        """
        self._fibList = fibListIn;
        self._DomLens = Dom.getLens();
        for iFib in range(self._Nfib):
            # Initialize memory
            self._fibList[iFib] = DiscretizedFiber(self._fiberDisc);
            Intersect = True;
            while (Intersect):
                # Initialize straight fiber positions at t=0
                self._fibList[iFib].initFib(Dom.getLens());
                Intersect = StericEval.CheckIntersectWithPrev(self._fibList,iFib,Dom,nDiameters);
        self.fillPointArrays()
    
    def initPointForceVelocityArrays(self, Dom):
        """
        Method to initialize the memory for lists of points, tangent 
        vectors and lambdas for the fiber collection
        
        Parameters
        ----------
        Domain: Domain object
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
    def formBlockDiagRHS(self,XNonLocal,XLocal,t,exForce,Lamstar,Dom,RPYEval):
        """
        This method forms the RHS of the block diagonal saddle point system. 
        Specifically, the saddle point solve that we will eventually solve
        can be written as
        $$
        B \\alpha=M_LF+U_\\textrm{Ex}
        $$
        The point of this method is to return $F$ (the forces) and $U_\\textrm{Ex}$ (the velocity 
        being treated explicitly). 
        
        Parameters
        -----------
        XNonLocal: array
            Chebyshev point locations for the nonlocal velocity evaluation
        XLocal: array
            Chebyshev point locations for the local velocity evaluation (can be different 
            depending on the temporal integrator being used)
        t: double
            time
        exForce: array
            The force being treated explicitly (e.g., gravity, cross linking forces)
        Lamstar: array
            The constraint forces Lambda for the nonlocal velocity calculation
        Dom: Domain object
        RPYEval: RPYVelocityEvaluator object
           Computes nonlocal velocities from forces 
        
        Returns
        --------
        (array,array)
            A pair of arrays. The first contains the $F$ = BendingForceMatrix*(XLocal)+exForce,
            and the second contains the velocity 
            $U_\\textrm{Ex}$ = M_NonLocal*(Lamstar+exForce+BendingForceMatrix*(XNonLocal)) + U0, where U0 
            is the background flow
        """
        # Compute elastic forces at X
        LocalBendForce = self.evalBendForce(XLocal);
        NonLocalBendForce = self.evalBendForce(XNonLocal);
        U0 = self.evalU0(XNonLocal,t);
        UNonLoc = self.nonLocalVelocity(XNonLocal,Lamstar+exForce+NonLocalBendForce, Dom, RPYEval);
        TotalForce = LocalBendForce+exForce;
        return TotalForce, (UNonLoc+U0);

    def calcResidualVelocity(self,BlockDiagAnswer,X,Xs,dtimpco,Lamstar, Dom, RPYEval):
        """
        This method computes the right hand side for the RESIDUAL system. The idea is that
        we first solve a saddle point problem with a guess for the nonlocal force. We then 
        subtract that result from the system with the nonlocal force also treated implicitly
        to get a system to solve. This is described in Section 7.3 in Maxian's PhD dissertation. 
        Specifically, the RHS we are forming here is
        $$
        U = M_{NL} \\left(F\\left(X^n+\\Delta t c K \\alpha -X^{n+1/2,*}\\right)+\\Lambda-\\Lambda^*\\right)
        $$
        where by $\\Lambda$ and $\\alpha$ we mean the results from the block diagonal solver with the initial guess for 
        the nonlocal forcing. 
        
        Parameters
        -----------
        BlockDiagAnswer: array
            The answer for $\\Lambda$$ and $\\alpha$ (in that order) from the block diagonal solver,
        dtimpco: double
            $\\Delta t \\times c$, here $c$ is the implicit coefficient. 
            The implicit coefficient $c$ depends on the
            temporal integrator being used.
        Lamstar: array
            Guess for $\\Lambda$$ from treating nonlocal forcing explicitly
        Dom: Domain object, 
        RPYEval: RPYVelocityEvaluator object   
        
        Returns
        ---------
        array
            The right-hand side for GMRES. This has a block of velocities $U$
            and then a block of zeros (for the second equation $K^T \\Lambda=0$).    
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
    
    def TotalVelocity(self,X,TotalForce,Dom,RPYEval):
        SameSelf = self._fiberDisc._RPYDirectQuad or self._fiberDisc._RPYOversample;
        if (self._fiberDisc._RPYDirectQuad):
            print('Not sure direct quad really makes sense')
        if (not self._nonLocal):
            return self.LocalVelocity(X,TotalForce);
        if (not SameSelf):
            Local = self.LocalVelocity(X,TotalForce,CorrectFat=self._FatCorrection);
            nonLocal = self.nonLocalVelocity(X,TotalForce,Dom,RPYEval,subSelf=False);
            return Local+nonLocal;
        else:
            return self.nonLocalVelocity(X,TotalForce,Dom,RPYEval,subSelf=False);
        
        
    def CheckResiduals(self,LamAlph,impcodt,X,Xs,Dom,RPYEval,t,UAdd=0,ExForces=0):
        """
        This is just for debugging. It verifies that we solved the system we think we did. 
        What it does specifically is take $\\Lambda$ and $\\alpha$ and compute
        $$
        U_1 = U_0+ M \\left(\\Lambda + \\Delta t c K \\alpha  + F_{in} \\right) 
        $$
        where here $M$ includes both the local and nonlocal parts, 
        and compare that result to $U_2 = K \\alpha$
        
        Parameters
        -----------
        LamAlph: array
            The answer for $\\Lambda$ and $\\alpha$ (in that order)
        impcodt: double
            $\\Delta t \\times c$, here $c$ is the implicit coefficient. 
            The implicit coefficient $c$ depends on the
            temporal integrator being used.
        X: array
            The Chebyshev points for the fibers
        Xs: array
            The tangent vectors for the fibers
        Dom: Domain object, 
        RPYEval: RPYVelocityEvaluator object   
        t: double
            The system time
        UAdd: array
            Additional velocity (e.g., from stochastic terms)
        ExForces: array
            Forces being treated explicitly (e.g. cross-linking forces)
        
        Returns
        ---------
        array
            The residual $U_1-U_2$    
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
        U0 = self.evalU0(X,t);
        AllVel = self.TotalVelocity(X,TotalForce,Dom,RPYEval);
        res = (AllVel+U0+UAdd)-Kalph;
        #print('In check residuals, max Kalph %f and res %f' %(np.amax(np.abs(Kalph)),np.amax(np.abs(res))))
        return res;
        
    def Mobility(self,LamAlph,impcodt,X,Xs,Dom,RPYEval):
        """
        Mobility calculation for GMRES. This method takes $\\Lambda$ and $\\alpha$ and computes
        $$
        \\begin{pmatrix}
        -M \\left(c \\Delta t F K \\alpha + \Lambda \\right) \\\\
        K^T \\Lambda
        \\end{pmatrix}
        $$
        In other words, it computes the total forcing from $\\alpha$ and $\\Lambda$ (which 
        is given by $c \\Delta t F K \\alpha + \Lambda $) and then applies the TOTAL
        (local and nonlocal) mobility to that
        
        Parameters
        -----------
        LamAlph: array
            The guess for $\\Lambda$ and $\\alpha$ (in that order)
        impcodt: double
            $\\Delta t \\times c$, here $c$ is the implicit coefficient. 
            The implicit coefficient $c$ depends on the
            temporal integrator being used.
        X: array
            The Chebyshev points for the fibers
        Xs: array
            The tangent vectors for the fibers
        Dom: Domain object, 
        RPYEval: RPYVelocityEvaluator object   
        
        Returns
        -------
        array
            An array with two components: the first block is the velocity (see docstring above) and 
            the second block is $K^T \\Lambda$
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
        AllVel = self.TotalVelocity(X,TotalForce,Dom,RPYEval);
        FirstBlock = -AllVel + Kalph; # zero if lambda, alpha = 0
        #print('Mobility time %f' %(time.time()-timeMe));
        return np.concatenate((FirstBlock,KTLam));
        
    def FactorizePreconditioner(self,X_nonLoc,Xs_nonLoc,implic_coeff,dt):
        """
        Factorize the diagonal solver. This is the first step in solving the saddle point system
        $$
        \\begin{matrix}
        \\left( K-\\Delta t c M_LFK \\right)\\alpha = M_L \\left(F_{ex} + \\Lambda \\right)+U_{ex} \\\\
        K^T \\Lambda = 0
        \end{matrix}
        $$
        for $\\Lambda$ and $\\alpha$. This corresponds to the matrix solve
        $$
        \\begin{pmatrix}
        -M_L & K-\\Delta t c M_LFK \\\\
        K^T &  0 
        \\end{pmatrix}
        \\begin{pmatrix}
        \\Lambda \\\\
        \\alpha 
        \\end{pmatrix}
        = 
        \\begin{pmatrix}
        M_L F_{ex}+U_{ex} \\\\
        0 
        \\end{pmatrix}
        $$
        where $M_L$ is the local mobility on each fiber separately (the argument NBands can 
        be used to make the mobility banded). The purpose of this method specifically is to form
        the Schur complement of the matrix on the LHS above. 
        
        Parameters
        -----------
        X_nonLoc: array
            The Chebyshev points for the fibers
        Xs_nonLoc: array
            The tangent vectors for the fibers
        implic_coeff: double 
            The implicit coefficient $c$ for the
            temporal integrator being used.  
        dt: double
            time step size 
        Nbands: int, optional
            The number of bands in the preconditioner (-1 uses all bands)
        """
        self._FibColCpp.FactorizePreconditioner(X_nonLoc,Xs_nonLoc,implic_coeff*dt,self._FPLoc,self._FatCorrection);
       
    def BlockDiagPrecond(self,UNonLoc,ExForces):
        """
        Solve the saddle point system given in the docstring of FactorizePreconditioner. 
        Here we pass in $F_{ex}$ and $U_{ex}$ and use the factorization from the previous method to
        solve the saddle point system. 
        
        Parameters
        -----------
        UNonLoc: array
            The part of the velocity being treated explicitly (typically nonlocal velocity from other fibers)
        ExForces: array
            The forces being treated explicitly (or forces that don't depend on $\\alpha$ or $\\Lambda$ 
            -- typically bending forces at time $n$ and cross linking forces)
        
        Returns
        ---------
        array
            The array $(\\Lambda,\\alpha)$ of forces and kinematic variables (the solution of the saddle
            point system in the docstring of FactorizePreconditioner)
        """
        UNonLoc = UNonLoc[:3*self._Nfib*self._NXpf];
        LamAlph= self._FibColCpp.applyPreconditioner(ExForces,UNonLoc);
        return LamAlph;
    
    def BrownianUpdate(self,dt,Randoms):
        """
        Random rotational and translational diffusion (assuming a rigid body) over time step dt. 
        This is for the special case when we can compute Brownian motion in a splitting step, 
        rather than using a special temporal integrator. 
        This function calls the corresponding C++ function, which computes a random rotation rate
        $$
        \\alpha = \\sqrt{\\frac{2k_BT}{\\Delta t}}N_{rig}^{1/2}W
        $$
        where $W \\sim $randn(0,1) and 
        $N_{rig} = \\left(K_r^T M_L^{-1}K_r \\right)^\dagger$ is the rigid body mobility matrix. 
        It then rotates and translates by $\\alpha\\Delta t = (\\Omega\\Delta t , U\\Delta t)$. 
        The inversion of the rigid mobility can be problematic because it is not well-posed for 
        fibers that are nearly-straight. We set a tolerance in the constructor using the expected
        time step. Note that this is a void method, but it updates the internal variables to do the
        Brownian motion. 
        
        Parameters
        -----------
        dt: double
            Time step size 
        Randoms: vector of doubles
            Random $N$(0,1) numbers of size 6$\\times$ NFib (3 for rotation and 3 for translation)
        """
        RandAlphas = self._FibColCpp.ThermalTranslateAndDiffuse(self._ptsCheb,self._tanvecs,Randoms, self._FPLoc,dt);
        CppXsMPX = self._FibColCpp.RodriguesRotations(self._tanvecs,self._Midpoints,RandAlphas,dt);   
        self._tanvecs = CppXsMPX[:self._Nfib*self._NTaupf,:];
        self._Midpoints = CppXsMPX[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        self._ptsCheb = CppXsMPX[self._Nfib*self._NXpf:,:];
        
    def nonLocalVelocity(self,X,forces, Dom, RPYEval,subSelf=True):
        """
        Compute the nonlocal velocity due to the fibers. This is the velocity 
        from hydrodynamics, plus the intra-fiber nonlocal velocity if the variable
        self._FPLoc is false. 
        
        Parameters
        -----------
        X: array
            Fiber positions as Npts $\\times$ 3 array
        forces: array 
            Forces as an 3*Npts 1D numpy array
        Dom: Domain object  
        RPYEval: RPYVelocityEvaluator object
        subself: bool, optional
            Whether we subtract the self terms in the RPY velocity evaluator / 
            Ewald splitting step. The default is true, but in some instances (e.g., when the
            self mobility is defined with oversampled RPY), the nonlocal mobility includes
            the self mobility and as such we don't need to subtract it. 
        
        Returns
        --------
        array
            The nonlocal velocity as a one-dimensional array.
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
        if (subSelf):      
            SelfTerms = self.SubtractSelfTerms(self._Ndirect,Xupsampled,Fupsampled,RPYEval._a,RPYEval.NeedsGPU());
            RPYVelocityUp -= SelfTerms;
        if (verbose>=0):
            print('Self time %f' %(time.time()-thist));
            thist=time.time();
        RPYVelocity = self.getDownsampledVelocity(RPYVelocityUp);
        RPYVelocity =  np.reshape(RPYVelocity,self._Nfib*self._NXpf*3);
        if (verbose>=0):
            print('Downsampling time %f' %(time.time()-thist));
        if (np.any(np.isnan(RPYVelocity))):
            raise ValueError('Velocity is nan - stability issue!') 
        return RPYVelocity+doFinitePart*finitePart;
         
    def updateLambdaAlpha(self,lamalph):
        """
        Update the $\\Lambda$ and $\\alpha$ internal variables after the solve is complete. 
        
        Parameters
        -----------
        Lamalph: array
            $\\Lambda$ and $\\alpha$ obtained from the saddle point solve    
        """
        self._lambdas = lamalph[:3*self._Nfib*self._NXpf];
        self._alphas = lamalph[3*self._Nfib*self._NXpf:];
           
    def updateAllFibers(self,dt):
        """
        Update the fibers using the Rodrigues rotation formula. The previous method sets the $\\alpha$ 
        parameter, which contains a tangent vector rotation rate $\\Omega$ and fiber midpoint velocity
        $U_{mp}$. This update step is to first rotate the tangent vectors using the Rodrigues rotation
        formula given in (6.72) of Maxian thesis. 
        
        Parameters
        -----------
        dt: double
            Time step size
        
        Returns
        --------
        double
            After updating the internal variables, it returns the maximum of the Chebyshev point positions
            to check that the simulation is stable. 
        """
        CppXsMPX = self._FibColCpp.RodriguesRotations(self._tanvecs,self._Midpoints,self._alphas,dt);   
        self._tanvecs = CppXsMPX[:self._Nfib*self._NTaupf,:];
        self._Midpoints = CppXsMPX[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        self._ptsCheb = CppXsMPX[self._Nfib*self._NXpf:,:];
        return np.amax(self._ptsCheb);
           
    def getX(self):
        """
        Returns
        -------
        array
            The Chebyshev point positions of all fibers
        """
        return self._ptsCheb;

    def getXs(self):
        """
        Returns
        -------
        array
            The tangent vectors of all fibers
        """
        return self._tanvecs;
    
    def getLambdas(self):
        return self._lambdas; 
    
    def FiberStress(self,XBend,XLam,Tau_n,Volume):
        """
        Compute fiber stress. The formula for the deterministic 
        stress for a given force is 
        $$ \\sigma = \\frac{1}{V}\sum_j X_j F_j^T,$$
        where $j$ refers to the index of Chebyshev points and $V$ 
        is the domain volume. This 
        method, being for the deterministic base class, computes the 
        stress due to the bending force and constraint force
        
        Parameters
        -----------
        XBend: array
            The locations to use to compute the bending force
        XLam: array
            The locations to use to compute the stress from $\\Lambda$
            (which is saved as an internal variable). In order for the 
            stress to be symmetric, these locations have to be the same
            as the one used in the solve $K[X]^T \\Lambda=0$, which means
            they could be different from the bend locations. 
        Tau_n: array
            Tangent vectors at time n. Not used in this base class. 
        Volume: double
            The volume of the domain on which we are computing the stress.             
            
        Returns
        -------
        (array,array,array)
            Three 3 x 3 matrices of stress. The first is the stress
            from the bend forces, the second the stress from $\\Lambda$, and 
            the third is the stochastic drift stress (zero here). 
        """
        BendStress = 1/Volume*self._FibColCpp.evalBendStress(XBend);
        LamStress = 1/Volume*self._FibColCpp.evalStressFromForce(XLam,self._lambdas)
        DriftStress = np.zeros((3,3));
        return BendStress, LamStress,DriftStress;
              
    def uniformForce(self,strengths):
        """
        Generate a uniform force on all fibers.
        
        Parameters
        ----------
        strengths: 3-array
            The strength of the uniform force 
        
        Returns
        --------
        array
            An array of length 3*Nfib*$N_x$ which simply
            tiles the uniform force and assigns it to every
            collocation point on every fiber 
        """
        return np.tile(strengths,self._Nfib*self._NXpf);
    
    def getUniformPoints(self,chebpts):
        """
        Obtain uniform points from a set of Chebyshev points. 
        
        Parameters
        -----------
        chebpts: 2D array
            The Chebyshev points of the fibers, organized 
            into a  (tot#ofpts x 3) array
            
        Returns
        ---------
        2D array
            All fibers sampled at the uniform points. The number of
            uniform points is given by self._fiberDisc._nptsUniform, and 
            is set in the constructor of this class. 
        """
        return self._FibColCpp.getUniformPoints(chebpts);
        
    def getPointsForUpsampledQuad(self,chebpts):
        """
        Obtain upsampled points for direct quadrature
        from a set of Chebyshev points. This simply evaluates
        the Chebyshev interpolant at the upsampled points. 
        
        Parameters
        -----------
        chebpts: 2D array
            The Chebyshev points of the fibers, organized 
            into a  (tot#ofpts x 3) array
            
        Returns
        ---------
        2D array
            All fibers sampled at the points for direct quadrature. The number of
            uniform points is given by self._Ndirect, and 
            is set in the constructor of this class. 
        """
        return self._FibColCpp.getUpsampledPoints(chebpts);
        
    def getForcesForUpsampledQuad(self,chebforces):
        """
        Obtain upsampled forces for direct quadrature
        from a set of Chebyshev points. Unlike the method for 
        $X$, which evaluates the interpolant at a new set of Chebyshev
        points, for forces the steps are more subtle. We need to 
        multiply by $\\widetilde{W}^{-1}$ to get force density from force,
        then extend to an upsampled grid and multiply by weights. The 
        formula is therefore
        $$F_{up} = W_{up} E_{up} \\widetilde{W}^{-1} F,$$
        see (7.10) in Maxian's PhD thesis. 
        
        Parameters
        -----------
        chebforces: 2D array
            The forces on the Chebyshev collocation points, organized
            into a  (tot#ofpts x 3) array
            
        Returns
        ---------
        2D array
            All forces sampled at the points for direct quadrature. The number of
            uniform points is given by self._Ndirect, and 
            is set in the constructor of this class. 
        """
        return self._FibColCpp.getUpsampledForces(chebforces);
        
    def getDownsampledVelocity(self,upsampledVel):
        """
        Downsample the velocity obtained on an upsampled grid. This turns out 
        to be the transpose of the force upsampling matrix discussed in the 
        previous method, so that we are computing
        $$U = \\widetilde{W}^{-1} E^T_{up} W^T_{up} U_{up},$$
        see (7.10) in Maxian's PhD thesis. 
        
        Parameters
        -----------
        upsampledVel: 2D array
            The velocities on the upsampled points points, organized
            into a  (self._Ndirect*Nfib x 3) array
            
        Returns
        ---------
        2D array
            The velocities at the Chebyshev points, using the multiplication above.
        """
        return self._FibColCpp.getDownsampledVelocities(upsampledVel);
        
    def ResampleFromOtherGrid(self,Xother,Nother,typeother):
        XCheb=np.zeros((self._totnumX,3)); 
        for iFib in range(self._Nfib):
            indsNew = np.arange(iFib*self._NXpf, (iFib+1)*self._NXpf);
            indsUni = np.arange(iFib*Nother,(iFib+1)*Nother);
            XCheb[indsNew,:]=self._fiberDisc.ResampleFromOtherGrid(Xother[indsUni,:],Nother,typeother);
        return XCheb;
    
    def SampleToOtherGrid(self,XSelf,Nother,typeother):
        Xnew=np.zeros((self._Nfib*Nother,3)); 
        for iFib in range(self._Nfib):
            indsNew = np.arange(iFib*self._NXpf, (iFib+1)*self._NXpf);
            indsUni = np.arange(iFib*Nother,(iFib+1)*Nother);
            Xnew[indsUni,:]=self._fiberDisc.SampleToOtherGrid(XSelf[indsNew,:],Nother,typeother);
        return Xnew;
    
    def ForceFromForceDensity(self,ForceDen):
        RSAns = False;
        if (ForceDen.shape[0] == 3*self._Nfib*self._NXpf):
            ForceDen = np.reshape(ForceDen,(self._Nfib*self._NXpf,3))
            RSAns = True;
        Forces=np.zeros((self._Nfib*self._NXpf,3)); 
        for iFib in range(self._Nfib):
            indsNew = np.arange(iFib*self._NXpf, (iFib+1)*self._NXpf);
            Forces[indsNew,:]=self._fiberDisc.ForceFromForceDensity(ForceDen[indsNew,:]);
        if (RSAns):
            Forces = np.reshape(Forces,3*self._Nfib*self._NXpf);
        return Forces;
    
    def getg(self,t):
        """
        Get the value of the strain g according to what the background flow dictates.
        If the background flow is oscillatory ($\\omega > 0$), the strain (integral of
        rate of strain) is given by
        $$g = \\frac{\\gamma_0}{\\omega} \\sin{(\\omega t)}.$$
        Otherwise, if $\\omega=0$, we just have a constant shear flow and $g = \\gamma_0 t$. 
        
        Parameters
        -----------
        t: double
            The current time
        
        Returns
        --------
        double
            The non-dimensional strain $g$ 
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
        """
        This method gives the total dimension of the saddle
        point system. If the fibers are inextensible (but not rigid), 
        the dimension is $6 N_x F$, where $F$ is the number of fibers,
        since each collocation point gets a velocity and a constraint force.
        If the fibers are rigid, the dimension is $3N_x F$ (for the constraint
        forces) $+ 6 F$ (for rigid motions).
        
        Returns
        -------
        int
            The dimension of the saddle point system (see docstring of 
            FactorizePreconditioner)
        """
        if (self._rigid):
            return 3*self._Nfib*self._NXpf+6*self._Nfib;
        return 6*self._Nfib*self._NXpf;
    
    def getBlockOneSize(self):
        return 3*self._Nfib*self._NXpf;    
    
    def averageTangentVectors(self):
        """
        Computes the average tangent vectors along each fiber, which are
        useful for analysis. It will return an array with nFib entries, where
        each entry is given by
        $$\\tau_{avg} = \\frac{1}{L} \\int_0^L \\tau(s) ds$$
        
        Returns
        --------
        array
            Array of length nFib x 3, where each row $i$ is the average tangent 
            vector of fiber $i$. 
        """
        avgfibTans = np.zeros((self._Nfib,3));
        wts = np.reshape(self._fiberDisc._w,(self._Npf,1));
        for iFib in range(self._Nfib):
            fiberTans = self._tanvecs[self.getRowInds(iFib),:];
            avgfibTans[iFib,:]=np.sum(wts*fiberTans,axis=0)/self._Lf;
        return avgfibTans;
    
    def initializeLambdaForNewFibers(self,newfibs,ExForces,t,dt,implic_coeff,other=None):
        """
        This method solves a local problem to initialize values of lambda on the fibers
        that were recently (add the current time step) birthed into the system. The 
        local problem we are solving is 
        $$
        \\begin{matrix}
        \\left( K-\\Delta t c M_LFK \\right)\\alpha = M_L \\left(F_{ex} + F X + \\Lambda \\right)+U_{0} \\\\
        K^T \\Lambda = 0
        \end{matrix}
        $$
        for $\\Lambda$ and $\\alpha$. This corresponds to the matrix solve
        $$
        \\begin{pmatrix}
        -M_L & K-\\Delta t c M_LFK \\\\
        K^T &  0 
        \\end{pmatrix}
        \\begin{pmatrix}
        \\Lambda \\\\
        \\alpha 
        \\end{pmatrix}
        = 
        \\begin{pmatrix}
        M_L \\left(FX+F_{ex}\\right)+U_0 \\\\
        0 
        \\end{pmatrix}
        $$
        where $M_L$ is the local mobility on each fiber separately. Notice that 
        there is no nonlocal velocity here; only $U_0$ is the RHS velocity. This makes
        the problem relatively easy to solve. 
        
        Parameters
        -----------
        newfibs: list
            Indicies of fibers that were replaced / birthed in previous time step. 
        ExForces: array
            The external forces (gravity or CL) applied to the fibers. This method will
            compute the bending forces at time $n$ internally. That said, because the 
            birthed fibers are always straight, the bending force is zero. 
        t: double
            Current system time (to generate the background flow)
        dt: double
            Time step size $\\Delta t$
        implic_coeff: double
            The implicit coefficient $c$ in the solve above. 
        other: optional, other fiberCollection object
            This is used to also assign the values of $\\Lambda$ at the previous time 
            step, if we are doing Crank-Nicolson (we never do). 
            
        Returns
        --------
        null
            Nothing, but updates the internal values of $$\Lambda$ with the values computed in the 
            local solve. 
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
        Method for fiber turnover. This method computes the 
        fiber death rate for all fibers, then computes a time 
        for a fiber to die based on that (-log(1-rand)/rate). 
        As long as the time stays below $\\Delta t$, the method keeps 
        choosing fibers uniformly at random to kill off. 
        
        Parameters
        ----------
        tstep: double
            Time step $\\Delta t$
        other: optional, other fiberCollection object
            This is used to also replace the locations at 
            the previous time step, if we are doing Crank-Nicolson
        
        Returns
        --------
        list
            A list of the indicies of all fibers that were replaced 
            during this time step. 
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
        
        Parameters
        ------------
        FileName: string
            The name of the file to write to 
        wora: char 'a' or 'w'
            'a' to append to an existing file (default),
            'w' to write to a new file
        """
        f=open(FileName,wora)
        np.savetxt(f, self._ptsCheb);#, fmt='%1.13f')
        f.close()
        
    ## ====================================================
    ##  "PRIVATE" METHODS NOT ACCESSED OUTSIDE OF THIS CLASS
    ## ====================================================
    def SubtractSelfTerms(self,Ndir,Xupsampled,fupsampled,hydroRadius,useGPU=False):
        """
        Compute the self integrals of the RPY kernel along a single fiber. 
        The reason for doing this is to subtract them from the result we 
        get when doing Ewald splitting, which necessarily gives the velocity
        from all fibers on all others. Typically, we want to do the self
        term as a separate calculation using some more accurate technique. 
        
        Parameters
        -----------
        Ndir: int
            Number of points for direct quad (upsampled quadrature used in 
            Ewald splitting)
        Xupsampled: array
            Upsampled positions of Chebyshev points
        fupsampled: array 
            Upsampled forces (not densities) on the Chebyshev points
        hydroRadius: double
            The hydrodynamic radius
        useGPU: bool, optional
            Whether to use a GPU to compute the self interactions
            (default is false, as C++ is fast enough to not be even close to a 
            bottleneck)
        
        Returns
        ---------
        array 
            The RPY direct sum on each Chebyshev point, 
            $$U_i =\\sum_j M_{RPY}(X_i, X_j)F_j,$$
            where j and i are Chebyshev collocation points on the same fiber.
        """
        if (not useGPU):
            Fat = False;
            if (hydroRadius > self._fiberDisc._a):
                Fat = True;
            U1=self._FibColCpp.SingleFiberRPYSum(Ndir,Xupsampled,fupsampled,Fat);
            return np.reshape(U1,(self._Nfib*Ndir,3));
        
        #print('Using GPU self')
        # GPU Version
        selfMobility= 1.0/(6*pi*self._mu*hydroRadius);
        
        precision = np.float64;
        fupsampledg = np.array(fupsampled,precision);
        Xupsampledg = np.array(Xupsampled,precision);
        
        #It is really important that the result array has the same floating precision as the compiled module, otherwise
        # python will just silently pass by copy and the results will be lost
        MF=np.zeros(self._Nfib*3*Ndir, precision);
        gpurpy.computeMdot(np.reshape(Xupsampledg,self._Nfib*3*Ndir), np.reshape(fupsampledg,self._Nfib*3*Ndir), MF,
                       self._Nfib, Ndir,selfMobility, hydroRadius)
        if (np.amax(np.abs(MF))==0 and np.amax(np.abs(fupsampledg)) > 0):
        	raise ValueError('You are getting zero velocity with finite force, your UAMMD precision is wrong!') 
        return np.reshape(MF,(self._Nfib*Ndir,3));
        
    def LocalVelocity(self,X,Forces,CorrectFat=False):
        """
        Compute the local velocity on each fiber. Given a vector of points
        and forces, this method computes $$U = M_L[X]F.$$ The nature of $M_L$
        is defined in the constructor of this class. 
        
        Parameters
        -----------
        X: array
            Chebyshev point positions of all fibers as a (tot#ofpts*3) 1D array
        Forces: array
            The forces on all fibers, also as a 1D array
            
        Returns
        --------
        array
            The local velocity $M_L[X]F$ on all fibers as a 1D array
        """
        return self._FibColCpp.evalLocalVelocities(X,Forces,self._FPLoc, CorrectFat);
    
    def evalU0(self,Xin,t):
        """
        Compute the background flow on the Chebyshev points. Here we only
        support a shear flow with strength $\\gamma_0$ and frequency $\\omega$, 
        so that 
        $$U_0(x,y,z) = \\gamma_0 \cos{\\left(\\omega t\\right)} (y,0,0)$$
        (in the $x$ direction). 
        
        Parameters
        -----------
        Xin: array
            An N x 3 array with each column being the x, y, and z locations of the 
            Chebyshev points at which we compute the flow
        t: double
            The time
            
        Returns
        --------
        array
            The background flow $U_0$ as a 3N one-dimensional numpy array.
        """
        U0 = np.zeros(Xin.shape);
        U0[:,0]=self._gam0*np.cos(self._omega*t)*Xin[:,1];
        return np.reshape(U0,3*len(Xin[:,0]));
    
    def evalBendForce(self,X_nonLoc):
        """
        Evaluate the bending FORCES on the fibers by applying the 
        precomputed matrix $F$.
        
        Parameters
        -----------
        X_nonLoc: array
            The Chebyshev point positions to evaluate the forces at. 
            The total length of this array is used to determine the number
            of fibers involved. 
        
        Returns
        --------
        array
            The bending forces $FX$ as a 3D numpy array. 
        """
        return self._FibColCpp.evalBendForces(X_nonLoc);
        
    def calcCurvatures(self,X):
        """
        Evaluate fiber curvatures on all fibers. Useful for analyzing
        the results of simulations. The mean $L^2$ curvature
        $$
        \\sqrt{\\frac{1}{L} \\int_0^L X_{ss}(s) \\cdot X_{ss}(s) ds }
        $$
        is returned for each fiber. 
        
        Parameters
        ----------
        X: array
            The Chebyshev points of the fibers.
        
        Returns
        --------
        array
            An array of length nFib with the mean $L^2$ curvature by fiber 
        """
        Curvatures=np.zeros(self._Nfib);
        for iFib in range(self._Nfib):
            Curvatures[iFib] = self._fiberDisc.calcFibCurvature(X[iFib*self._NXpf:(iFib+1)*self._NXpf,:]);
        return Curvatures;

    def getXInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 2D arrays
        of fiber position, for fiber number iFib. 
        This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        
        Parameters
        -----------
        iFib: int
            The fiber index from 0 to NFib-1
            
        Returns
        --------
        The indices of that fiber in an array of all the fiber positions 
        """
        return range(iFib*self._NXpf,(iFib+1)*self._NXpf);
    
    def getTauInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 2D arrays
        of fiber tangent vectors (different from positions) for fiber number iFib. 
        This method could easily be modified to allow
        for a bunch of fibers with different numbers of tangent vectors. 
        
        Parameters
        -----------
        iFib: int
            The fiber index from 0 to NFib-1
            
        Returns
        --------
        The indices of that fiber in an array of all the fiber tangent vectors
        """
        return range(iFib*self._NTaupf,(iFib+1)*self._NTaupf);
        
    def getStackInds(self,iFib):
        """
        Method to get the indices in any (Nfib*Nperfib*3) long 1D arrays
        of fiber positions, for fiber number iFib. 
        This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        
        Parameters
        -----------
        iFib: int
            The fiber index from 0 to NFib-1
            
        Returns
        --------
        The indices of that fiber in a stacked array of all the fiber positions
        """
        return range(iFib*3*self._NXpf,(iFib+1)*3*self._NXpf);
        
class SemiflexiblefiberCollection(fiberCollection):

    """
    This class is a child of fiberCollection which implements BENDING FLUCTUATIONS. 
    There are some additional methods required in this case. In particular, we need
    methods to compute the stochastic drift terms and Brownian velocity $M^{1/2}W$
    """
    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, kbT,fibDiscFat=None,nThreads=1,rigidFibs=False,dt=1e-3):
        super().__init__(nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,fibDiscFat,nThreads,rigidFibs,dt);
        self._iT = 0;
        SPDMatrix = self._fiberDisc._RPYDirectQuad or self._fiberDisc._RPYOversample;
        #if (self._nonLocal==1 and not SPDMatrix and fibDiscFat is None):
        #    raise TypeError('If doing nonlocal hydro with fluctuations, can only do direct quadrature or oversampled or fat discretization')
        
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,t,ExForces,lamstar,Dom,RPYEval=None):
        """
        This method forms the RHS of the block diagonal saddle point system. 
        Specifically, the saddle point solve that we will eventually solve
        can be written as
        $$
        B \\alpha=MF+U_0
        $$
        The point of this method is to return $F$ (the forces) and $U_\\textrm{0}$ (the velocity 
        being treated explicitly). Importantly, this method overwrites the one in
        the deterministic class, because (at the moment), when we include fluctuations
        the matrix $M$ above includes ALL parts of the mobility, so that only velocity
        that is treated explicitly in the background flow $U_0$. 
        
        Parameters
        -----------
        X_nonLoc: array
            Chebyshev point locations for the velocity evaluation
        t: double
            time
        ExForces: array
            The force being treated explicitly (e.g., gravity, cross linking forces)
        lamstar: array
            The constraint forces $\\Lambda^*$ for the nonlocal velocity calculation (they do
            not enter here, but are needed to keep consistency with parent class)
        Dom: Domain object 
            Needed to keep consistency with parent class
        RPYEval: RPYVelocityEvaluator object
            Needed to keep consistency with parent class
        
        Returns
        --------
        (array,array)
            A pair of arrays. The first contains the $F$ = BendingForceMatrix*(X_nonLoc)+ExForces,
            and the second contains the velocity $U_0$ (the background flow)
        """
        self._iT+=1;
        BendForce = self.evalBendForce(X_nonLoc);
        U0 = self.evalU0(X_nonLoc,t);
        TotalForce = BendForce+ExForces;
        return TotalForce, U0;
        
    def BrownianUpdate(self,dt,Randoms):
        """
        For semiflexible fluctuating filaments, the Brownian updates are included as part
        of the temporal integrator (see methods listed below)
        -- we cannot split them into a separate step. 
        For this reason, this method, which is intended to only be used in the split
        case, does nothing and outputs a warning.
        """
        warn('The Brownian update is implemented as part of the solve, not a separate split step!') 
       
    def MHalfAndMinusHalfEta(self,X,RPYEvaluator,Dom):
        """
        This method computes $M[X]^{1/2}W$, where $W \\sim$randn(0,1) is 
        a random vector of i.i.d. Gaussian variables. The method we use to
        compute this depends on what kind of hydrodynamics we have. If
        self._nonLocal=1, we are doing fully nonlocal hydrodynamics, and 
        so we use the provided Ewald splitter argument to call PSE to
        obtain $M^{1/2}W$. Otherwise, we have local hydrodynamics only, 
        and $M^{1/2}W$ is generated fiber-by-fiber using dense linear algebra. 
        In this latter case, it becomes useful to also compute $M^{-1/2}W$
        for later use in the drift term.
        
        Parameters
        -----------
        X: array
            All of the Chebyshev points that are arguments for $M[X]$
        Ewald: RPYVelocityEvaluator object
            Used to call the GPU PSE function to compute $M^{1/2}$ (for 
            nonlocal hydro only)
        Dom: Domain object
            Specifies the periodic domain when doing nonlocal hydro
            
        Returns
        --------
        (array,array)
            Two arrays of size $3FN_x$ (3 times the total number of Chebyshev
            points). The first array is $M^{1/2}W$, and the second is $M^{-1/2}W$
        """
        if (self._nonLocal==1):
            Xupsampled = self.getPointsForUpsampledQuad(X);
            MHalfEtaUp = RPYEvaluator.calcMOneHalfW(Xupsampled,Dom,self._nThreads);
            MHalfEta = np.reshape(self.getDownsampledVelocity(MHalfEtaUp),3*self._Nfib*self._NXpf);
            MMinusHalfEta = 0; # never used
            if (self._FatCorrection):
                # Add the correction for the skinnier radius   
                RandVec1 = np.random.randn(3*self._Nfib*self._NXpf);
                MHalfCorrection = self._FibColCpp.MHalfAndMinusHalfEta(X,RandVec1, self._FPLoc,True);
                MHalfCorrection = MHalfCorrection[:3*self._Nfib*self._NXpf];
                MHalfEta = MHalfEta + MHalfCorrection;
            return MHalfEta, MMinusHalfEta;
        else:
            RandVec1 = np.random.randn(3*self._Nfib*self._NXpf);
            MHalfAndMinusHalf = self._FibColCpp.MHalfAndMinusHalfEta(X,RandVec1, self._FPLoc,False);
            MHalfEta = MHalfAndMinusHalf[:3*self._Nfib*self._NXpf];
            MMinusHalfEta = MHalfAndMinusHalf[3*self._Nfib*self._NXpf:];
        return MHalfEta, MMinusHalfEta;
            
    def StepToMidpoint(self,MHalfEta,dt):
        """
        This function steps to the midpoint by inverting K. Specifically, we are computing
        $$\\alpha^*= K^\dagger \\sqrt{\\frac{2k_BT}{\\Delta t}}M^{1/2}W,$$
        and then evolving the fiber to the "midpoint"
        by taking a step of size $\\Delta t/2$ using tangent vector rotation rates
        $\\alpha^*=\\left(\\Omega^*,U_{mp}^*\\right)$
        
        Parameters
        -----------
        MHalfEta: array
            This is $M[X]^{1/2}W$, obtained using the previous method 
        dt: double
            The time step size
            
        Returns
        --------
        (array, array, array)
            Three arrays which represent the tangent vectors $\\tau^{n+1/2,*}$,
            fiber midpoints $X_{mp}^{n+1/2,*}$, and fiber positions
            $X^{n+1/2,*}$ which are obtained via the update with $\\alpha^*$
        """
        MidtimeCoordinates = self._FibColCpp.InvertKAndStep(self._tanvecs,self._Midpoints,sqrt(2*self._kbT/dt)*MHalfEta,0.5*dt);
        TauMidtime = MidtimeCoordinates[:self._Nfib*self._NTaupf,:];
        MPMidtime = MidtimeCoordinates[self._Nfib*self._NTaupf:self._Nfib*self._NXpf,:];
        XMidtime = MidtimeCoordinates[self._Nfib*self._NXpf:,:];
        return TauMidtime, MPMidtime, XMidtime;
    
    def DriftPlusBrownianVel(self,XMidTime,MHalfEta, MMinusHalfEta,dt,ModifyBE,Dom,RPYEval):
        """
        The purpose of this method is to return the velocity that goes on the RHS of the saddle 
        point system for fluctuating fibers. This has three components to it: \n
        1) The Brownian velocity $\\sqrt{\\frac{2k_BT}{\\Delta t}}M^{1/2}W$ \n
        2) The extra term $\\sqrt{k_B T} ML^{1/2}W$ that comes in when we use modified backward Euler, see
           (8.20) in Maxian's PhD thesis. Here $L$ is the bending energy matrix. \n 
        3) The stochastic drift velocity necessary to ensure we sample from the correct
           equilibrium distribution.      
           The formula for the drift term depends on the type of hydrodynamics being considered. 
           If we are considering only LOCAL hydrodynamics (no inter-fiber communication), it is 
           given by
           $$U_{MD}=\\sqrt{\\frac{2 k_B T}{\\Delta t}} \\left(M^{n+1/2,*}-M^n\\right)\\left(M^n\\right)^{-T/2}\\eta,$$
           where $\\eta \\sim$randn(0,1) and this formula is computed using dense linear algebra on
           each fiber separately. 
           In the case when there is inter-fiber hydrodynamics, this resistance problem becomes 
           expensive, and so we use an alternative approach, computing
           $$U_{MD} = \\frac{k_B T}{\\delta L} \\left(M\\left(\\tau^{(RFD)}\\right)-M\\left(\\tau^n\\right)\\right)\\eta$$
           where the RFD for $\\tau$ is obtained by computing $\\mu=K^\dagger \\eta$ and rotating 
           $\\tau^n$ by the oriented angle $\\delta L \\mu$. Here we use $\\delta=10^{-5}$ as the 
           small parameter in this random finite difference.
           For more details on this, see formulas (8.31) and (8.32) in Maxian's PhD thesis.
        
        Parameters
        -----------
        XMidTime: array
            The positions $X^{n+1/2,*}$ at all Chebyshev points
        MHalfEta: array
            The Brownian velocity $M^{1/2}W$ at all Chebyshev points
        MMinusHalfEta: array
            The Brownian velocity $M^{-1/2}W$ at all Chebyshev points (this 
            is only necessary when we use the first formula for the drift term -- 
            local hydrodynamics only)
        dt: double
            Time step size
        ModifyBE: bool
            Whether to include in the velocity the term $\\sqrt{k_B T} ML^{1/2}W$ 
            for modified backward Euler
        Dom: Domain object
            Specifies the periodic domain when doing nonlocal hydro
        RPYEval: RPYVelocityEvaluator object
            Used to call the GPU PSE function to apply $M$ (for
            nonlocal hydro only)
        
        Returns
        --------
        array
            The total velocity to be treated explicitly in the saddle point
            solve, which is simply the sum of terms 1-3 above.
            
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
            # If everything is oversampled this can just be a call to nonlocalVel. Check. 
            VelPlus = self.TotalVelocity(X_RFD,RandVec3,Dom,RPYEval);
            BEForce = int(ModifyBE)*disp/self._kbT*LHalfBERand; # to cancel later
            # BE force is applied to M and not Mtilde; should give same statistics. 
            VelMinus = self.TotalVelocity(self._ptsCheb,RandVec3-BEForce,Dom,RPYEval);
            DriftAndMBEVel = self._kbT/disp*(VelPlus-VelMinus);
        else:
            # This includes the velocity for modified backward Euler!
            #np.savetxt('RandVec2.txt',RandVec2)
            DriftAndMBEVel = self._FibColCpp.ComputeDriftVelocity(XMidTime,MHalfEta,MMinusHalfEta,RandVec2,dt,ModifyBE,self._FPLoc); 
        TotalRHSVel = DriftAndMBEVel+sqrt(2*self._kbT/dt)*MHalfEta;
        return TotalRHSVel;
    
    def FiberStress(self,XBend,XLam,Tau_n,Volume):
        """
        Compute fiber stress. The formula for the deterministic 
        stress for a given force is 
        $$ \\sigma = \\frac{1}{V}\sum_j X_j F_j^T,$$
        where $j$ refers to the index of Chebyshev points and $V$ 
        is the domain volume. In this subclass, we also compute
        a Brownian part to the stress, which is given by the formula
        $$\\frac{\\partial S_{ij}}{\\partial X_k} K_{ka}K^{-1}_{aj},$$
        where $S[X]$ is the $9 \\times 3N_x$ tensor that gives stress from the 
        forces on the Chebyshev points. This formula has NOT been verified or
        tested, but is included here for completeness.
        
        
        Parameters
        -----------
        XBend: array
            The locations to use to compute the bending force
        XLam: array
            The locations to use to compute the stress from $\\Lambda$
            (which is saved as an internal variable). In order for the 
            stress to be symmetric, these locations have to be the same
            as the one used in the solve $K[X]^T \\Lambda=0$, which means
            they could be different from the bend locations. 
        Tau_n: array
            Tangent vectors at time n.
        Volume: double
            The volume of the domain on which we are computing the stress.             
            
        Returns
        -------
        (array,array,array)
            Three 3 x 3 matrices of stress. The first is the stress
            from the bend forces, the second the stress from $\\Lambda$, and 
            the third is the stochastic drift stress.
        """
        #raise NotImplementedError('Stress not implemented for semiflexible filaments')
        BendStress = 1/Volume*self._FibColCpp.evalBendStress(XLam);
        LamStress = 1/Volume*self._FibColCpp.evalStressFromForce(XLam,self._lambdas)
        DriftStress = self._kbT*1/Volume*self._FibColCpp.evalDriftPartofStress(Tau_n);
        #np.savetxt('LambdasN.txt',self._lambdas)
        #np.savetxt('BendStress.txt',BendStress)
        #np.savetxt('LamStress.txt',LamStress)
        return BendStress, LamStress, DriftStress;
    
