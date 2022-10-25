import numpy as np
from DiscretizedFiberNew import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial, CellLinkedList
from FiberCollectionNew import FiberCollectionNew
#import FiberUpdateNumba as NumbaColloc
import copy
import numba as nb
import scipy.sparse as sp
import time
from math import sqrt, exp, pi
#import BatchedNBodyRPY as gpurpy

# Documentation last updated 03/12/2021

# Definitions (for special quadrature)
verbose = -1;               # debug / timings

class fiberCollection(object):

    """
    This is a class that operates on a list of fibers together. Its main role
    is to compute the nonlocal hydrodynamics between fibers. 
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, nThreads=1):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, turnover time = mean time for each fiber to turnover,
        fibDisc = discretization object that each fiber will get a copy of, 
        nonLocal = are we doing nonlocal hydro? (0 = local drag, 1 = full hydrodynamics, 
        2 = hydrodynamics without finite part, 3 = hydrodynamics without special quad, 4 = 
        hydrodynamics without other fibers (only finite part))
        mu = fluid viscosity,omega = frequency of background flow oscillations, 
        gam0 = base strain rate of the background flow, Dom = Domain object to initialize the SpatialDatabase objects, 
        nThreads = number of OMP threads for parallel calculations
        """
        self._Nfib = nFibs;
        self._fiberDisc = fibDisc;
        self._NXpf = self._fiberDisc._Nx;
        self._NTaupf = self._fiberDisc._Ntau;
        self._Ndirect = self._fiberDisc._nptsDirect;
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._nThreads = nThreads;
        self._deathrate = 1/turnovertime; # 1/s
        self.initPointForceVelocityArrays(Dom);
        if (self._nonLocal==1):
            raise ValueError('Correction quad not supported anymore');
        
        DLens = Dom.getPeriodicLens();
        for iD in range(len(DLens)):
            if (DLens[iD] is None):
                DLens[iD] = 1e99;
        
        # Initialize C++ class 
        self._FibColCpp = FiberCollectionNew(nFibs,self._NXpf,self._NTaupf,\
            self._fiberDisc._L,self._fiberDisc._a,self._mu,nThreads);
        self._FibColCpp.initMatricesForPreconditioner(self._fiberDisc._D4BC, self._fiberDisc._D4BCForce,  \
            self._fiberDisc._D4BCForceHalf,\
            self._fiberDisc._XonNp1MatNoMP,self._fiberDisc._XsFromX,self._fiberDisc._MidpointMat,\
            self._fiberDisc._stackWTilde_Nx, self._fiberDisc._stackWTildeInverse_Nx);
        self._FibColCpp.initResamplingMatrices(self._fiberDisc._nptsUniform,self._fiberDisc._MatfromNtoUniform);
        self._FibColCpp.initMobilityMatrices(self._fiberDisc._sX, fibDisc._leadordercs,\
            self._fiberDisc._FPMatrix.T,self._fiberDisc._DoubletFPMatrix.T,\
            self._fiberDisc._RLess2aResamplingMat,self._fiberDisc._RLess2aWts,\
            self._fiberDisc._DXGrid,self._fiberDisc._truRPYMob);
    
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
        # Initialize the spatial database objects
        self._SpatialCheb = CellLinkedList(self._ptsCheb,Dom,nThr=self._nThreads);
        self._SpatialUni = CellLinkedList(np.zeros((self._fiberDisc._nptsUniform*self._Nfib,3)),Dom,nThr=self._nThreads);
        self._SpatialDirectQuad = CellLinkedList(np.zeros((self._totnumDirect,3)),Dom,nThr=self._nThreads);

    def fillPointArrays(self):
        """
        Copy the X and Xs arguments from self._fibList (list of fiber
        objects) into large (tot#pts x 3) arrays that are stored in memory
        """
        print('Calling fill point arrays')
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
    def formBlockDiagRHS(self, X_nonLoc,Xs_nonLoc,t,exForceDen,lamstar,Dom,RPYEval,FPimplicit=0,returnForce=False):
        """
        RHS for the block diagonal GMRES system. 
        Inputs: X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, t = system time, exForceDen = external force density (treated explicitly), 
        lamstar = lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, FPimplicit = whether to do the finite part integral implicitly,
        returnForce = whether we need the force (we do if this collection is part of a collection of species and 
        the hydro for the species collection is still outstanding) 
        The block diagonal RHS is 
        [M^Local*(F*X^n+f^ext) + M^NonLocal * (F*X^(n+1/2,*)+lambda*+f^ext)+u_0; 0]
        M^Local is the part that is being treated explicitly, and M^NonLocal is the part that is being treated
        implicitly
        """
        fnBend = self.evalBendForceDensity(self._ptsCheb); 
        TotalForceDen = exForceDen+fnBend; 
        # Multiply by Wtilde to get force density
        Local = self.calcLocalVelocities(X_nonLoc,TotalForceDen);
        if (FPimplicit==1 and self._nonLocal > 0): # do the finite part separetely if it's to be treated implicitly
            Local+= self._FibColCpp.FinitePartVelocity(X_nonLoc, TotalForceDen,self._fiberDisc._truRPYMob)
            #print('Doing FP separate, it''s being treated implicitly')
        U0 = self.evalU0(X_nonLoc,t);
        fNLBend = self.evalBendForceDensity(X_nonLoc);
        totForce = lamstar+exForceDen+fNLBend;
        # Compute upsampled points once and for all iterations
        if (self._nonLocal == 3):
            self._Xupsampled = self.getPointsForUpsampledQuad(X_nonLoc);
            self._SpatialDirectQuad.updateSpatialStructures(self._Xupsampled,Dom);   
        nonLocal = self.nonLocalVelocity(X_nonLoc,Xs_nonLoc,totForce,Dom,RPYEval,1-FPimplicit);
        return Local+nonLocal+U0;
    
    def calcNewRHS(self,BlockDiagAnswer,X_nonLoc,Xs_nonLoc,dtimpco,lamstar, Dom, RPYEval,FPimplicit=0,returnForce=False):
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
        lambdas = BlockDiagAnswer[:3*self._Nfib*self._Npf];
        alphas = BlockDiagAnswer[3*self._Nfib*self._Npf:];
        Kalph, Kstlam = self.KProductsAllFibers(Xs_nonLoc,alphas,lambdas);
        fbendCorrection = self.evalBendForceDensity(self._ptsCheb- X_nonLoc+np.reshape(dtimpco*Kalph,(self._Nfib*self._Npf,3)));
        lamCorrection = lambdas-lamstar;
        totForceD = fbendCorrection+lamCorrection;
        nonLocal = self.nonLocalVelocity(X_nonLoc,Xs_nonLoc,totForceD,Dom,RPYEval,1-FPimplicit);
        if (returnForce):
            return np.concatenate((nonLocal,np.zeros((2*self._nPolys+3)*self._Nfib))), totForceD;
        return np.concatenate((nonLocal,np.zeros((2*self._nPolys+3)*self._Nfib)));
        
    def Mobility(self,lamalph,impcodt,X_nonLoc,Xs_nonLoc,Dom,RPYEval,returnForce=False):
        """
        Mobility calculation for GMRES
        Inputs: lamalph = the input lambdas and alphas impcodt = delta t * implicit coefficient,
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors,Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms, 
        returnForce = whether we need the force (we do if this collection is part of a collection 
        of species and  the hydro for the species collection is still outstanding) 
        The calculation is 
        [-(M^Local+M^NonLocal)*(impco*dt*F*K*alpha +lambda); K*lambda ]
        """
        lamalph = np.reshape(lamalph,len(lamalph))
        lambdas = lamalph[:3*self._Nfib*self._Npf];
        alphas = lamalph[3*self._Nfib*self._Npf:];
        Kalph, Ktlam = self.KProductsAllFibers(Xs_nonLoc,alphas,lambdas);
        XnLForBend = np.reshape(impcodt*Kalph,(self._Nfib*self._Npf,3)); # dt/2*K*alpha       
        forceDs = self.evalBendForceDensity(XnLForBend) +lambdas;
        Local = self.calcLocalVelocities(np.reshape(Xs_nonLoc,self._totnum*3),forceDs);
        nonLocal = self.nonLocalVelocity(X_nonLoc,Xs_nonLoc,forceDs,Dom,RPYEval);
        FirstBlock = -(Local+nonLocal)+Kalph; # zero if lambda, alpha = 0
        SecondBlock = Ktlam;
        if (returnForce):
            return np.concatenate((FirstBlock,SecondBlock)), forceDs;   
        return np.concatenate((FirstBlock,SecondBlock));
       
    def BlockDiagPrecond(self,b,Xs_nonLoc,dt,implic_coeff,X_nonLoc,doFP=0):
        """
        Block diagonal preconditioner for GMRES. 
        b = RHS vector. Xs_nonLoc = tangent vectors as an Npts x 3 array. 
        dt = timestep, implic_coeff = implicit coefficient, X_nonLoc = fiber positions as 
        an Npts x 3 array, doFP = whether to do finite part implicitly. 
        The preconditioner is 
        [-M^Local K-impco*dt*M^Local*F*K; K^* 0]
        """
        b1D = np.reshape(b,len(b));
        return self._FibColCpp.applyPreconditioner(X_nonLoc,Xs_nonLoc,b1D,implic_coeff*dt,\
            doFP,self._fiberDisc._truRPYMob,1e-10);
    
    def BrownianUpdate(self,dt,kbT,Randoms):
        # Compute the average X and Xs
        wRs = np.reshape(self._fiberDisc._w,(self._Npf,1));
        self._ptsCheb, self._tanvecs= NumbaColloc.BrownianVelocities(self._ptsCheb,self._tanvecs,wRs,\
            np.array([self._fiberDisc._alpha,self._fiberDisc._beta,self._fiberDisc._gamma]),self._Nfib,self._Npf,self._mu,self._Lf,kbT,dt,Randoms);
        
    def nonLocalVelocity(self,X_nonLoc,Xs_nonLoc,forceDs, Dom, RPYEval,doFinitePart=1):
        """
        Compute the non-local velocity due to the fibers.
        Inputs: Arguments X_nonLoc, Xs_nonLoc = fiber positions and tangent vectors as Npts x 3 arrays, 
        forceDs = force densities as an 3*npts 1D numpy array
        Dom = Domain object the computation happens on,  
        RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities, doFinitePart = whether to 
        include the finite part integral in the calculation (no if it's being done implicitly)
        Outputs: nonlocal velocity as a tot#ofpts*3 one-dimensional array.
        """
        # If not doing nonlocal hydro, return nothing
        if (self._nonLocal==0):
            return np.zeros(3*self._Nfib*self._NXpf);
        elif (self._nonLocal==2):
            #print('Finite part off')
            doFinitePart=0;
            
        forceDs = np.reshape(forceDs,(self._Nfib*self._NXpf,3));       
        
        # Calculate finite part velocity
        thist=time.time();
        finitePart = self._FibColCpp.FinitePartVelocity(X_nonLoc, forceDs,self._fiberDisc._truRPYMob);
        if (verbose>=0):
            print('Time to eval finite part %f' %(time.time()-thist));
            thist=time.time();
        if (self._nonLocal==4):
            return doFinitePart*finitePart;
        
        # Ewald w/ special quadrature corrections
        if (self._nonLocal==1 or self._nonLocal==2):
            # Update the spatial objects for Chebyshev and uniform
            self._SpatialCheb.updateSpatialStructures(X_nonLoc,Dom);
            # Multiply by the weights to get force from force density. 
            forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(self._totnum,1));
            RPYVelocity = RPYEval.calcBlobTotalVel(X_nonLoc,forces,Dom,self._SpatialCheb,self._nThreads);
            if (verbose>=0):
                print('Ewald time %f' %(time.time()-thist));
                thist=time.time();
            SelfTerms = self._FibColCpp.SubtractAllRPY(X_nonLoc,forceDs,self._fiberDisc.getw());
            RPYVelocity-= SelfTerms;
        else: # Direct quad upsampled
            fupsampled = self.getPointsForUpsampledQuad(forceDs);
            forcesUp = fupsampled*np.reshape(np.tile(self._fiberDisc.getwDirect(),self._Nfib),(self._Ndirect*self._Nfib,1));
            if (verbose>=0):
                print('Upsampling time %f' %(time.time()-thist));
                thist=time.time();
            RPYVelocityUp = RPYEval.calcBlobTotalVel(self._Xupsampled,forcesUp,Dom,self._SpatialDirectQuad,self._nThreads);
            if (verbose>=0):
                print('Upsampled Ewald time %f' %(time.time()-thist));
                thist=time.time();
            SelfTerms = self.GPUSubtractSelfTerms(self._Ndirect,self._Xupsampled,forcesUp);
            #SelfTerms2 = self._FibColCpp.SubtractAllRPY(self._Xupsampled,fupsampled,self._fiberDisc.getwDirect());
            #print('Difference bw Raul and me %f' %(np.amax(np.abs(SelfTerms-SelfTerms2))))
            if (verbose>=0):
                print('Self time %f' %(time.time()-thist));
                thist=time.time();
            RPYVelocityUp -= SelfTerms;
            RPYVelocity = self.getValuesfromDirectQuad(RPYVelocityUp);
            if (verbose>=0):
                print('Downsampling time %f' %(time.time()-thist));
            if (np.any(np.isnan(RPYVelocity))):
                raise ValueError('Velocity is nan - stability issue!') 
            return np.reshape(RPYVelocity,self._totnum*3)+doFinitePart*finitePart;

        # Corrections
        thist=time.time();
        alltargets=[];
        numTbyFib=[];
        self._uniPoints = self.getUniformPoints(X_nonLoc);
        self._SpatialUni.updateSpatialStructures(self._uniPoints,Dom);
        alltargets,numTbyFib = self.determineQuadLists(X_nonLoc,self._uniPoints,Dom);
        if (verbose>=0):
            print('Determine quad corrections time %f' %(time.time()-thist));
            thist=time.time();
        corVels = self._FibColCpp.CorrectNonLocalVelocity(X_nonLoc,self._uniPoints,forceDs,finitePart,Dom.getg(),numTbyFib,alltargets)
        RPYVelocity+=corVels;
        if (verbose>=0):
            print('Correction quadrature time %f' %(time.time()-thist));
            print('Maximum correction velocity %f' %np.amax(np.abs(corVels)));

        if (np.any(np.isnan(RPYVelocity))):
            raise ValueError('Velocity is nan - stability issue!') 
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(RPYVelocity,self._totnum*3)+doFinitePart*finitePart;
         
    def updateLambdaAlpha(self,lamalph,Xsarg):
        """
        Update the lambda and alphas after the solve is complete
        """
        self._lambdas = lamalph[:3*self._Nfib*self._NXpf];
        self._alphas = lamalph[3*self._Nfib*self._NXpf:];
           
    def updateAllFibers(self,dt,XsforNL):
        """
        Update the fiber configurations, assuming self._alphas has been computed above. 
        Inputs: dt = timestep
        XsforNL = unnecessary legacy argument
        """
        CppXsMPX = self._FibColCpp.RodriguesRotations(self._tanvecs,self._Midpoints,self._alphas,dt,1e-10);   
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
        return NumbaColloc.useNumbatoUpsample(chebpts,self._Nfib,self._Npf,self._Ndirect,self._fiberDisc._MatfromNtoDirectN);
        b= self._FibColCpp.upsampleForDirectQuad(chebpts);
        
    def getValuesfromDirectQuad(self,upsampledVals):
        """
        Obtain values at Chebyshev nodes from upsampled 
        Inputs: upsampled values as a (nPtsUpsample*Nfib x 3) array.
        Outputs: vals at cheb pts as a (tot#ofpts x 3) array
        """
        return NumbaColloc.useNumbatoDownsample(upsampledVals,self._Nfib,self._Npf,self._Ndirect,self._fiberDisc._MatfromDirectNtoN);
        b = self._FibColCpp.downsampleFromDirectQuad(upsampledVals);
        
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
      
    def getaRPY(self):
        return aRPYFac;
    
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
        
    def getSysDimension(self):
        """
        Dimension of (lambda,alpha) sysyem for GMRES
        """
        return self._Nfib*(self._Npf*3+2*self._nPolys+3);
    
    def updateFiberObjects(self):
        for iFib in range(self._Nfib): # pass to fiber object
            fib = self._fibList[iFib];
            fib.passXsandX(self._tanvecs[rowinds,:],self._Midpoints[iFib,:])
    
    def initializeLambdaForNewFibers(self,newfibs,exForceDen,t,dt,implic_coeff,other=None):
        """
        Solve local problem to initialize values of lambda on the fibers
        Inputs: newfibs = list of replaced fiber indicies, 
        exForceDen = external (gravity/CL) force density on those fibers, 
        t = system time, dt = time step, implic_coeff = implicit coefficient (usually 1/2), 
        other = the other (previous time step) fiber collection
        """
        fD = self._fiberDisc;
        if (fD._truRPYMob):
            raise NotImplementedError('RPY kernel not implemented to respawn lambda for turned-over fibers')
        for iFib in newfibs:
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            X = np.reshape(self._ptsCheb[rowinds,:],3*self._Npf);
            Xs = np.reshape(self._tanvecs[rowinds,:],3*self._Npf);
            fnBend = fD.calcfE(X);
            fEx = exForceDen[stackinds];
            #print('Max CL force on new fiber %f' %np.amax(np.abs(fEx)))
            Local = self.calcLocalVelocities(Xs,fnBend+fEx);
            U0 = self.evalU0(self._ptsCheb[rowinds,:],t);
            RHS = np.concatenate((Local+U0,np.zeros(2*self._nPolys+3)));
            lamalph =  NumbaColloc.linSolveAllFibersForGM(1,self._nPolys,self._Npf,RHS, self._tanvecs[rowinds,:],Xs,dt*implic_coeff, \
                fD._leadordercs, self._mu, fD._MatfromNto2N, fD._UpsampledChebPolys, fD._WeightedUpsamplingMat,fD._LeastSquaresDownsampler, \
                fD._Dpinv2N, fD._D4BC,fD._I,fD._wIt,self._ptsCheb[rowinds,:],fD.getFPMatrix(),fD._Dmat,fD._s,0);
            self._lambdas[stackinds] = lamalph[:3*self._Npf];
            if other is not None:
                other._lambdas[stackinds] = lamalph[:3*self._Npf];    
                
    
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
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            X, Xs = self._fibList[iFib].getXandXs();
            self._ptsCheb[rowinds,:] = X;
            self._tanvecs[rowinds,:] = Xs;
            self._lambdas[stackinds] = 0;
            self._alphas[(2*self._nPolys+3)*(iFib-1):(2*self._nPolys+3)*iFib] = 0;
            self._velocities[stackinds] = 0;
            if (other is not None): # if previous fibCollection is supplied reset
                other._ptsCheb[rowinds,:] = X;
                other._tanvecs[rowinds,:] = Xs;
                other._lambdas[stackinds] = 0;
                other._alphas[(2*self._nPolys+3)*(iFib-1):(2*self._nPolys+3)*iFib] = 0;
                other._velocities[stackinds] = 0; 
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
    def GPUSubtractSelfTerms(self,Ndir,Xupsampled,fupsampled):
        """
        Subtract the self RPY integrals on each fiber
        Inputs: Ndir = number of points for direct quad,
        Xupsampled = upsampled positions, fupsampled = upsampled forces
        Return thr direct RPY velocities
        """
        hydrodynamicRadius = aRPYFac*self._epsilon*self._Lf;
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
        
    def calcLocalVelocities(self,X,forceDsAll):
        """
        Compute the LOCAL velocity on each fiber. 
        Inputs: Xs = positions as a (tot#ofpts*3) 1D array
        forceDsAll = the force densities respectively on the fibers as a 1D array
        Outputs: the velocity M_loc*f on all fibers as a 1D array
        """
        return self._FibColCpp.evalLocalVelocities(X,forceDsAll,self._fiberDisc._truRPYMob)
        
    def KProductsAllFibers(self,Xsarg,alphas,lambdas):
        """
        Compute the products K*alpha and K^T*lambda.
        Inputs: Xsarg = tangent vectors as a totnum x 3 array, 
        alphas as a 1D array, lambdas as a 1D array
        Outputs: products Kalpha, K^*lambda
        """
        fD = self._fiberDisc;
        Kalphs2, Kstlam2 = NumbaColloc.calcKAlphasAndKstarLambda(self._Nfib,self._Npf,self._nPolys,Xsarg, fD._MatfromNto2N, fD._UpsampledChebPolys,\
            fD._LeastSquaresDownsampler,fD._WeightedUpsamplingMat, fD._Dpinv2N, fD._I, fD._wIt,alphas,lambdas)
        return Kalphs2, Kstlam2;
           
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
    
    def evalBendForceDensity(self,X_nonLoc):
        """
        Evaluate the bending FORCE (not density)
        Inputs: X to evaluate at
        Outputs: the forceDensities -EX_ssss also as a (tot#ofpts) x 3 array
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

class SemiflexiblefiberCollection(fiberCollection):

    def __init__(self,nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, kbT,eigValThres,nThreads=1):
        super().__init__(nFibs,turnovertime, fibDisc,nonLocal,mu,omega,gam0,Dom, nThreads);
        self._kbT = kbT;
        self._iT = 0;
        self._eigValThres = eigValThres;
        
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,Xs_nonLoc,t,exForceDen,lamstar,Dom,RPYEval,FPimplicit=0,returnForce=False):
        """
        """
        self._exForceDen = exForceDen;
        return np.zeros(3*self._Nfib*self._NXpf);
    
    def BlockDiagPrecond(self,b,Xs_nonLoc,dt,implic_coeff,X_nonLoc,ModifyBE=1,doFP=0):
        """
        Block diagonal preconditioner for GMRES. 
        b = RHS vector. Xs_nonLoc = tangent vectors as an Npts x 3 array. 
        dt = timestep, implic_coeff = implicit coefficient, X_nonLoc = fiber positions as 
        an Npts x 3 array, doFP = whether to do finite part implicitly. 
        The preconditioner is 
        [-M^Local K-impco*dt*M^Local*F*K; K^* 0]
        """
        self._iT+=1;
        b1D = np.reshape(b,len(b));
        RandVec1 = np.random.randn(3*self._Nfib*self._NXpf);
        RandVec2 = np.random.randn(3*self._Nfib*self._NXpf);
        #np.savetxt('RandVec1_'+str(self._iT)+'.txt',RandVec1);
        #np.savetxt('RandVec2_'+str(self._iT)+'.txt',RandVec2);
        #np.savetxt('ExForceDen_'+str(self._iT)+'.txt',self._exForceDen);
        U0 = np.zeros(3*self._Nfib*self._NXpf);
        #np.savetxt('X_'+str(self._iT)+'.txt',X_nonLoc);
        #np.savetxt('Xs_'+str(self._iT)+'.txt',Xs_nonLoc);
        return self._FibColCpp.applyThermalPreconditioner(X_nonLoc,Xs_nonLoc,self._Midpoints,\
            U0,self._exForceDen,self._fiberDisc._BendMatX0,RandVec1,RandVec2, dt,implic_coeff, \
            ModifyBE,doFP,self._fiberDisc._truRPYMob,self._kbT,1e-10,self._eigValThres);
        
    
