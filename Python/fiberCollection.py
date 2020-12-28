import numpy as np
from DiscretizedFiber import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial
import ManyFiberMethods as ManyFibCpp
import FiberUpdateNumba as NumbaColloc
import numba as nb
import scipy.sparse as sp
import time
from math import sqrt, exp

# Definitions
FattenFib = False;          # make fiber fatter to cap mobility in nonlocal hydrodynamics
upsampneeded = 0.15*1.05;   # upsampneeded*Lfiber = distance where upsampling is needed
specneeded = 0.06*1.20;     #specneed*Lfiber = distance where special quad is needed
rho_crit = 1.114;           # critical Bernstein radius for special quad (set for 3 digits using 32 points)
aRPYFac = exp(1.5)/4        # equivalent RPY blob radius a = aRPYFac*epsilon*L;
dstarCenterLine = 2*aRPYFac;      # dstarCenterLine*eps*L is distance where we set v to centerline v
dstarInterpolate = 4*aRPYFac;     # dstarInterpolate*eps*L is distance where we start blending centerline and quadrature result
dstar2panels = 8.8;         #dstar2panels*eps*L is distance below which we need 2 panels for special quad
verbose = -1;               # debug / timings


class fiberCollection(object):

    """
    This is a class that operates on a list of fibers together. Its main role
    is to compute the nonlocal hydrodynamics between fibers. 
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,Nfibs,fibDisc,nonLocal,mu,omega,gam0,Dom, nThreads=1):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, fibDisc = discretization object that each fiber will get
        a copy of, nonLocal = are we doing nonlocal hydro? (0 = local drag, 1 = full hydrodynamics, 
        2 = hydrodynamics without finite part, 3 = hydrodynamics without special quad, 4 = 
        hydrodynamics without other fibers (only finite part))
        mu = fluid viscosity,omega = frequency of background flow oscillations, 
        gam0 = base strain rate of the background flow, Dom = Domain object to initialize the SpatialDatabase objects, 
        nThreads = number of OMP threads for parallel calculations
        """
        self._Nfib = Nfibs;
        self._fiberDisc = fibDisc;
        self._Npf = self._fiberDisc.getN();
        self._Nunifpf = self._fiberDisc.getNumUniform();
        self._Ndirect = self._fiberDisc.getNumDirect();
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._epsilon, self._Lf = self._fiberDisc.getepsilonL();
        self._nThreads = nThreads;
        self.initPointForceVelocityArrays(Dom);
        
        # Initialize C++
        DLens = Dom.getPeriodicLens();
        for iD in range(len(DLens)):
            if (DLens[iD] is None):
                DLens[iD] = 1e99;
        if (FattenFib):
            print('Fattening fiber')
            rfat = 4*self._epsilon*self._Lf; # the radius at which we cap the mobility
            dstarCL = rfat;
            dstarInt = 2*rfat;
            if (dstarInt > specneeded*self._Lf):
                raise ValueError('CL blending distance has to be less than special quad distance');
            ManyFibCpp.initFiberVars(mu,DLens,self._epsilon, self._Lf,aRPYFac,Nfibs,self._Npf,self._Nunifpf,fibDisc._delta,rfat);
            ManyFibCpp.initSpecQuadParams(rho_crit,dstarCL,dstarInt,dstar2panels,upsampneeded,specneeded);
        else:
            r = self._epsilon*self._Lf;
            ManyFibCpp.initFiberVars(mu,DLens,self._epsilon, self._Lf,aRPYFac,Nfibs,self._Npf,self._Nunifpf,fibDisc._delta,r);
            dstarCL = dstarCenterLine*self._epsilon*self._Lf;
            dstarInt = dstarInterpolate*self._epsilon*self._Lf;
            dstar2 = dstar2panels*self._epsilon*self._Lf
            ManyFibCpp.initSpecQuadParams(rho_crit,dstarCL,dstarInt,dstar2,upsampneeded,specneeded);
        
        ManyFibCpp.initNodesandWeights(fibDisc.gets(), fibDisc.getwDirect(),fibDisc.getSpecialQuadNodes(),fibDisc.getUpsampledWeights(),\
            np.vander(fibDisc.getSpecialQuadNodes(),increasing=True).T);
        ManyFibCpp.initResamplingMatrices(fibDisc.getUpsamplingMatrix(), fibDisc.get2PanelUpsamplingMatrix(),\
                    fibDisc.getValstoCoeffsMatrix());
        ManyFibCpp.initFinitePartMatrix(fibDisc.getFPMatrix(), fibDisc.getDiffMat());
    
    def initFibList(self,fibListIn, Dom,pointsfileName=None,tanvecfileName=None):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
        Inputs: list of Discretized fiber objects (typically empty), Domain object
        """
        self._fibList = fibListIn;
        if (tanvecfileName is None): # initialize straight fibers
            for jFib in range(self._Nfib):
                if self._fibList[jFib] is None:
                    # Initialize memory
                    self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                    # Initialize straight fiber positions at t=0
                    self._fibList[jFib].initFib(Dom.getLens());
                    
        elif (pointsfileName is None): # Read initial tangent vectors from file
            AllXs = np.loadtxt(tanvecfileName)
            for jFib in range(self._Nfib): 
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                XsThis = AllXs[self.getRowInds(jFib),:];
                self._fibList[jFib].initFib(Dom.getLens(),Xs=XsThis[:,np.random.permutation(3)]);
                
        else: # read both locations and tangent vectors from file 
            AllX = np.loadtxt(pointsfileName)
            AllXs = np.loadtxt(tanvecfileName)
            for jFib in range(self._Nfib): 
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                XsThis = AllXs[self.getRowInds(jFib),:];
                XThis = AllX[self.getRowInds(jFib),:];
                self._fibList[jFib].initFib(Dom.getLens(),X=XThis,Xs=XsThis);
    
    def initPointForceVelocityArrays(self, Dom):
        """
        Method to initialize the memory for lists of points, tangent 
        vectors and lambdas for the fiber collection
        Input: Dom = Domain object 
        """
        # Establish large lists that will be used for the non-local computations
        self._totnum = self._Nfib*self._Npf;                          
        self._ptsCheb=np.zeros((self._totnum,3));                     
        self._tanvecs=np.zeros((self._totnum,3));                     
        self._lambdas=np.zeros((self._totnum*3));                     
        self._alphas = np.zeros((2*self._Npf+1)*self._Nfib);
        self._velocities = np.zeros(self._Npf*self._Nfib*3);         
        # Initialize the spatial database objects
        self._SpatialCheb = ckDSpatial(self._ptsCheb,Dom);
        self._SpatialDirectQuad = ckDSpatial(np.zeros((self._Nfib*self._Ndirect,3)),Dom);
        self._SpatialUni = ckDSpatial(np.zeros((self._Nunifpf*self._Nfib,3)),Dom);
    
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,Xs_nonLoc,t,exForceDen,lamstar,Dom,RPYEval,FPimplicit=0):
        """
        RHS for the block diagonal GMRES system. 
        Inputs: X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, t = system time, exForceDen = external force density (treated explicitly), 
        lamstar = lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, FPimplicit = whether to do the finite part integral implicitly. 
        The block diagonal RHS is 
        [M^Local*(F*X^n+f^ext) + M^NonLocal * (F*X^(n+1/2,*)+lambda*+f^ext)+u_0; -w^T * F*X^n]
        """
        fnBend = self.evalBendForceDensity(self._ptsCheb);
        Local = self.calcLocalVelocities(np.reshape(Xs_nonLoc,self._totnum*3),exForceDen+fnBend);
        U0 = self.evalU0(X_nonLoc,t);
        fNLBend = self.evalBendForceDensity(X_nonLoc);
        nonLocal = self.nonLocalVelocity(X_nonLoc,Xs_nonLoc,lamstar+exForceDen+fNLBend,Dom,RPYEval,1-FPimplicit);
        if (FPimplicit==1 and self._nonLocal): # do the finite part separetely if it's to be treated implicitly
            nonLocal+= ManyFibCpp.FinitePartVelocity(X_nonLoc, fnBend, Xs_nonLoc)
            print('Doing FP separate, it''s being treated implicitly')
        return np.concatenate((Local+nonLocal+U0,np.zeros((2*self._Npf+1)*self._Nfib)));
    
    def calcNewRHS(self,BlockDiagAnswer,X_nonLoc,Xs_nonLoc,dtimpco,lamstar, Dom, RPYEval,FPimplicit=0):
        """
        New RHS (after block diag solve). This is the residual form of the system that gets 
        passed to GMRES. 
        Inputs: lamalph = the answer for lambda and alpha from the block diagonal solver, 
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
        nonLocal = self.nonLocalVelocity(X_nonLoc,Xs_nonLoc,fbendCorrection+lamCorrection,Dom,RPYEval,1-FPimplicit);
        return np.concatenate((nonLocal,np.zeros((2*self._Npf+1)*self._Nfib)));
        
    def Mobility(self,lamalph,impcodt,X_nonLoc,Xs_nonLoc,Dom,RPYEval):
        """
        Mobility calculation for GMRES
        Inputs: lamalph = the input lambdas and alphas
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, impcodt = delta t * implicit coefficient,
        Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms, 
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
        return np.concatenate((FirstBlock,SecondBlock));
       
    def BlockDiagPrecond(self,b,Xs_nonLoc,dt,implic_coeff,X_nonLoc,doFP=0):
        """
        Block diagonal preconditioner for GMRES. 
        b = RHS vector. Xs_nonLoc = tangent vectors as an Npts x 3 array. 
        dt = timestep, implic_coeff = implicit coefficient, X_nonLoc = fiber positions as 
        an Npts x 3 array, doFP = whether to do finite part implicitly. 
        The preconditioner is 
        [-M^Local K-impco*dt*M^Local*F*K; K^* impco*w^T*F*K]
        """
        XsAll = np.reshape(Xs_nonLoc,self._Npf*self._Nfib*3);
        fD = self._fiberDisc;
        b1D = np.reshape(b,len(b));
        lamalph = NumbaColloc.linSolveAllFibersForGM(self._Nfib,self._Npf,b1D, Xs_nonLoc,XsAll,dt*implic_coeff, \
            fD._leadordercs, self._mu, fD._MatfromNto2N, fD._UpsampledChebPolys, fD._WeightedUpsamplingMat,fD._LeastSquaresDownsampler, fD._Dpinv2N, \
            fD._D4BC,fD._I,fD._wIt,X_nonLoc,fD.getFPMatrix(),fD._Dmat,fD._s,doFP);
        return lamalph;
        
    def nonLocalVelocity(self,X_nonLoc,Xs_nonLoc,forceDs, Dom, RPYEval,doFinitePart=1):
        """
        Compute the non-local velocity due to the fibers.
        Inputs: Arguments X_nonLoc, Xs_nonLoc = fiber positions and tangent vectors as Npts x 3 arrays, 
        forceDs = force densities as an 3*npts 1D numpy array
        Dom = Domain object the computation happens on,  
        RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities, doFinitePart = whether to 
        include the finite part integral in the calculation 
        Outputs: nonlocal velocity as a tot#ofpts*3 one-dimensional array.
        """
        # If not doing nonlocal hydro, return nothing
        noSpecialPart = False;
        otherWt = 1.0;
        if (self._nonLocal==0):
            return np.zeros(self._totnum*3);
        elif (self._nonLocal==2):
            #print('Finite part off')
            doFinitePart=0;
        elif (self._nonLocal==3):
            #print('Special quad off')
            noSpecialPart = True;
        elif (self._nonLocal==4):
            #print('No other fibs')
            otherWt=0.0;
            
        forceDs = np.reshape(forceDs,(self._totnum,3));
        
        # Calculate finite part velocity
        thist=time.time();
        finitePart = ManyFibCpp.FinitePartVelocity(X_nonLoc, forceDs, Xs_nonLoc);
        if (verbose>=0):
            print('Time to eval finite part %f' %(time.time()-thist));
        if (self._nonLocal==4):
            return finitePart;
        
        # Update the spatial objects for Chebyshev and uniform
        self._SpatialCheb.updateSpatialStructures(X_nonLoc,Dom);
        self._uniPoints = self.getUniformPoints(X_nonLoc);
        self._SpatialUni.updateSpatialStructures(self._uniPoints,Dom);
        # Ewald splitting to compute the far field and near field
        # Multiply by the weights to get force from force density. 
        thist=time.time();
        #forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(self._totnum,1));
        #RPYVelocity = RPYEval.calcBlobTotalVel(X_nonLoc,forces,Dom,self._SpatialCheb,self._nThreads);
         # Upsampled version
        Xupsampled = self.getPointsForUpsampledQuad(X_nonLoc);
        self._SpatialDirectQuad.updateSpatialStructures(Xupsampled,Dom);
        fupsampled = self.getPointsForUpsampledQuad(forceDs);
        forcesUp = fupsampled*np.reshape(np.tile(self._fiberDisc.getwDirect(),self._Nfib),(self._Ndirect*self._Nfib,1));
        RPYVelocityUp = RPYEval.calcBlobTotalVel(Xupsampled,forcesUp,Dom,self._SpatialDirectQuad,self._nThreads);
        SelfTerms = ManyFibCpp.SubtractAllRPY(Xupsampled,fupsampled,self._fiberDisc.getwDirect());
        RPYVelocityUp -= SelfTerms;
        RPYVelocity = self.getValuesfromDirectQuad(RPYVelocityUp);
        if (verbose>=0):
            print('Total Ewald time %f' %(time.time()-thist));
        
        thist=time.time();
        alltargets=[];
        numTbyFib=[];
        if (self._nonLocal < 3): # Correction quadrature
            alltargets,numTbyFib = self.determineQuadLists(X_nonLoc,self._uniPoints,Dom);
            if (verbose>=0):
                print('Determine quad corrections time %f' %(time.time()-thist));
                thist=time.time();
            corVels = ManyFibCpp.CorrectNonLocalVelocity(X_nonLoc,self._uniPoints,forceDs,finitePart,Dom.getg(),\
                    numTbyFib,alltargets,self._nThreads)
            RPYVelocity+=corVels;
            if (verbose>=0):
                print('Correction quadrature time %f' %(time.time()-thist));
                print('Maximum correction velocity %f' %np.amax(np.abs(corVels)));

        if (np.any(np.isnan(RPYVelocity))):
            raise ValueError('Velocity is nan - stability issue!') 
        # Return the velocity due to the other fibers + the finite part integral velocity
        return otherWt*np.reshape(RPYVelocity,self._totnum*3)+doFinitePart*finitePart;
         
    def updateLambdaAlpha(self,lamalph,Xsarg):
        """
        Update the lambda and alphas after the solve is complete
        """
        self._lambdas = lamalph[:3*self._Nfib*self._Npf];
        self._alphas = lamalph[3*self._Nfib*self._Npf:];
        fD = self._fiberDisc;
        self._velocities = NumbaColloc.calcKAlphas(self._Nfib,self._Npf,Xsarg,fD._MatfromNto2N, fD._UpsampledChebPolys, \
            fD._LeastSquaresDownsampler, fD._Dpinv2N, fD._I,self._alphas);
        
    def updateAllFibers(self,dt,XsforNL,exactinex=1):
        """
        Update the fiber configurations, assuming self._alphas has been computed above. 
        Inputs: dt = timestep, XsforNL = the tangent vectors we use to compute the 
        Rodriguez rotation, exactinex = whether to preserve exact inextensibility
        """
        t=time.time()
        if (exactinex==0): # not exact inextensibility. Just update by K*alpha
            for iFib in range(self._Nfib):
                stackinds = self.getStackInds(iFib);
                rowinds = self.getRowInds(iFib);
                fib = self._fibList[iFib];
                XsforOmega = np.reshape(XsforNL[rowinds,:],self._Npf*3);
                alphaInds = range(iFib*(2*self._Npf+1),(iFib+1)*(2*self._Npf+1));
                fib.updateXsandX(dt,self._velocities[stackinds],XsforOmega,self._alphas[alphaInds],exactinex);
        else: # exact inextensibility with Rodriguez rotation. Compute omega, then rotate
            fD = self._fiberDisc;
            AllXs, AllX = NumbaColloc.updateXsNumba(self._Nfib,self._Npf,self._ptsCheb,\
                self._tanvecs,XsforNL,fD._MatfromNto2N, fD._Matfrom2NtoN,\
                fD._Lmat,fD._Dmat,np.reshape(self._velocities,(self._Npf*self._Nfib,3)),dt,self._Lf)
            for iFib in range(self._Nfib): # pass to fiber object
                fib = self._fibList[iFib];
                rowinds = self.getRowInds(iFib);
                fib.passXsandX(AllX[rowinds,:],AllXs[rowinds,:])
        if (verbose>=0):
            print('Rodrigez time %f' %(time.time()-t));
        return np.amax(AllX);
        
    def fillPointArrays(self):
        """
        Copy the X and Xs arguments from self._fibList (list of fiber
        objects) into large (tot#pts x 3) arrays that are stored in memory
        """
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            fib = self._fibList[iFib];
            X, Xs = fib.getXandXs();
            self._ptsCheb[rowinds,:] = X;
            self._tanvecs[rowinds,:] = Xs;
    
    def getX(self):
        return self._ptsCheb;

    def getXs(self):
        return self._tanvecs;
    
    def getLambdas(self):
        return self._lambdas; 
    
    @staticmethod
    @nb.njit((nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],\
        nb.int64,nb.int64,nb.float64[:]),parallel=True,cache=True)
    def FiberStress(X, fbend,lams,Lens,Nfib,N,w):
        """
        Compute the stress in the suspension due to the fiber via Batchelor's formula
        Inputs: lams = constraint forces on the fibers as a totnum*3 1D numpy array, 
        Lens = 3 array of periodic lengths (to compute volume for the stress)
        Output: the 3 x 3 stress tensor 
        """
        stressLams = np.zeros((3,3));
        stressFB = np.zeros((3,3)); # from bending 
        for iPt in nb.prange(N*Nfib):
            wt = w[iPt%N];
            # Ugly but faster than np.outer 
            stressLams-= np.array([[X[iPt,0]*lams[3*iPt], X[iPt,0]*lams[3*iPt+1], X[iPt,0]*lams[3*iPt+2]],\
                               [X[iPt,1]*lams[3*iPt], X[iPt,1]*lams[3*iPt+1], X[iPt,1]*lams[3*iPt+2]],\
                               [X[iPt,2]*lams[3*iPt], X[iPt,2]*lams[3*iPt+1], X[iPt,2]*lams[3*iPt+2]]])*wt;
            stressFB-= np.array([[X[iPt,0]*fbend[3*iPt], X[iPt,0]*fbend[3*iPt+1], X[iPt,0]*fbend[3*iPt+2]],\
                               [X[iPt,1]*fbend[3*iPt], X[iPt,1]*fbend[3*iPt+1], X[iPt,1]*fbend[3*iPt+2]],\
                               [X[iPt,2]*fbend[3*iPt], X[iPt,2]*fbend[3*iPt+1], X[iPt,2]*fbend[3*iPt+2]]])*wt;
        stressLams/=np.prod(Lens);
        stressFB/=np.prod(Lens);
        return stressLams, stressFB;
           
    def uniformForce(self,strengths):
        """
        A uniform force density on all fibers with strength strength 
        """
        return np.tile(strengths,self._Nfib*self._Npf);
    
    def getUniformPoints(self,chebpts):
        """
        Obtain uniform points from a set of Chebyshev points. 
        Inputs: chebpts as a (tot#ofpts x 3) array
        Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
        """
        uniPoints = np.zeros((self._Nfib*self._Nunifpf,3));
        for iFib in range(self._Nfib):
            uniinds = range(iFib*self._Nunifpf,(iFib+1)*self._Nunifpf);
            rowinds = self.getRowInds(iFib);
            uniPoints[uniinds]=self._fiberDisc.resampleUniform(chebpts[rowinds]);
        return uniPoints;

    def getPointsForUpsampledQuad(self,chebpts):
        """
        Obtain upsampled points from a set of Chebyshev points. 
        Inputs: chebpts as a (tot#ofpts x 3) array
        Outputs: upsampled points as a (nPtsUpsample*Nfib x 3) array.
        """
        upsampledPoints = np.zeros((self._Nfib*self._Ndirect,3));
        for iFib in range(self._Nfib):
            upinds = range(iFib*self._Ndirect,(iFib+1)*self._Ndirect);
            rowinds = self.getRowInds(iFib);
            upsampledPoints[upinds]=self._fiberDisc.resampleForDirectQuad(chebpts[rowinds]);
        return upsampledPoints;
       
    def getValuesfromDirectQuad(self,upsampledVals):
        """
        Obtain values at Chebyshev nodes from upsampled 
        Inputs: upsampled values as a (nPtsUpsample*Nfib x 3) array.
        Outputs: vals at cheb pts as a (tot#ofpts x 3) array
        """
        chebValues = np.zeros((self._Nfib*self._Npf,3));
        for iFib in range(self._Nfib):
            upinds = range(iFib*self._Ndirect,(iFib+1)*self._Ndirect);
            rowinds = self.getRowInds(iFib);
            chebValues[rowinds]=self._fiberDisc.downsampleFromDirectQuad(upsampledVals[upinds]);
        return chebValues;        
          
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
        
    def getSysDimension(self):
        """
        Dimension of (lambda,alpha) sysyem for GMRES
        """
        return self._Nfib*self._Npf*5+self._Nfib;

    def writeFiberLocations(self,of):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        for fib in self._fibList:
            fib.writeLocs(of);  
    
    def writeFiberTangentVectors(self,of):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        for fib in self._fibList:
            fib.writeTanVecs(of);  
            
    ## ====================================================
    ##  "PRIVATE" METHODS NOT ACCESSED OUTSIDE OF THIS CLASS
    ## ====================================================
    def calcLocalVelocities(self,Xs,forceDsAll):
        """
        Compute the LOCAL velocity on each fiber. 
        Inputs: Xs = tangent vectors as a (tot#ofpts*3) 1D array
        forceDsAll = the force densities respectively on the fibers as a 1D array
        Outputs: the velocity M_loc*f on all fibers as a 1D array
        """
        if False: # pure python 
            LocalOnly = np.zeros(len(Xs_nonLoc));
            for iFib in range(self._Nfib):
                stackinds = self.getStackInds(iFib);
                XsiFib = Xs[stackinds];
                forceD = forceDsAll[stackinds];
                LocalOnly[stackinds] = self._fiberDisc.calcLocalVelocity(XsiFib,forceD);
                
        LocalOnly = NumbaColloc.calcLocalVelocities(Xs,forceDsAll,self._fiberDisc._leadordercs,self._mu,self._Npf,self._Nfib);
        return LocalOnly;
        
    def determineQuadLists(self, XnonLoc, Xuniform, Dom):
        """
        Determine which (target, fiber) pairs are close to each other 
        This method uses the uniform points to determine if a fiber is close to a target via
        looking at the uniform points which are neighbors of the Chebyshev points. 
        Inputs: XnonLoc, Xuniform = tot#ofpts x 3 arrays of Chebyshev and uniform points on 
        all the fibers, Dom = domain object we are doing computation on.
        Outputs: two numpy arrays. alltargets = sequential list of targets that need correcting
        numByFib = number of targets that need correction for each fiber. This is enough 
        information to match target-fiber pairs
        """
        qUpsamplecut = upsampneeded*self._Lf; # cutoff for needed to correct Ewald
        # Get the neighbors from the SpatialDatabase
        t = time.time();
        # Determine which Chebyshev points are neighbors of the uniform points (fibers)
        neighborList = self._SpatialUni.otherNeighborsList(self._SpatialCheb, qUpsamplecut);
        if (verbose>=0):
            print('Neighbor search uniform-Cheb time %f' %(time.time()-t))
        t = time.time();
        alltargets=[];
        numByFib = np.zeros(self._Nfib,dtype=int);
        for iFib in range(self._Nfib):
            # For each fiber, obtain the targets that are nearby and not on this fiber
            # Loop through neighbor list Nunifpf points at a time and find the unique targets that 
            # are close to this fiber 
            fibTargets=list(set([iT for sublist in neighborList[iFib*self._Nunifpf:(iFib+1)*self._Nunifpf] \
                for iT in sublist if iT//self._Npf!=iFib]));
            numByFib[iFib]=len(fibTargets);
            alltargets.extend(fibTargets); # all the targets for each uniform point
        if (verbose>=0):
            print('Organize targets & fibers time %f' %(time.time()-t))
        return alltargets, numByFib;
    
    def KProductsAllFibers(self,Xsarg,alphas,lambdas):
        """
        Compute the products K*alpha and K^T*lambda for. 
        Inputs: Xsarg = tangent vectors as a totnum x 3 array, 
        alphas as a 1D array, lambdas as a 1D array
        Outputs: products Kalpha, K^*lambda
        """
        fD = self._fiberDisc;
        Kalphs2, Kstlam2 = NumbaColloc.calcKAlphasAndKstarLambda(self._Nfib,self._Npf,Xsarg, fD._MatfromNto2N, fD._UpsampledChebPolys,\
            fD._LeastSquaresDownsampler,fD._WeightedUpsamplingMat, fD._Dpinv2N, fD._I, fD._wIt,alphas,lambdas)
        return Kalphs2, Kstlam2;
        # pure python
        Kalphs = np.zeros(3*self._Npf*self._Nfib);
        Kstlam = np.zeros((2*self._Npf+1)*self._Nfib);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            Xs = Xsarg[rowinds,:];
            alphainds = range(iFib*(2*self._Npf+1),(iFib+1)*(2*self._Npf+1));
            alphaiFib = alphas[alphainds];
            lamsiFib = lambdas[stackinds];
            Kalphs[stackinds], Kstlam[alphainds] = self._fiberDisc.KalphProduct(Xs,alphaiFib,lamsiFib);
        print(np.amax(np.abs(Kalphs1-Kalphs)))
        print(np.amax(np.abs(Kalphs2-Kalphs)))
        print(np.amax(np.abs(Kstlam2-Kstlam)))
        return Kalphs, Kstlam;
    
    def calcAllh(self,Xin):
        """
        Compute the products w^T*F*X (integrals of the bending force for a given X)
        Input: X as a totnum x 2 array
        Output: the products w^T*F*X
        """
        h = np.zeros((2*self._Npf+1)*self._Nfib);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            h[(iFib+1)*(2*self._Npf+1)-3:(iFib+1)*(2*self._Npf+1)] = \
                self._fiberDisc.calcH(np.reshape(Xin[rowinds,:],self._Npf*3));
        return h;
    
    def getRowInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 2D arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._Npf,(iFib+1)*self._Npf);
        
    def getStackInds(self,iFib):
        """
        Method to get the row indices in any (Nfib*Nperfib*3) long 1D arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*3*self._Npf,(iFib+1)*3*self._Npf);

    def evalU0(self,Xin,t):
        """
        Compute the background flow on input array Xin at time t.
        Xin is assumed to be an N x 3 array with each column being
        the x, y, and z locations. 
        This method returns the background flow in the same format 
        (N x 3).
        """
        U0 = np.zeros(Xin.shape);
        U0[:,0]=self._gam0*np.cos(self._omega*t)*Xin[:,1]; # x velocity is linear shear in y
        return np.reshape(U0,3*self._Npf*self._Nfib);
    
    def evalBendForceDensity(self,X_nonLoc):
        """
        Evaluate the bending force density for a given X (given the 
        known fiber discretization of this class.
        Inputs: X to evaluate -EX_ssss at, as a (tot#ofpts) x 3 array
        Outputs: the forceDensities -EX_ssss also as a (tot#ofpts) x 3 array
        """
        FEMatrix = self._fiberDisc._D4BC;
        forceDs1 = NumbaColloc.EvalAllBendForces(self._Npf,self._Nfib,np.reshape(X_nonLoc,self._Npf*self._Nfib*3),FEMatrix);
        return forceDs1;
        totnum = self._Nfib*self._Npf;
        forceDs=np.zeros(totnum*3);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            Xin = np.reshape(X_nonLoc[rowinds,:],self._Npf*3);
            forceDs[stackinds] = self._fiberDisc.calcfE(Xin);
        print(np.amax(np.abs(forceDs1-forceDs)))
        return forceDs;
     
    def calcCurvatures(self,X):
        """
        Evaluate fiber curvatures on fibers with locations X. 
        Returns an Nfib array of the mean L^2 curvature by fiber 
        """
        Curvatures=np.zeros(self._Nfib);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            Curvatures[iFib] = self._fiberDisc.calcFibCurvature(X[rowinds,:]);
        return Curvatures;
