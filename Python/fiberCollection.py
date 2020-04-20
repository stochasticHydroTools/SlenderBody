import numpy as np
from DiscretizedFiber import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial
import ManyFiberMethods as ManyFibCpp
import scipy.sparse as sp
import time
from math import sqrt

# Definitions
upsampneeded = 0.15*1.05; # upsampneeded*Lfiber = distance where upsampling is needed
specneeded = 0.06*1.20; #specneed*Lfiber = distance where special quad is needed
rho_crit = 1.114; # critical Bernstein radius for special quad
dstarCenterLine = 2.2; # dstarCenterLine*eps*L is distance where we set v to centerline v
dstarInterpolate = 4.4; # dstarInterpolate*eps*L is distance where we start blending centerline
                        # and quadrature result
dstar2panels = 8.8; #dstar2panels*eps*L is distance below which we need 2 panels for special quad

class fiberCollection(object):

    """
    This is a class that operates on a list of fibers together. Its main role
    is to compute the non-local hydrodynamics. 
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,Nfibs,fibDisc,nonLocal,mu,omega,gam0,Dom):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, fibDisc = discretization object that each fiber will get
        a copy of, nonLocal = are we doing nonlocal hydro? (1 for yes, 0 no), mu = fluid viscosity, 
        omega = frequency of background flow oscillations, gam0 = base strain rate of the background flow, 
        Dom = Domain object to initialize the SpatialDatabase objects
        """
        self._Nfib = Nfibs;
        # Initialize fiber discretization. Each fiber gets a copy of the same one!
        self._fiberDisc = fibDisc;
        self._Npf = self._fiberDisc.getN();
        self._Nunifpf = self._fiberDisc.getNumUniform();
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._epsilon, self._Lf = self._fiberDisc.getepsilonL();
        self._aRPY = sqrt(1.5)*self._epsilon*self._Lf;
        # Allocate memory for arrays of forces, force densities, force velocities
        self.initPointForceVelocityArrays(Dom);
        # Initialize C++
        ManyFibCpp.initFiberVars(mu,Dom.getPeriodicLens(),self._epsilon, self._Lf,Nfibs,self._Npf,self._Nunifpf);
        ManyFibCpp.initSpecQuadParams(rho_crit,dstarCenterLine,dstarInterpolate,dstar2panels,upsampneeded,specneeded);
        ManyFibCpp.initNodesandWeights(fibDisc.gets(), fibDisc.getw(),fibDisc.getSpecialQuadNodes(),fibDisc.getUpsampledWeights());
        ManyFibCpp.initResamplingMatrices(fibDisc.getUpsamplingMatrix(), fibDisc.get2PanelUpsamplingMatrix(),\
                    fibDisc.getValstoCoeffsMatrix());
        ManyFibCpp.initFinitePartMatrix(fibDisc.getFPMatrix(), fibDisc.getDiffMat());
    
    def initFibList(self,fibListIn, Dom):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
        Inputs: list of Discretized fiber objects (typically empty), Domain object
        """
        self._fibList = fibListIn;
        for jFib in range(self._Nfib):
           if self._fibList[jFib] is None:
                # Initialize memory
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                # Initialize straight fiber positions at t=0
                self._fibList[jFib].initFib(Dom.getLens());
    
    def initPointForceVelocityArrays(self, Dom):
        """
        Method to initialize the memory for lists of points, tangent 
        vectors and lambdas for the fiber collection
        Input: Dom = Domain object 
        """
        # Establish large lists that will be used for the non-local computations
        totnum = self._Nfib*self._Npf;                          # total # of points
        self._ptsCheb=np.zeros((totnum,3));                     # Chebyshev points
        self._tanvecs=np.zeros((totnum,3));                     # Tangent vectors
        self._lambdas=np.zeros((totnum*3));                     # lambdas that enforce inextensibility
        self._alphas = np.zeros((2*self._Npf-2)*self._Nfib);
        self._velocities = np.zeros(self._Npf*self._Nfib*3);
        # Initialize the spatial database objects
        self._SpatialCheb = ckDSpatial(self._ptsCheb,Dom);
        self._SpatialUni = ckDSpatial(np.zeros((self._Nunifpf*self._Nfib,3)),Dom);
    
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================
    def nonLocalBkgrndVelocity(self,X_nonLoc,Xs_nonLoc,lam_nonLoc, t, exForceDen, Dom, RPYEval):
        """
        Compute the non-local velocity + background flow due to the fibers.
        Inputs: Arguments X_nonLoc, Xs_nonLoc, lam_nonLoc = X, Xs, and lambda for 
        the computation, all as tot#pts x 3 numpy arrays.
        t = current time to evaluate background flow, Dom = Domain object the computation
        happens on, RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities, 
        exForceDen = tot#pts*3 1D numpy array of the external forcing (gravity or cross linking) 
        Outputs: non-local velocity as a tot#ofpts*3 one-dimensional array.
        This includes the background flow and finite part integrals. 
        """
        # Donev: I suggest using spacing to break this down into pieces instead of one long routine.
        # Just like we put paragraphs in writing spacing helps structure code into sections. I did it for this routine
        BkgrndFlow = self.evalU0(X_nonLoc,t);
        # If not doing non-local hydro, return just background flow and stop.
        totnum = self._Nfib*self._Npf;
        if (not self._nonLocal):
            return np.reshape(BkgrndFlow,totnum*3);
            
        # Compute the total force density from lambda, the external force, and also
        # the bending force, which can be determined from X and the fiber discretization. 
        thist=time.time();
        forceDs = np.reshape(self.evalBendForceDensity(X_nonLoc) + exForceDen+lam_nonLoc,(totnum,3));
        # Donev: I suggest that you introduce a global "verbocity" variable which controls how and what is printed to screen
        # For example, -1 means don't print stuff for long "real" runs, 0 means print timing, 1 means print some more, etc.
        # This way you can have code that can run in debug mode and also run in production mode
        print('Time to eval bend forces %f' %(time.time()-thist));
        
        # Calculate finite part velocity
        thist=time.time();
        finitePart = ManyFibCpp.FinitePartVelocity(X_nonLoc, forceDs, Xs_nonLoc);
        print('Time to eval finite part %f' %(time.time()-thist));
        # Donev: I added space here
        # While it does not make sense physically, for testing which terms contribute to the physics it is useful to have an option
        # where we include the finite part but not the fiber-fiber interactions. So make self._nonLocal be an integer not logical with 
        # 0=local, 1=local+FP, >1 all terms or something like that
        
        # Update the spatial objects for Chebyshev and uniform
        self._SpatialCheb.updateSpatialStructures(X_nonLoc,Dom);
        uniPoints = self.getUniformPoints(X_nonLoc);
        self._SpatialUni.updateSpatialStructures(uniPoints,Dom);
        # Ewald splitting to compute the far field and near field
        nThreads=4; # Donev: This variable should be higher up and come in as input or be part of the class and set in Init.
        # Multiply by the weights to get force from force density. 
        thist=time.time();
        forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(totnum,1));
        RPYVelocity = RPYEval.calcBlobTotalVel(X_nonLoc,forces,Dom,self._SpatialCheb,nThreads);
        print('Total Ewald time %f' %(time.time()-thist));
        
        # Corrections for fibers that are close together
        # First determine which sets of fibers and targets need corrections
        thist=time.time();
        alltargets,numTbyFib = self.determineQuadLists(X_nonLoc,uniPoints,Dom);
        print('Determine quad corrections time %f' %(time.time()-thist));
        # Do the corrections
        thist=time.time();
        corVels = ManyFibCpp.CorrectNonLocalVelocity(X_nonLoc,uniPoints,forceDs,finitePart, Dom.getg(),numTbyFib,alltargets, nThreads)
        print('Correction quadrature time %f' %(time.time()-thist));
        print('Maximum correction velocity %f' %np.amax(np.abs(corVels)));
        
        RPYVelocity+=corVels;
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(RPYVelocity+BkgrndFlow,totnum*3)+finitePart;
    
    def linSolveAllFibers(self,XsforNL,nLvel,forceExt,dt,implic_coeff):
        """
        Compute alpha and lambda on all the fibers for a given RHS
        Inputs: XsforNL = nPts x 3 array of tangent vectors, nLvel = nPts*3 one-d array
        of velocities from non-local solve, forceExt = nPts*3 one-d array of force densities, 
        dt = timestep, implic_coeff = implicit coefficient for the matrix solve
        """
        # Linear solve on each fiber
        for iFib in range(self._Nfib):
            fib = self._fibList[iFib];
            stackinds = self.getStackInds(iFib);
            rowinds = self.getRowInds(iFib);
            alphaInds = range(iFib*(2*self._Npf-2),(iFib+1)*(2*self._Npf-2));
            Xin = np.reshape(self._ptsCheb[rowinds,:],self._Npf*3); # X^n is the input, not XnonLoc!!
            Xsin = np.reshape(XsforNL[rowinds,:],self._Npf*3);
            self._alphas[alphaInds], self._velocities[stackinds], self._lambdas[stackinds] = \
                self._fiberDisc.alphaLambdaSolve(Xin,Xsin,dt,implic_coeff,nLvel[stackinds],forceExt[stackinds]);
        
    def updateAllFibers(self,dt,XsforNL,exactinex=1):
        """
        Update the fiber configurations, assuming fixed point is over and self._alphas has 
        been computed above. 
        Inputs: dt = timestep, XsforNL = the tangent vectors we use to compute the 
        Rodriguez rotation, exactinex = whether to preserve exact inextensibility
        """
        for iFib in range(self._Nfib):
            stackinds = self.getStackInds(iFib);
            rowinds = self.getRowInds(iFib);
            fib = self._fibList[iFib];
            XsforOmega = np.reshape(XsforNL[rowinds,:],self._Npf*3);
            alphaInds = range(iFib*(2*self._Npf-2),(iFib+1)*(2*self._Npf-2));
            fib.updateXsandX(dt,self._velocities[stackinds],XsforOmega,self._alphas[alphaInds],exactinex);    

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

    def FiberStress(self, lams,Lens):
        """
        Compute the stress in the suspension due to the fiber via Batchelor's formula
        Inputs: lams = constraint forces on the fibers as a totnum*3 1D numpy array, 
        Lens = 3 array of periodic lengths (to compute volume for the stress)
        Output: the 3 x 3 stress tensor 
        """
        totnum = self._Nfib*self._Npf;
        X = self._ptsCheb;
        # Compute the total force density from lambda, and also
        # the bending force, which can be determined from X and the fiber discretization. 
        forceDs = np.reshape(self.evalBendForceDensity(X) + lams,(totnum,3));
        # Multiply by the weights to get force from force density. 
        forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(totnum,1));
        stress = np.zeros((3,3));
        for iPt in range(totnum):
            stress-= np.outer(X[iPt,:],forces[iPt,:]);
        stress/=np.prod(Lens);
        return stress;
        
    def converged(self, lambdasm1,fptol):
        """
        Check whether any of the fibers are not converged (i.e. whether
        the overall fixed point iteration to determine lambda has converged).
        Inputs: lambdasm1 = Npts*3 one-d vector of lambdas from the prior iteration,
        fptol = tolerance for the fixed point convergence.
        """
        reler = np.linalg.norm(self._lambdas-lambdasm1)/max(1.0,np.linalg.norm(self._lambdas));
        return (reler < fptol); 
    
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
    
    def getUpsampledValues(self,chebpts):
        """
        Obtain upsampled values from a set of values at Chebyshev nodes.
        Inputs: values at Chebyhsev nodes as a (tot#ofpts x 3) array
        Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
        """
        nUpsample = self._fiberDisc.getNumUpsample();
        upsamp = np.zeros((self._Nfib*nUpsample,3));
        upsamp2Pan = np.zeros((self._Nfib*nUpsample*2,3));
        for iFib in range(self._Nfib):
            upsampInds = range(iFib*nUpsample,(iFib+1)*nUpsample);
            upsamp2PanInds = range(iFib*2*nUpsample,(iFib+1)*2*nUpsample);
            rowinds = self.getRowInds(iFib);
            upsamp[upsampInds]=self._fiberDisc.upsampleGlobally(chebpts[rowinds]);
            upsamp2Pan[upsamp2PanInds]=self._fiberDisc.upsample2Panels(chebpts[rowinds]);
        return upsamp, upsamp2Pan;

    def getg(self,t):
        """
        Get the value of g according to what the background flow dictates.
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
    
    def getRowInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._Npf,(iFib+1)*self._Npf);

    def writeFiberLocations(self,of):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        for fib in self._fibList:
            fib.writeLocs(of);  
            
    ## ====================================================
    ##  "PRIVATE" METHODS NOT ACCESSED OUTSIDE OF THIS CLASS
    ## ====================================================
    def determineQuadLists(self, XnonLoc, Xuniform, Dom):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        This method uses the uniform points to determine if a fiber is close to a target via
        looking at the uniform points which are neighbors of the Chebyshev points. 
        Inputs: XnonLoc, Xuniform = tot#ofpts x 3 arrays of Chebyshev and uniform points on 
        all the fibers, Dom = domain object we are doing computation on.
        Outputs: two numpy arrays. alltargets = sequential list of targets that need correcting
        numByFib = number of targets that need correction for each fiber. This is enough 
        information to match target-fiber pairs
        """
        q1cut = upsampneeded*self._Lf; # cutoff for needed to correct Ewald
        # Get the neighbors from the SpatialDatabase
        t = time.time();
        neighborList = self._SpatialUni.otherNeighborsList(self._SpatialCheb, q1cut);
        print('Neighbor search uniform-Cheb time %f' %(time.time()-t))
        t = time.time();
        alltargets=[];
        numByFib = np.zeros(self._Nfib,dtype=int);
        for iFib in range(self._Nfib):
            fibTargets=list(set([iT for sublist in neighborList[iFib*self._Nunifpf:(iFib+1)*self._Nunifpf] \
                for iT in sublist if iT//self._Npf!=iFib]));
            numByFib[iFib]=len(fibTargets);
            alltargets.extend(fibTargets); # all the targets for each uniform point
        print('Method 2 time %f' %(time.time()-t))
        return alltargets, numByFib;
    
        
    def getStackInds(self,iFib):
        """
        Method to get the row indices in any (Nfib*Nperfib*3) long arrays
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
        return U0;
    
    def evalBendForceDensity(self,X_nonLoc):
        """
        Evaluate the bending force density for a given X (given the 
        known fiber discretization of this class.
        Inputs: X to evaluate -EX_ssss at, as a (tot#ofpts) x 3 array
        Outputs: the forceDensities -EX_ssss also as a (tot#ofpts) x 3 array
        """
        totnum = self._Nfib*self._Npf;
        forceDs=np.zeros(totnum*3);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            Xin = np.reshape(X_nonLoc[rowinds,:],self._Npf*3);
            forceDs[stackinds] = self._fiberDisc.calcfE(Xin);
        return forceDs;

    # This is the dumb quadratic method just to check the SpatialDatabase one is working. 
    # Obsolete
    def determineQuad_quad(self, Dom, XCheb, Xuniform):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        Inputs: cheb points and uniform points as Npts x 3 arrays. 
        Outputs: 4 arrays. Targs[i] = target number that needs correction, 
        fibers[i] = fiber number that is used for the correction at target targs[i], 
        methods[i] = method that needs to be used to do that correction (1 for direct
        with 32 points, 2 for special quadrature), shifts[i] = periodic shift in the 
        fiber so that it is actually numerically close to the target.
        """
        totnum = self._Nfib*self._Npf;
        Lens = Dom.getLens();
        g = Dom.getg();
        # Sort the uniform points.
        # Figure out how many bins to put them in.
        rcutu = upsampneeded*self._Lf;
        qtypes = sp.lil_matrix((totnum,self._Nfib)); # List of quadrature types by fiber
        xShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the x direction
        yShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the y direction
        zShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the z direction
        for iPt in range(totnum): # looping over the target points
            iFib = iPt//self._Npf;
            for jPt in range(self._Nfib*self._Nunifpf):
                jFib = jPt//self._Nunifpf;
                if (iFib!=jFib):
                    rvec = Dom.calcShifted(XCheb[iPt,:]-Xuniform[jPt,:]);
                    nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
                    # Use the distance to determine what type of quadrature is needed
                    if (nr < self._Lf*specneeded and qtypes[iPt,jFib] < 2):
                        qtypes[iPt,jFib] = 2; # special quad needed
                        xShifts[iPt,jFib] =  XCheb[iPt,0]-Xuniform[jPt,0]-rvec[0]+1e-99;
                        yShifts[iPt,jFib] =  XCheb[iPt,1]-Xuniform[jPt,1]-rvec[1]+1e-99;
                        zShifts[iPt,jFib] =  XCheb[iPt,2]-Xuniform[jPt,2]-rvec[2]+1e-99;
                    elif (nr < rcutu and qtypes[iPt,jFib] < 1):
                        qtypes[iPt,jFib]=1; # direct quad with N=32
                        xShifts[iPt,jFib] =  XCheb[iPt,0]-Xuniform[jPt,0]-rvec[0]+1e-99;
                        yShifts[iPt,jFib] =  XCheb[iPt,1]-Xuniform[jPt,1]-rvec[1]+1e-99;
                        zShifts[iPt,jFib] =  XCheb[iPt,2]-Xuniform[jPt,2]-rvec[2]+1e-99;
        # Shift data types so we can loop through the points that need correction
        qtypes = sp.coo_matrix(qtypes);
        xS = (sp.coo_matrix(xShifts)).data;
        yS = (sp.coo_matrix(yShifts)).data;
        zS = (sp.coo_matrix(zShifts)).data;
        shifts = np.concatenate(([xS],[yS],[zS])).T;
        targs = qtypes.row;
        fibers = qtypes.col;
        methods = qtypes.data;
        return targs, fibers, methods, shifts
