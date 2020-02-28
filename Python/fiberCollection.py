import numpy as np
import TargetFiberCorrector as TFibCorVec
from DiscretizedFiber import DiscretizedFiber
from SpatialDatabase import SpatialDatabase, ckDSpatial
import EwaldUtils as ewc
import SpecQuadUtils as sq
import EwaldNumba as ewNum
import scipy.sparse as sp
from math import sqrt

# Definitions
upsampneeded = 0.15*1.05; # upsampneeded*Lfiber = distance where upsampling is needed
specneeded = 0.06*1.20; #specneed*Lfiber = distance where special quad is needed

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
        a copy of, nonLocal = are we doing nonlocal hydro?, mu = fluid viscosity, 
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
    
    def initFibList(self,fibListIn, Dom):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
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
    def nonLocalBkgrndVelocity(self,X_nonLoc,Xs_nonLoc,lam_nonLoc, t, exForceDen, Dom, Ewald):
        """
        Compute the non-local velocity + background flow due to the fibers.
        Inputs: Arguments X_nonLoc, Xs_nonLoc, lam_nonLoc = X, Xs, and lambda for 
        the computation, all as tot#pts x 3 arrays.
        t = current time to evaluate background flow, Dom = Domain object the computation
        happens on, Ewald = EwaldSplitter to compute velocities.
        Outputs: non-local velocity as a tot#ofpts*3 one-dimensional array.
        This includes the background flow and finite part integrals. 
        """
        BkgrndFlow = self.evalU0(X_nonLoc,t);
        # If not doing non-local hydro, return just background flow and stop.
        totnum = self._Nfib*self._Npf;
        if (not self._nonLocal):
            return np.reshape(BkgrndFlow,totnum*3);
        forceDReshaped = np.reshape(exForceDen+lam_nonLoc,(totnum,3));
        # Compute the total force density from lambda, the external force, and also
        # the bending force, which can be determined from X and the fiber discretization. 
        forceDs = self.evalBendForceDensity(X_nonLoc) + forceDReshaped;
        # Multiply by the weights to get force from force density. 
        forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(totnum,1));
        # Fill up all the non-local velocities that are on each fiber separately
        selfRPY, finitePart, onlyLocal = self.calcSelfNLVelocities(X_nonLoc,Xs_nonLoc,forceDs, forces);
        # Update the spatial objects for Chebyshev and uniform
        self._SpatialCheb.updateSpatialStructures(X_nonLoc,Dom);
        uniPoints = self.getUniformPoints(X_nonLoc);
        self._SpatialUni.updateSpatialStructures(uniPoints,Dom);
        # Ewald splitting to compute the far field and near field
        EwaldVelocity = Ewald.calcBlobTotalVel(totnum,X_nonLoc,forces,Dom,self._SpatialCheb);
        EwaldVelocity -= selfRPY; # The periodized Ewald sum, with self term subtracted
        # Corrections for fibers that are close together
        # First determine which sets of points and targets need corrections
        targs, fibers, methods, shifts = self.determineQuadLists(X_nonLoc,uniPoints,Dom);
        # Loop over the points that need corrections and do the corrections
        centerVels = np.reshape(finitePart+onlyLocal,(self._Npf*self._Nfib,3)); # argument for corrections
        corVels = self.CorrectEwaldSpecialQuad(np.array(fibers),np.array(targs),X_nonLoc,forceDs,\
                        forces,np.array(methods),shifts,centerVels);
        EwaldVelocity+=corVels;
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(EwaldVelocity+BkgrndFlow,totnum*3)+finitePart;
    
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
        """
        g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        return g;

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
    def calcSelfNLVelocities(self,X_nonLoc,Xs_nonLoc,forceDsAll,forcesAll):
        """
        Compute the non-local velocities that are localized to 
        a single fiber. This means the self RPY term, the finite part term, and
        also the local velocity (necessary when we do corrections).
        Inputs: X_nonLoc, Xs_nonLoc = non-local arguments as (tot#ofpts x 3) arrays
        forceDsAll, forcesAll = the force densities and forces, respectively on the 
        fiber, also as (tot#ofpts x 3) arrays. 
        Outputs: the selfRPY term (as a tot#ofpts x 3 array), the finite part integral
        (as a tot#ofpts*3 one-dimensional array), and the LocalDrag term (also as 
        a tot#ofpts*3 one-dimensional array) evaluated at X_nonLoc, Xs_nonLoc
        """
        selfRPY = np.zeros(X_nonLoc.shape);
        finitePart = np.zeros(len(X_nonLoc)*3);
        LocalOnly = np.zeros(len(X_nonLoc)*3);
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            stackinds = self.getStackInds(iFib);
            X = X_nonLoc[rowinds,:];
            Xs = Xs_nonLoc[rowinds,:];
            forceD = forceDsAll[rowinds,:];
            force = forcesAll[rowinds,:];
            # Compute self velocity from RPY (which will be subtracted), finite part integral
            # velocity, and purely local velocity (this is needed in the non-local corrections
            # for very close fibers).
            selfRPY[rowinds,:] = np.reshape(ewc.RPYSBTKernel(self._Npf,X[:,0],X[:,1],X[:,2],self._Npf,X[:,0],\
                    X[:,1],X[:,2],force[:,0],force[:,1],force[:,2],self._mu,self._aRPY,0),(self._Npf,3));
            finitePart[stackinds] = self._fiberDisc.calcFPVelocity(X,Xs,forceD);
            LocalOnly[stackinds] = self._fiberDisc.calcLocalVelocity(np.reshape(Xs,(3*self._Npf)),\
                                np.reshape(forceD,(3*self._Npf)));
        return selfRPY, finitePart, LocalOnly;
    
    def CorrectEwaldSpecialQuad(self,fibers,targets,X_nonLoc,forceDs,forces,methods,shifts,centerVels):
        """
        Method to correct the results of Ewald splitting with special quadrature. 
        Inputs: fibers, targets = one-d integer arrays that have, respectively, the fiber number and 
        target number that need to be corrected with some new form of quadratur. 
        X_nonLoc = position arguments as tot#pts x 3 array, forceDs = force densities as tot#pts x 3 array, 
        forces = forceDs*weights on N point grid as a tot#pts x 3 array, methods = types of corrections for 
        each pair (1 for 32 point upsampled direct quad, 2 for special quad), shifts = periodic shifts in the
        targets as a #ofcorrections x 3 array, centerVels = tot#ofpts x 3 array that has the velocity 
        (Local + FP integral) on the fiber centerline for each point
        Output: a tot#ofpts x 3 array of correction velocites
        """
        subVels = np.zeros((self._Npf*self._Nfib,3));
        nUpsample = self._fiberDisc.getNumUpsample();
        w_upsampled = np.reshape(self._fiberDisc.getUpsampledWeights(),(nUpsample,1));
        for iFib in range(self._Nfib):
            targNums = targets[fibers==iFib];
            targPts = X_nonLoc[targNums,:]-shifts[fibers==iFib,:];
            rowinds = self.getRowInds(iFib);
            # Subtract the Ewald value at all the targets (free space RPY kernel)
            subVels[targNums,:]-= np.reshape(ewc.RPYSBTKernel(len(targNums),targPts[:,0],targPts[:,1],targPts[:,2],\
                self._Npf,X_nonLoc[rowinds,0],X_nonLoc[rowinds,1],X_nonLoc[rowinds,2],\
                forces[rowinds,0],forces[rowinds,1],forces[rowinds,2],self._mu,self._aRPY,0),(len(targNums),3));
            # Upsample the fiber
            Xup = self._fiberDisc.upsampleGlobally(X_nonLoc[rowinds,:]);
            forceDUp = self._fiberDisc.upsampleGlobally(forceDs[rowinds,:]);
            cVelUpsamp = self._fiberDisc.upsampleGlobally(centerVels[rowinds,:]);
            forceUp = forceDUp*w_upsampled;
            qtypesAll = methods[fibers==iFib];
            # Do method 1 targets directly
            t1Targs = targNums[qtypesAll==1];
            # Direct quadrature using C++ function
            subVels[t1Targs,:]+= np.reshape(ewc.RPYSBTKernel(len(t1Targs),targPts[qtypesAll==1,0],\
                targPts[qtypesAll==1,1],targPts[qtypesAll==1,2],\
                nUpsample,Xup[:,0],Xup[:,1],Xup[:,2],forceUp[:,0],forceUp[:,1],forceUp[:,2],self._mu,\
                self._aRPY,1),(len(t1Targs),3));
            # The ones left over are for special quadrature
            t2Targs = targNums[qtypesAll==2];
            target2Points = targPts[qtypesAll==2,:];
            # Upsample to 2 panels
            X2Pan = self._fiberDisc.upsample2Panels(X_nonLoc[rowinds,:]);
            forceD2Pan = self._fiberDisc.upsample2Panels(forceDs[rowinds,:]);
            # Vectorized version for special quadrature values
            cvelsV = TFibCorVec.SpecQuadVel(len(t2Targs),target2Points,self._fiberDisc,
                    Xup, X2Pan,forceUp,forceDUp,forceD2Pan,\
                    centerVels[rowinds,:],cVelUpsamp,w_upsampled,\
                    self._epsilon,self._Lf,self._aRPY,self._mu,self._Npf,nUpsample);
            subVels[t2Targs,:]+=cvelsV; 
        return subVels;

    def determineQuadLists(self, XnonLoc, Xuniform, Dom):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        This method uses the uniform points to determine if a fiber is close to a target via
        looking at the uniform points which are neighbors of the Chebyshev points. 
        Inputs: XnonLoc, Xuniform = tot#ofpts x 3 arrays of Chebyshev and uniform points on 
        all the fibers, Dom = domain object we are doing computation on.
        Outputs: 4 arrays. Targs[i] = target number that needs correction, 
        fibers[i] = fiber number that is used for the correction at target targs[i], 
        methods[i] = method that needs to be used to do that correction (1 for direct
        with nUpsample points, 2 for special quadrature), shifts[i] = periodic shift in the
        target so that it is actually numerically close to the fiber.
        """
        q1cut = upsampneeded*self._Lf; # cutoff for needed to correct Ewald
        q2cut = specneeded*self._Lf; # cutoff for needing special quadrature
        Lens = Dom.getLens();
        g = Dom.getg();
        # Get the neighbors from the SpatialDatabase
        neighborList = self._SpatialCheb.otherNeighborsList(self._SpatialUni, q1cut);
        #import time
        #t = time.time();
        # Compute the number of neighbors by point and a flattened list of neighbors that
        # are sorted by point (for pt 1, sorted neighbors, then for pt 2, sorted neighbprs..)
        # using "fast" list comprehensions. Then pass these to the C++ routine which will return
        # the targets, fibers, methods, and shifts.
        nNeighbors = [len(n) for n in neighborList];
        sortedNeighbs = [neighb for n in neighborList for neighb in sorted(n)];
        outputs = sq.determineCorQuad(nNeighbors, sortedNeighbs, XnonLoc[:,0], XnonLoc[:,1], XnonLoc[:,2], \
            len(nNeighbors),Xuniform[:,0],Xuniform[:,1],Xuniform[:,2],self._Nunifpf,g,Lens,q1cut,q2cut);
        targets = outputs[0];
        fibers = outputs[1];
        methods = outputs[2];
        shifts = np.reshape(np.array(outputs[3]),(len(targets),3));
        #print('Cpp time %f' %(time.time()-t));
        if (False):
            # Below are 2 implementations of the same method that are significantly slower
            # than the pure C++ that we are actually using.
            # First is the numba implementation with numpy arrays. Numba seems to be really
            # slow at appending things to the end of an array as we need to do in this method.
            import time
            t = time.time();
            # Turn into a 2D numpy array for numba
            numNeighbors = np.array([len(a) for a in neighborList],dtype=int);
            neighbors2DArray = -np.ones((self._Nfib*self._Npf,np.amax(numNeighbors)),dtype=int);
            for iRow in range(self._Nfib*self._Npf):
                 neighbors2DArray[iRow,:numNeighbors[iRow]] = np.array(neighborList[iRow],dtype=int);
            targets1, fibers1, methods1, shifts1 = ewNum.testList(self._Nfib,self._Npf,\
                self._Nunifpf,numNeighbors,neighbors2DArray,XnonLoc,Xuniform,g,Lens,q1cut,q2cut);
            print('Numba time %f' %(time.time()-t));
            t = time.time();
            # Following is the Python/C++ implementation of the same thing using lists and appending
            # in python. It is faster than the numba, bus still much slower than the pure (ugly) C++.
            t = time.time();
            targets = [];
            fibers = [];
            methods=[];
            shifts=[];
            for iPt in range(self._Nfib*self._Npf): # loop over points
                iFib = iPt//self._Npf; # integer division to get the fiber point i is on
                pNeighbors = np.array(neighborList[iPt],dtype=int); # neighbors of point i
                fNeighbors = pNeighbors//self._Nunifpf; # fibers that are neighbors w point i
                oFibs = np.unique(fNeighbors[fNeighbors!=iFib]); # exclude itself
                for neFib in oFibs: # loop over neighboring fibers
                    oPts = pNeighbors[fNeighbors==neFib]; # close points on those fibers
                    # C++ function to determine the quadrature type (returns a tuple)
                    qtypeShift = ewc.findQtype(XnonLoc[iPt,:],len(oPts),Xuniform[oPts,0],\
                        Xuniform[oPts,1],Xuniform[oPts,2],g,Lens[0],Lens[1],Lens[2],\
                        q1cut, q2cut);
                    qtype=qtypeShift[0];
                    shift=qtypeShift[1];
                    if (qtype > 0): # add to the list of corrections
                        targets.append(iPt);
                        fibers.append(neFib);
                        methods.append(qtype);
                        shifts.append(shift);
            print('Python_cpp time %f' %(time.time()-t));
            print(np.amax(np.abs(targets1-np.array(targets))))
            print(np.amax(np.abs(fibers1-np.array(fibers))))
            print(np.amax(np.abs(methods1-np.array(methods))))
            print(np.amax(np.abs(shifts1-np.array(shifts))))
        return targets, fibers, methods, shifts;
    
    
    def getRowInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._Npf,(iFib+1)*self._Npf);
    
    def getStackInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib x 3) long arrays
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
        forceDs=np.zeros((totnum,3));
        for iFib in range(self._Nfib):
            rowinds = self.getRowInds(iFib);
            Xin = np.reshape(X_nonLoc[rowinds,:],self._Npf*3);
            forceDs[rowinds,:] = np.reshape(self._fiberDisc.calcfE(Xin),(self._Npf,3));
        return forceDs;

    # This is the dumb quadratic method just to check the SpatialDatabase one is working. 
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
