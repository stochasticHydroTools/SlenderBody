import numpy as np
from EwaldSplitter import EwaldSplitter
import TargetFiberCorrector as TFibCor
from DiscretizedFiber import DiscretizedFiber
import EwaldUtils as ewc
import scipy.sparse as sp
from math import sqrt

# Definitions
upsampneeded = 0.15*1.05; # upsampneeded*Lfiber = distance where upsampling is needed
specneeded = 0.06*1.20; #specneed*Lfiber = distance where special quad is needed

class fiberCollection(object):

    ## METHODS FOR INITIALIZATION
    def __init__(self,Nfibs,fibDisc,nonLocal,mu,omega,gam0):
        """
        Constructor for the fiberCollection class. 
        Inputs: Nfibs = number of fibers, fibDisc = discretization object that each fiber will get
        a copy of, nonLocal = are we doing nonlocal hydro?, Dom = domain object we are simulating,
        xi = Ewald splitting parameter, mu = fluid viscosity, omega = frequency of background
        flow oscillations, gam0 = base strain rate of the background flow, Lf = fiber length, 
        epsilon = aspect ratio of the fiber (r/L).
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
        # Allocate memory for arrays of forces, force densities, force velocities
        self._initPointForceVelocityArrays();
    
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
    
    ## "PUBLIC" METHODS (NEEDED EXTERNALLY) 
    def nonLocalBkgrndVelocity(self,X_nonLoc,Xs_nonLoc,lam_nonLoc, t, exForceDen, Dom, Ewald):
        """
        Compute the non-local velocity + background flow due to the fibers.
        Inputs: Arguments X, Xs, lambda, and any external forcing as tot#pts x 3 arrays, 
        and the time t that we evaluate the background flow at. 
        Outputs: non-local velocity as a tot#ofpts*3 one-dimensional array.
        This includes the background flow and finite part integrals. 
        """
        BkgrndFlow = self._evalU0(X_nonLoc,t);
        # If not doing non-local hydro, return just background flow and stop.
        totnum = self._Nfib*self._Npf;
        if (not self._nonLocal):
            return np.reshape(BkgrndFlow,totnum*3);
        forceDReshaped = np.reshape(exForceDen+lam_nonLoc,(totnum,3));
        # Compute the total force density from lambda, the external force, and also
        # the bending force, which can be determined from X and the fiber discretization. 
        forceDs = self._evalBendForceDensity(X_nonLoc) + forceDReshaped;
        # Multiply by the weights to get force from force density. 
        forces = forceDs*np.reshape(np.tile(self._fiberDisc.getw(),self._Nfib),(totnum,1));
        # Fill up all the non-local velocities that are on each fiber separately
        selfRPY, finitePart, onlyLocal = self._calcSelfNLVelocities(X_nonLoc,Xs_nonLoc,forceDs, forces);
        # Update the ckD tree for the Chebyshev points
        chebTree = Dom.makekDTree(X_nonLoc);
        # Update the ckD tree for the uniform points
        uniPoints = self.getUniformPoints(X_nonLoc);
        uniformTree = Dom.makekDTree(uniPoints);
        # Ewald splitting to compute the far field and near field
        EwaldVelocity = Ewald.EwaldTotalVel(totnum,X_nonLoc,forces,Dom,chebTree);
        EwaldVelocity -= selfRPY; # The periodized Ewald sum, with self term subtracted
        # Corrections for fibers that are close together
        # First determine which sets of points and targets need corrections
        targs, fibers, methods, shifts = self._determineQuadkDLists(X_nonLoc, uniPoints,Dom, chebTree, uniformTree);
        # Loop over the points that need corrections and do the corrections
        for iT in range(len(targs)):
            target = X_nonLoc[targs[iT],:];
            jFib = fibers[iT];
            rowinds = self._getRowInds(jFib);
            fibpts = X_nonLoc[rowinds,:]+shifts[iT];
            forcesonfib = forces[rowinds,:];
            forcedsonfib = forceDs[rowinds,:];
            stackinds = self._getStackInds(jFib);
            centerVels = np.reshape(finitePart[stackinds]+onlyLocal[stackinds],(self._Npf,3));
            cvel = TFibCor.correctVel(target,self._fiberDisc,fibpts,forcesonfib,\
                        forcedsonfib,centerVels,methods[iT]);
            EwaldVelocity[targs[iT],:]+=cvel; # add the correction to the value at the target
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(EwaldVelocity+BkgrndFlow,totnum*3)+finitePart;

    def linSolveAllFibers(self,XsforNL,nLvel,forceExt,fptol,dt, implic_coeff):
        """
        Perform one timestep by fixed point iterations with maximum number of iterations
        itmax (usually 1). 
        Inputs: dt = timestep, t = time value for background flow calculations (usually t+dt/2)
        solver = temporal integrator (2 for Crank-Nicolson, 1 for backward Euler, -1 for forward
        Euler), LMMlam = whether to use 2*lambda-1*lambdaprev as the initial guess for fixed point
        iteration (LMMlam=1 if yes, if LMMlam=0, will use lambda as the initial guess). 
        itmax = maximum number of iterations of fixed point, fptol = tolerance for the fixed point
        iterations, forceExt = external forcing on the fibers, exactinex = whether to do the 
        update in a way that maintains exact inextensibility. 
        """
        # Linear solve on each fiber
        for iFib in range(self._Nfib):
            fib = self._fibList[iFib];
            stackinds = self._getStackInds(iFib);
            rowinds = self._getRowInds(iFib);
            alphaInds = range(iFib*(2*self._Npf-2),(iFib+1)*(2*self._Npf-2));
            Xin = np.reshape(self._ptsCheb[rowinds,:],self._Npf*3); # X^n is the input, not XnonLoc!!
            Xsin = np.reshape(XsforNL[rowinds,:],self._Npf*3);
            self._alphas[alphaInds], self._velocities[stackinds], self._lambdas[stackinds] = \
                self._fiberDisc.alphaLambdaSolve(Xin,Xsin,dt,implic_coeff,nLvel[stackinds],forceExt[stackinds]);
        
    def updateAllFibers(self,dt,XsforNL,exactinex=1):
        # Fixed point over - update the fiber configurations X and Xs
        # This updates the configurations in the OBJECT, which is much cleaner
        for iFib in range(self._Nfib):
            stackinds = self._getStackInds(iFib);
            rowinds = self._getRowInds(iFib);
            fib = self._fibList[iFib];
            XsforOmega = np.reshape(XsforNL[rowinds,:],self._Npf*3);
            alphaInds = range(iFib*(2*self._Npf-2),(iFib+1)*(2*self._Npf-2));
            fib.updateXsandX(dt,self._velocities[stackinds],XsforOmega,self._alphas[alphaInds],exactinex);    

    def fillPointArrays(self):
        """
        Copy the X and Xs arguments from self._fibList (list of fiber
        objects) into large (tot#pts x 3) arrays that are stored in memory
        """
        self._ptsChebPrev = self._ptsCheb.copy();
        self._tanvecsPrev = self._tanvecs.copy();
        for iFib in range(self._Nfib):
            rowinds = self._getRowInds(iFib);
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
        Inputs: tolerance for the fixed point convergence.
        """
        reler = np.linalg.norm(self._lambdas-lambdasm1)/max(1.0,np.linalg.norm(self._lambdas));
        return (reler < fptol); 
    
    def getUniformPoints(self,chebpts):
        """
        Obtain uniform points from a set of Chebyshev points. 
        Inputs: chebpts as a (tot#ofpts x 3) array
        Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
        This method is necessary for cross-linking
        """
        uniPoints = np.zeros((self._Nfib*self._Nunifpf,3));
        for iFib in range(self._Nfib):
            uniinds = range(iFib*self._Nunifpf,(iFib+1)*self._Nunifpf);
            rowinds = self._getRowInds(iFib);
            uniPoints[uniinds]=self._fiberDisc.resampleUniform(chebpts[rowinds]);
        return uniPoints;

    def getg(self,t):
        """
        Change the value of g according to what the background flow dictates.
        """
        g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        return g;

    def writeFiberLocations(self,of):
        """
        Write the locations of all files to a file
        object named of.
        """
        for fib in self._fibList:
            fib.writeLocs(of);    

    ## "PRIVATE" METHODS NOT ACCESSED OUTSIDE OF THIS CLASS         
    def _calcSelfNLVelocities(self,X_nonLoc,Xs_nonLoc,forceDsAll,forcesAll):
        """
        Fill up the arrays of non-local velocities that are localized to 
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
            rowinds = self._getRowInds(iFib);
            stackinds = self._getStackInds(iFib);
            X = X_nonLoc[rowinds,:];
            Xs = Xs_nonLoc[rowinds,:];
            forceD = forceDsAll[rowinds,:];
            force = forcesAll[rowinds,:];
            # Compute self velocity from RPY (which will be subtracted), finite part integral
            # velocity, and purely local velocity (this is needed in the non-local corrections
            # for very close fibers).
            selfRPY[rowinds,:] = self._fiberDisc.subtractRPY(X,force);
            finitePart[stackinds] = self._fiberDisc.calcFPVelocity(X,Xs,forceD);
            LocalOnly[stackinds] = self._fiberDisc.calcLocalVelocity(np.reshape(Xs,(3*self._Npf)),\
                                np.reshape(forceD,(3*self._Npf)));
        return selfRPY, finitePart, LocalOnly;

    def _determineQuadkDLists(self, XnonLoc, Xuniform, Dom, chebTree, uniformTree):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        This method uses the uniform points to determine if a fiber is close to a target via
        looking at the kD trees of the uniform/Chebyshev points.
        Inputs: XnonLoc, Xuniform = tot#ofpts x 3 arrays of Chebyshev and uniform points on 
        all the fibers. chebTree and uniformTree = ckD trees of those points. 
        Outputs: 4 arrays. Targs[i] = target number that needs correction, 
        fibers[i] = fiber number that is used for the correction at target targs[i], 
        methods[i] = method that needs to be used to do that correction (1 for direct
        with nUpsample points, 2 for special quadrature), shifts[i] = periodic shift in the
        fiber so that it is actually numerically close to the target.
        """
        targets = [];
        fibers = [];
        methods=[];
        shifts=[];
        q1cut = upsampneeded*self._Lf; # cutoff for needed to correct Ewald
        q2cut = specneeded*self._Lf; # cutoff for needing special quadrature
        # Get the neighbors from the kD trees
        neighborList = Dom.TwokDTreesNeighbors(chebTree, uniformTree, q1cut);
        Lens = Dom.getLens();
        g = Dom.getg();
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
        return targets, fibers, methods, shifts;
    

    def _initPointForceVelocityArrays(self):
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
    
    def _getRowInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib) x 3 arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*self._Npf,(iFib+1)*self._Npf);
    
    def _getStackInds(self,iFib):
        """
        Method to get the row indices in any (Nfib x Nperfib x 3) long arrays
        for fiber number iFib. This method could easily be modified to allow
        for a bunch of fibers with different numbers of points. 
        """
        return range(iFib*3*self._Npf,(iFib+1)*3*self._Npf);
    

    def _getNonLocalArguments(self,tint,LMMlam):
        """
        Method to return the X, Xs, and lambda arguments for non-local solves. 
        Inputs: tint = type of temporal integrator (2 for Crank-Nicolson, 
        1 for backward Euler, -1 for forward Euler), LMMlam = whether to extrapolate
        for lambda as well. 
        Outputs: The non-local arguments X, Xs, lambda as total#ofpoints x 3 arrays. 
        """
        totnum = self._Nfib*self._Npf;
        LMM_XCoeff = 0.5*(np.abs(tint)-1);
        X_nonLoc = (1+LMM_XCoeff)*self._ptsCheb-LMM_XCoeff*self._ptsChebPrev;
        Xs_nonLoc = (1+LMM_XCoeff)*self._tanvecs-LMM_XCoeff*self._tanvecsPrev;
        lam_nonLoc = np.reshape((1+LMMlam)*self._lambdas - LMMlam*self._lambdasPrev,(totnum,3));
        return X_nonLoc, Xs_nonLoc, lam_nonLoc;

    def _evalU0(self,Xin,t):
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
    
    def _evalBendForceDensity(self,X_nonLoc):
        """
        Evaluate the bending force density for a given X (given the 
        known fiber discretization of this class.
        Inputs: X to evaluate -EX_ssss at, as a (tot#ofpts) x 3 array
        Outputs: the forceDensities -EX_ssss also as a (tot#ofpts) x 3 array
        """
        totnum = self._Nfib*self._Npf;
        forceDs=np.zeros((totnum,3));
        for iFib in range(self._Nfib):
            rowinds = self._getRowInds(iFib);
            Xin = np.reshape(X_nonLoc[rowinds,:],self._Npf*3);
            forceDs[rowinds,:] = np.reshape(self._fiberDisc.calcfE(Xin),(self._Npf,3));
        return forceDs;

    # This is the dumb quadratic method just to check the kD tree one is working. 
    def _determineQuad_quad(self, Dom, XCheb, Xuniform):
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
