import numpy as np
from EwaldSplitter import EwaldSplitter
from TargetFiberCorrector import TargetFiberCorrector
from DiscretizedFiber import DiscretizedFiber
import EwaldUtils as ewc
import scipy.sparse as sp
from math import sqrt

# Definitions
rho_crit = 1.114; # critical Bernstein radius for special quad
nptsUpsample = 32; # number of points to upsample to for special quadrature
nptsUniform = 16; # number of uniform points to estimate distance
upsampneeded = 0.15*1.05; # upsampneeded*Lfiber = distance where upsampling is needed
specneeded = 0.06*1.20; #specneed*Lfiber = distance where special quad is needed

class fiberCollection(object):

    def __init__(self,Nfibs,fibDisc,nonLocal,Dom,xi,mu,omega,gam0):
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
        self._fiberDisc.initSpecialQuadMatrices(nptsUniform,nptsUpsample);
        self._Npf = self._fiberDisc.getN();
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._epsilon, self._Lf = self._fiberDisc.getepsilonL();
        self._Dom = Dom; # Same Domain object for fiberCollection and EwaldSplitter (by reference!)
        # Initialize Ewald
        self._Ew = EwaldSplitter(self._Dom,xi,sqrt(1.5)*self._Lf*self._epsilon,mu);
        # Allocate memory for arrays of forces, force densities, force velocities
        self.initPointForceVelocityArrays();
        # Initialize TargetFiberCorrector for special quadrature corrections
        self._TargFibCor = TargetFiberCorrector(rho_crit,nptsUpsample);
    
    def initFibList(self,fibListIn):
        """
        Initialize the list of fibers. This is done by giving each fiber
        a copy of the discretization object, and then initializing positions
        and tangent vectors.
        """
        self._fibList = fibListIn;
        for jFib in xrange(self._Nfib):
           if self._fibList[jFib] is None:
                # Initialize memory
                self._fibList[jFib] = DiscretizedFiber(self._fiberDisc);
                # Initialize straight fiber positions at t=0
                self._fibList[jFib].initFib(self._Dom.getLens());
                self._fibList[jFib].initPastVariables();
    
    def initPointForceVelocityArrays(self):
        # Establish large lists that will be used for the non-local Ewald
        totnum = self._Nfib*self._Npf;                          # total # of points
        self._ptsCheb=np.zeros((totnum,3));                     # Chebyshev points
        self._tanvecs=np.zeros((totnum,3));                     # Tangent vectors
        self._ptsuni=np.zeros((nptsUniform*self._Nfib,3));      # 16 uniformly distributed points
        self._forceDs=np.zeros((totnum,3));                     # force densities
        self._forces = np.zeros((totnum,3));                    # forces
        self._selfVel = np.zeros((totnum,3));                   # RPY self velocity (to subtract)
        self._FpVel = np.zeros(totnum*3);                       # Finite part integral velocities
        self._LocVel = np.zeros(totnum*3);                      # Purely local velocities
        self._U0fibers = np.zeros((totnum,3));                  # Background flows

    def fillPointBkgrndVelocityArrays(self, tint, t):
        """
        Fill up the arrays of positions, tangent vectors, and background 
        velocities. This has to be done before calling external forces, and 
        it also has to be done whether we do local drag or nonlocal hydro. 
        Inputs: tint = temporal integrator (2 for CN, 1 for BE, -1 for FE),
        t = real time (for background flow computation). 
        """
        if (not self._nonLocal): # if only local, only need background flow
            for iFib in xrange(self._Nfib):
                fib = self._fibList[iFib];
                X, _ = fib.getXandXs(tint);
                rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
                # Background flow values
                self._U0fibers[rowinds,:] = self.evalU0(X,t);
        else: # Fill up the lists by looping through the fibers
            for iFib in xrange(self._Nfib):
                fib = self._fibList[iFib];
                # Get the arguments from the fiber class.
                X, Xs = fib.getXandXs(tint);
                rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
                self._ptsCheb[rowinds,:] = X;
                self._ptsuni[iFib*nptsUniform:(iFib+1)*nptsUniform,:] = \
                        self._fiberDisc.resampleUniform(X);
                self._tanvecs[rowinds,:] = Xs;
                # Background flow values
                self._U0fibers[rowinds,:] = self.evalU0(X,t);
        # Sort the uniform and Chebyshev points into a KD trees
        self.makekDTrees();

    def fillForceArrays(self,tint,LMMlam,fext):
        """
        Fill up the arrays of fprces. This only has to be done
        in the non-local hydro case.
        Inputs: tint = temporal integrator (2 for CN, 1 for BE, -1 for FE),
        LMMlam = whether to do a linear multistep combo to get lambdas,
        fext = the external forcing on the fiber.
        """
        if self._nonLocal:
            for iFib in xrange(self._Nfib):
                fib = self._fibList[iFib];
                # Get the arguments from the fiber class.
                fE, lam = fib.getfeandlambda(tint,LMMlam);
                rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
                exFib = fext[rowinds,:];
                w = np.reshape(self._fiberDisc.getw(),(self._Npf,1));
                self._forceDs[rowinds,:] = fE+lam+exFib
                #self._forceDs[rowinds,:] = exFib; #use to test for given external forcing
                self._forces[rowinds,:] = (fE+lam+exFib)*w;
                #self._forces[rowinds,:] = exFib*w; #use to test for given external forcing
                #print 'Using just external forcing for debugging'

    def calcSelfNLVelocities(self):
        """
        Fill up the arrays of non-local velocities that are localized to 
        a single fiber. This means the self RPY term, the finite part term, and
        also the local velocity (necessary when we do corrections).
        """
        for iFib in xrange(self._Nfib):
            rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
            stackinds = range(iFib*3*self._Npf,(iFib+1)*3*self._Npf);
            X = self._ptsCheb[rowinds,:];
            Xs = self._tanvecs[rowinds,:];
            forceDs = self._forceDs[rowinds,:];
            forces = self._forces[rowinds,:];
            # Compute self velocity from RPY (which will be subtracted), finite part integral
            # velocity, and purely local velocity (this is needed in the non-local corrections
            # for very close fibers).
            self._selfVel[rowinds,:] = self._fiberDisc.subtractRPY(X,forces);
            self._FpVel[stackinds] = self._fiberDisc.calcFPVelocity(X,Xs,forceDs);
            self._LocVel[stackinds] = self._fiberDisc.calcLocalVelocity(np.reshape(Xs,(3*self._Npf)),\
                                np.reshape(forceDs,(3*self._Npf)));

    def setg(self,t):
        """
        Change the value of g according to what the background flow dictates.
        """
        g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        self._Dom.setg(g);
    
    def getg(self):
        return self._Dom.getg();
    
    def linSolveAllFibers(self,dt,solver,iters,uNonLoc,forceExt):
        """
        Solve the linear system for a given dt, temporal integrator, 
        non-local background flow, and external forcing, to obtain 
        alpha and lambda on each fiber. 
        Inputs: dt (timestep), solver = temporal integrator (2 for CN, 1 for BE
        -1 for FE), iters = iteration count (to determine when to copy lambda),
        uNonLoc = N*Nfib array of background/non-local velocities, forceExt = 
        N*Nfib array of external forces
        """
        for iFib in xrange(self._Nfib):
            fib = self._fibList[iFib];
            stackinds = range(iFib*3*self._Npf,(iFib+1)*3*self._Npf);
            fib.alphaLambdaSolve(dt,solver,iters,uNonLoc[stackinds],forceExt[stackinds]);
    
    def AnyNotConverged(self, fptol):
        """
        Check whether any of the fibers are not converged (i.e. whether
        the overall fixed point iteration to determine lambda has converged).
        Inputs: tolerance for the fixed point convergence.
        """
        notconvbyFib = np.ones(self._Nfib);
        for jFib in xrange(self._Nfib):
            fib = self._fibList[jFib];
            notconvbyFib[jFib] = fib.notConverged(fptol);
        return sum(notconvbyFib);

    def updateAllFibers(self,dt,solver,exactinex=1):
        """
        Once the fixed point iteration has converged, update the tangent vectors
        and then positions of all the fibers.
        Inputs: timestep dt and solver = temporal integrator (2 for CN, 1 for BE
        -1 for FE), exactinex=1 for exact inextensibility, 0 otherwise.
        """
        for fib in self._fibList:
            fib.updateXsandX(dt,solver,exactinex);

    def writeFiberLocations(self,of):
        """
        Write the locations of all files to a file
        object named of.
        """
        for fib in self._fibList:
            fib.writeLocs(of);

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

    def makekDTrees(self):
        """
        Make kD trees for the uniform and Chebyshev points
        """
        self._chebTree = self._Dom.makekDTree(self._ptsCheb);
        self._uniformTree = self._Dom.makekDTree(self._ptsuni);

    def nonLocalBkgrndVelocity(self):
        """
        Compute the non-local velocity due to the fibers.
        Outputs: non-local velocity as an (nFib x 3*N per fib) one-dimensional array.
        This includes the background flow.
        """
        # If not doing non-local hydro, return just background flow and stop.
        totnum = self._Nfib*self._Npf;
        if (not self._nonLocal):
            return np.reshape(self._U0fibers,totnum*3);
        # Fill up all the non-local velocities that are on each fiber separately
        self.calcSelfNLVelocities();
        # Ewald splitting to compute the far field and near field
        EwaldVelocity = self._Ew.EwaldTotalVel(totnum,self._ptsCheb,self._forces,self._chebTree);
        EwaldVelocity -= self._selfVel; # The periodized Ewald sum, with self term subtracted
        # Corrections for fibers that are close together
        # First determine which sets of points and targets need corrections
        targs, fibers, methods, shifts = self.determineQuadkDLists();
        # Loop over the points that need corrections and do the corrections
        for iT in xrange(len(targs)):
            target = self._ptsCheb[targs[iT],:];
            jFib = fibers[iT];
            rowinds = range(jFib*self._Npf,(jFib+1)*self._Npf)
            fibpts = self._ptsCheb[rowinds,:]+shifts[iT];
            forcesonfib = self._forces[rowinds,:];
            forcedsonfib = self._forceDs[rowinds,:];
            stackinds = range(jFib*3*self._Npf,(jFib+1)*3*self._Npf);
            centerVels = np.reshape(self._FpVel[stackinds]+self._LocVel[stackinds],(self._Npf,3));
            cvel = self._TargFibCor.correctVel(target,self._fibList[jFib].getFibDisc(),fibpts,forcesonfib,\
                        forcedsonfib,centerVels,methods[iT]);
            EwaldVelocity[targs[iT],:]+=cvel; # add the correction to the value at the target
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(EwaldVelocity+self._U0fibers,totnum*3)+self._FpVel;

    
    def determineQuadkDLists(self):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        This method uses the uniform points to determine if a fiber is close to a target via
        looking at the kD trees of the uniform/Chebyshev points.
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
        neighborList = self._Dom.TwokDTreesNeighbors(self._chebTree, self._uniformTree, q1cut);
        Lens = self._Dom.getLens();
        g = self._Dom.getg();
        for iPt in xrange(self._Nfib*self._Npf): # loop over points
            iFib = iPt//self._Npf; # integer division to get the fiber point i is on
            pNeighbors = np.array(neighborList[iPt],dtype=int); # neighbors of point i
            fNeighbors = pNeighbors//nptsUniform; # fibers that are neighbors w point i
            oFibs = np.unique(fNeighbors[fNeighbors!=iFib]); # exclude itself
            for neFib in oFibs: # loop over neighboring fibers
                oPts = pNeighbors[fNeighbors==neFib]; # close points on those fibers
                # C++ function to determine the quadrature type (returns a tuple)
                qtypeShift = ewc.findQtype(self._ptsCheb[iPt,:],len(oPts),self._ptsuni[oPts,0],\
                    self._ptsuni[oPts,1],self._ptsuni[oPts,2],g,Lens[0],Lens[1],Lens[2],\
                    q1cut, q2cut);
                qtype=qtypeShift[0];
                shift=qtypeShift[1];
                if (qtype > 0): # add to the list of corrections
                    targets.append(iPt);
                    fibers.append(neFib);
                    methods.append(qtype);
                    shifts.append(shift);
        return targets, fibers, methods, shifts;
    
    # This is the dumb quadratic method just to check the kD tree one is working. 
    def determineQuad_quad(self):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        Outputs: 4 arrays. Targs[i] = target number that needs correction, 
        fibers[i] = fiber number that is used for the correction at target targs[i], 
        methods[i] = method that needs to be used to do that correction (1 for direct
        with 32 points, 2 for special quadrature), shifts[i] = periodic shift in the 
        fiber so that it is actually numerically close to the target.
        """
        totnum = self._Nfib*self._Npf;
        Lens = self._Dom.getLens();
        g = self._Dom.getg();
        # Sort the uniform points.
        # Figure out how many bins to put them in.
        rcutu = upsampneeded*self._Lf;
        qtypes = sp.lil_matrix((totnum,self._Nfib)); # List of quadrature types by fiber
        xShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the x direction
        yShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the y direction
        zShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the z direction
        for iPt in xrange(totnum): # looping over the target points
            iFib = iPt//self._Npf;
            for jPt in xrange(self._Nfib*nptsUniform):
                jFib = jPt//nptsUniform;
                if (iFib!=jFib):
                    rvec = self._Dom.calcShifted(self._ptsCheb[iPt,:]-self._ptsuni[jPt,:]);
                    nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
                    # Use the distance to determine what type of quadrature is needed
                    if (nr < self._Lf*specneeded and qtypes[iPt,jFib] < 2):
                        qtypes[iPt,jFib] = 2; # special quad needed
                        xShifts[iPt,jFib] =  self._ptsCheb[iPt,0]-self._ptsuni[jPt,0]-rvec[0]+1e-99;
                        yShifts[iPt,jFib] =  self._ptsCheb[iPt,1]-self._ptsuni[jPt,1]-rvec[1]+1e-99;
                        zShifts[iPt,jFib] =  self._ptsCheb[iPt,2]-self._ptsuni[jPt,2]-rvec[2]+1e-99;
                    elif (nr < rcutu and qtypes[iPt,jFib] < 1):
                        qtypes[iPt,jFib]=1; # direct quad with N=32
                        xShifts[iPt,jFib] =  self._ptsCheb[iPt,0]-self._ptsuni[jPt,0]-rvec[0]+1e-99;
                        yShifts[iPt,jFib] =  self._ptsCheb[iPt,1]-self._ptsuni[jPt,1]-rvec[1]+1e-99;
                        zShifts[iPt,jFib] =  self._ptsCheb[iPt,2]-self._ptsuni[jPt,2]-rvec[2]+1e-99;
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
