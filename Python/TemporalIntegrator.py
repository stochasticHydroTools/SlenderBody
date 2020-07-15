import numpy as np
import copy
import time
from scipy.sparse.linalg import LinearOperator
from functools import partial
from mykrypy.linsys import LinearSystem, Gmres # mykrypy is krypy with modified linsys.py
from mykrypy.utils import ConvergenceError     # mykrypy is krypy with modified linsys.py
from fiberCollection import fiberCollection

# Definitions 
itercap = 1000; # cap on GMRES iterations if we converge all the way
GMREStolerance=1e-6;
verbose = -1;

class TemporalIntegrator(object):

    """
    Class to do the temporal integration. 
    Child classes: BackwardEuler and CrankNicolson
    Abstract class: does first order explicit
    """
    
    def __init__(self,fibCol,CLNetwork=None,FPimp=0):
        self._allFibers = fibCol;
        self._allFibersPrev = copy.deepcopy(self._allFibers); # initialize previous fiber network object
        self._CLNetwork = CLNetwork;
        self._impco = 0; #coefficient for linear solves
        self._maxGMIters = itercap;
        self._FPimplicit = FPimp; # treat finite part implicitly?
    
    def getXandXsNonLoc(self):
        """
        Get the positions and tangent vectors for the nonlocal hydro
        """
        return self._allFibers.getX().copy(), self._allFibers.getXs().copy();

    def getLamNonLoc(self,iT):
        """
        Get the constraint forces lambda for the nonlocal hydro
        """
        return self._allFibers.getLambdas().copy();
        
    def setMaxIters(self,nIters):
        """
        Pass in the maximum number of iterations
        1 = do block diagonal solves only 
        > 1 = do GMRES iterations
        """
        self._maxGMIters = nIters;

    def getMaxIters(self,iT):
        """
        Maximum number of GMRES iterations. Set to infinity for the 
        first step. After that cap at self._maxGMIters
        """
        if (iT==0):
            return itercap; # converge all the way
        return self._maxGMIters;
        
    def gettval(self,iT,dt):
        """
        Get time value for this time step
        """
        return iT*dt;
    
    def getFirstNetworkStep(self,dt):
        """
        Size of the step to update the dynamic CL network the first time 
        in a time step 
        """
        return dt;
    
    def getSecondNetworkStep(self,dt):
        """
        Size of the step to update the dynamic CL network the second time 
        in a time step 
        """
        return 0;
        
    def NetworkUpdate(self,Dom,t,tstep,of=None):
        """
        Update the CL network. Inputs: Dom = domain object, t = current time, tstep = time step to update 
        of = file handle to write the updated configuration to 
        """
        if (tstep==0):
            return;
        Dom.setg(self._allFibers.getg(t));
        self._CLNetwork.updateNetwork(self._allFibers,Dom,tstep,of);

    def updateAllFibers(self,iT,dt,numSteps,Dom,Ewald,gravden,outfile=None,outfileCL=None,write=1):
        """
        The main update method. 
        Inputs: the timestep number as iT, the timestep dt, the maximum number of steps numSteps,
        Dom = the domain object we are solving on Ewald = the Ewald splitter object
        we need to do the Ewald calculations, gravden = graviational force density, 
        outfile = handle to the output file to write to, outfileCL = handle to outfile to write 
        the CL network links to write=1 to write to it, 0 otherwise.  
        """   
        # Update the network (first time - when we do dynamic CLs)
        # self.NetworkUpdate(Dom,iT*dt,self.getFirstNetworkStep(dt,iT));
        # Set domain strain and fill up the arrays of points
        tvalSolve = self.gettval(iT,dt);
        Dom.setg(self._allFibers.getg(tvalSolve));
        
        # Get relevant arguments for local/nonlocal
        XforNL, XsforNL = self.getXandXsNonLoc(); # for M and K
        self._XforStress = XforNL; # save X for the stress calculation
        lamStar = self.getLamNonLoc(iT); # lambdas
        self._allFibersPrev = copy.deepcopy(self._allFibers); # copy old info over
        
        # Forces from gravity and CLs to be treated EXPLICITLY
        forceExt = self._allFibers.uniformForce([0,0,gravden]);
        thist = time.time()
        if (self._CLNetwork is not None):
            uniPoints = self._allFibers.getUniformPoints(XforNL);
            forceExt += self._CLNetwork.CLForce(uniPoints,XforNL,Dom);
        if (verbose > 0):
            print('Time to calc CL force %f' %(time.time()-thist))
            thist = time.time()
        
        # Block diagonal solve
        thist = time.time()
        # Solve the block diagonal way
        Dom.roundg(); # for nonlocal hydro
        RHS = self._allFibers.formBlockDiagRHS(XforNL,XsforNL,tvalSolve,forceExt,lamStar,Dom,Ewald);
        if (verbose > 0):
            print('Time to form RHS %f' %(time.time()-thist))
            thist = time.time()
        # Apply preconditioner to get (delta lambda,delta alpha) from the block diagonal solver
        BlockDiagAnswer = self._allFibers.BlockDiagPrecond(RHS,XsforNL,dt,self._impco,XforNL);
        if (verbose > 0):
            print('Time to apply preconditioner %f' %(time.time()-thist))
            thist = time.time()
        
        # Set answer = block diagonal or proceed with GMRES
        giters = self.getMaxIters(iT)-1; # subtract the first application of mobility/preconditioner
        if (giters==0):
            # Block diagonal acceptable 
            lamalph = BlockDiagAnswer;
        else:
            # GMRES set up and solve
            RHS, relnorm = self._allFibers.calcNewRHS(BlockDiagAnswer,XforNL,XsforNL,dt*self._impco,lamStar, Dom, Ewald);
            systemSize = self._allFibers.getSysDimension();
            A = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.Mobility,impcodt=self._impco*dt,\
                X_nonLoc=XforNL, Xs_nonLoc=XsforNL, Dom=Dom,RPYEval=Ewald),dtype=np.float64);
            Pinv = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.BlockDiagPrecond, \
                   Xs_nonLoc=XsforNL,dt=dt,implic_coeff=self._impco,X_nonLoc=XforNL,doFP=self._FPimplicit),dtype=np.float64);
            SysToSolve = LinearSystem(A,RHS,Mr=Pinv);
            Solution = TemporalIntegrator.GMRES_solve(SysToSolve,tol=GMREStolerance,maxiter=giters)
            lamalphT = Solution.xk;
            lamalph=np.reshape(lamalphT,len(lamalphT))+BlockDiagAnswer;
        if (verbose > 0):
            print('Time to solve GMRES and update alpha and lambda %f' %(time.time()-thist))
            thist = time.time()
            
        # Update alpha and lambda and fiber positions
        self._allFibers.updateLambdaAlpha(lamalph,XsforNL);
        # Converged, update the fiber positions
        maxX = self._allFibers.updateAllFibers(dt,XsforNL,exactinex=1);
        if (verbose > 0):
            print('Update fiber time %f' %(time.time()-thist))
            thist = time.time()
            
        # Copy individual fiber objects into large arrays of points, forces
        self._allFibers.fillPointArrays();
        if (verbose > 0):
            print('Fill arrays time %f' %(time.time()-thist))
            thist = time.time()
            
        # Update the network (second time - when we do dynamic CLs)
        #self.NetworkUpdate(Dom,(iT+1)*dt,self.getSecondNetworkStep(dt,iT,numSteps));
        if (write):
            self._allFibers.writeFiberLocations(outfile);
        return maxX;    

    def computeStress(self,Dom,iT,dt):
        """
        Calculate the stress in the suspension. 
        Inputs: Dom = domain object, iT = time step #, dt = time step size
        Outputs: the stress due to lambda, bending forces, and Cls
        """
        # Stress due to fibers 
        XforNL = self._XforStress;
        lamStress, ElasticStress = fiberCollection.FiberStress(XforNL,\
            self._allFibers.evalBendForceDensity(XforNL),self._allFibers.getLambdas(),\
            1.0*Dom.getLens(),self._allFibers._Nfib,self._allFibers._Npf,self._allFibers._fiberDisc._w);
        
        # Stress due to CLs 
        tvalSolve = self.gettval(iT,dt);
        Dom.setg(self._allFibers.getg(tvalSolve));  # Make sure the shift in the CL links is at time n+1/2
        CLstress = np.zeros((3,3))
        if (self._CLNetwork is not None):
            CLstress = self._CLNetwork.CLStress(self._allFibers,XforNL,Dom);
        return lamStress[1,0],ElasticStress[1,0], CLstress[1,0];
    
    @staticmethod
    def GMRES_solve(linear_system, **kwargs):
        """
        Run GMRES solver and catch convergence errors
        """
        try:
            return Gmres(linear_system, **kwargs)
        except ConvergenceError as e:
            # use solver of exception
            return e.solver    

class BackwardEuler(TemporalIntegrator):

    """
    Backward Euler temporal discretization. The same as the abstract parent class, but 
    with a different implicit coefficient in the solves
    """
 
    def __init__(self,fibCol,CLNetwork=None,FPimp=0):
        super().__init__(fibCol,CLNetwork,FPimp);
        self._impco = 1; #coefficient for linear solves

class CrankNicolson(TemporalIntegrator):

    """
    Crank Nicolson temporal discretization. The same as the abstract parent class, but 
    with a different implicit coefficient in the solves. 
    See parent class for method descriptions 
    """
    
    def __init__(self,fibCol,CLNetwork=None,FPimp=0):
        super().__init__(fibCol,CLNetwork,FPimp);
        self._impco = 0.5; # coefficient for implicit solves
      
    def getXandXsNonLoc(self):
        X = 1.5*self._allFibers.getX() - 0.5*self._allFibersPrev.getX();
        Xs = 1.5*self._allFibers.getXs() - 0.5*self._allFibersPrev.getXs();
        return X, Xs;
    
    def getLamNonLoc(self,iT):
        if (iT > 1):
            lamarg = 2.0*self._allFibers.getLambdas()-1.0*self._allFibersPrev.getLambdas();
        else: # more than 1 iter, lambda should be previous lambda from fixed point
            lamarg = self._allFibers.getLambdas().copy();
        return lamarg;

    def getMaxIters(self,iT):
        if (iT < 2):
            return itercap; # converge all the way
        return self._maxGMIters;

    def gettval(self,iT,dt):
        return (iT+0.5)*dt;
    
    def getFirstNetworkStep(self,dt,iT):
        if (iT==0):
            return dt*0.5;
        return dt; # interpret as a midpoint step
    
    def getSecondNetworkStep(self,dt,iT,numSteps):
        if (iT==numSteps-1):
            return dt*0.5;
        return 0;

    


