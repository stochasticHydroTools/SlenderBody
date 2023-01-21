import numpy as np
import copy
import time
from scipy.sparse.linalg import LinearOperator
from functools import partial
from mykrypy.linsys import LinearSystem, Gmres # mykrypy is krypy with modified linsys.py
from mykrypy.utils import ConvergenceError     # mykrypy is krypy with modified linsys.py
from warnings import warn

# Definitions 
itercap = 300; # cap on GMRES iterations if we converge all the way
GMREStolerance=1e-6; # larger than GPU tolerance
verbose = -1;

# Documentation last updated: 03/12/2021

class TemporalIntegrator(object):

    """
    Class to do temporal integration. 
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
        > 1 = do nIters-1 GMRES iterations
        """
        self._maxGMIters = nIters;
    
    def setLargeTol(self,tol):
        self._FiniteGMREStolerance = tol;

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
    
    def getFirstNetworkStep(self,dt,iT):
        """
        Size of the step to update the dynamic CL network the first time 
        in a time step 
        """
        return dt;
    
    def getSecondNetworkStep(self,dt,iT,numSteps):
        """
        Size of the step to update the dynamic CL network the second time 
        in a time step 
        """
        return 0;
        
    def NetworkUpdate(self,Dom,t,tstep,fixedg=None):
        """
        Update the CL network. Inputs: Dom = domain object, t = current time, 
        tstep = time step to update 
        """
        if (tstep==0):
            return;
        if (fixedg is None):
            Dom.setg(self._allFibers.getg(t));
        else:
            Dom.setg(fixedg);       
        self._CLNetwork.updateNetwork(self._allFibers,Dom,tstep);

    def updateAllFibers(self,iT,dt,numSteps,Dom,Ewald=None,gravden=0.0,outfile=None,write=1,updateNet=False,turnoverFibs=False,\
        BrownianUpdate=False,fixedg=None,stress=False):
        """
        The main update method. 
        Inputs: the timestep number as iT, the timestep dt, the maximum number of steps numSteps,
        Dom = the domain object we are solving on, Ewald = the Ewald splitter object
        we need to do the nonlocal RPY calculations, gravden = graviational force density (in the z direction), 
        outfile = handle to the output file to write to, write = whether to write the positions of the fibers
        to the output object. updateNet = whether to update the cross-linked network. TurnoverFibs = whether
        to turnover fibers, BrownianUpdate = whether to do a rigid body rotation and translation as a separate
        splitting step, fixedg = the value of strain if fixed, stress = whether to output stress
        """   
        # Birth / death fibers
        thist = time.time() 
        if (turnoverFibs):
            bornFibs = self._allFibers.FiberBirthAndDeath(self.getFirstNetworkStep(dt,iT),self._allFibersPrev);
            self._CLNetwork.deleteLinksFromFibers(bornFibs)
            if (verbose > 0):
                print('Time to turnover fibers (first time) %f' %(time.time()-thist))
                thist = time.time()    
        # Update the network (first time)
        if (updateNet):
            self.NetworkUpdate(Dom,iT*dt,self.getFirstNetworkStep(dt,iT),fixedg);
            if (verbose > 0):
                print('Time to update network (first time) %f' %(time.time()-thist))
                thist = time.time()
        # Brownian update (always first order)
        thist = time.time() 
        if (BrownianUpdate):
            if not (isinstance(self,BackwardEuler)):
                raise ValueError('Must use backward Euler with Brownian fluctuations!')
            Randoms = np.random.randn(6*self._allFibers._Nfib);
            self._allFibers.BrownianUpdate(dt,Randoms);
            if (verbose > 0):
                print('Time to do Brownian update %f' %(time.time()-thist))
                thist = time.time()  
        
        # Set domain strain and fill up the arrays of points
        tvalSolve = self.gettval(iT,dt);
        if (fixedg is None):
            Dom.setg(self._allFibers.getg(tvalSolve));
        else:
            Dom.setg(fixedg);
        self._gForStress = Dom.getg();
        
        # Get relevant arguments for local/nonlocal
        XforNL, XsforNL = self.getXandXsNonLoc(); # for M and K
        
        # Forces from gravity and CLs to be treated EXPLICITLY
        forceExt = self._allFibers.uniformForce([0,0,gravden]);
        if (self._CLNetwork is not None):
            uniPoints = self._allFibers.getUniformPoints(XforNL);
            forceExt += self._CLNetwork.CLForce(uniPoints,XforNL,Dom,self._allFibers);
            if (verbose > 0):
                print('Time to calc CL force %f' %(time.time()-thist))
                thist = time.time()
        
        # Block diagonal solve
        thist = time.time()
        # Solve the block diagonal way
        if (turnoverFibs): # reinitialize lambda for fibers that were turned over
            self._allFibers.initializeLambdaForNewFibers(bornFibs,forceExt,tvalSolve,dt,self._impco,self._allFibersPrev)    
        Dom.roundg(); # for nonlocal hydro, round g to [-1/2,1/2] (have to do this after computing CL forces)
        lamStar = self.getLamNonLoc(iT); # lambdas
        self._allFibersPrev = copy.deepcopy(self._allFibers); # copy old info over
        
        RHS = self._allFibers.formBlockDiagRHS(XforNL,XsforNL,tvalSolve,forceExt,lamStar,Dom,Ewald,FPimplicit=self._FPimplicit);
        if (verbose > 0):
            print('Time to form RHS %f' %(time.time()-thist))
            thist = time.time()
            
        # Apply preconditioner to get (delta lambda,delta alpha) from the block diagonal solver
        BlockDiagAnswer = self._allFibers.BlockDiagPrecond(XsforNL,dt,self._impco,XforNL,doFP=self._FPimplicit)
        if (verbose > 0):
            print('Time to apply preconditioner %f' %(time.time()-thist))
            thist = time.time()
        
        # Set answer = block diagonal or proceed with GMRES
        giters = self.getMaxIters(iT)-1; # subtract the first application of mobility/preconditioner
        giters = 0; 
        from warnings import warn
        warn('Bypassing GMRES iterations for now to test new discretization. Return later')
        if (giters==0):
            # Block diagonal acceptable 
            lamalph = BlockDiagAnswer;
            itsneeded = 0;
        else:
            # GMRES set up and solve
            RHS = self._allFibers.calcNewRHS(BlockDiagAnswer,XforNL,XsforNL,dt*self._impco,lamStar, Dom, Ewald,FPimplicit=self._FPimplicit);
            systemSize = self._allFibers.getSysDimension();
            A = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.Mobility,impcodt=self._impco*dt,\
                X_nonLoc=XforNL, Xs_nonLoc=XsforNL, Dom=Dom,RPYEval=Ewald),dtype=np.float64);
            Pinv = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.BlockDiagPrecond, \
                   Xs_nonLoc=XsforNL,dt=dt,implic_coeff=self._impco,X_nonLoc=XforNL,doFP=self._FPimplicit),dtype=np.float64);
            SysToSolve = LinearSystem(A,RHS,Mr=Pinv);
            if (giters == itercap-1):
                gtol = GMREStolerance; # larger than GPU tolerance
            else:
                gtol = self._FiniteGMREStolerance;
            Solution = TemporalIntegrator.GMRES_solve(SysToSolve,tol=gtol,maxiter=giters)
            lamalphT = Solution.xk;
            itsneeded = len(Solution.resnorms)
            if (giters > 5 and verbose > 0): 
                resno = Solution.resnorms
                print(resno)
                print('Number of iterations %d' %len(resno))
                print('Last residual %f' %resno[len(resno)-1])
            if (itsneeded == itercap):
                resno = Solution.resnorms
                print('WARNING: GMRES did not actually converge. The error at the last residual was %f' %resno[len(resno)-1])
            lamalph=np.reshape(lamalphT,len(lamalphT))+BlockDiagAnswer;
            if (verbose > 0):
                print('Time to solve GMRES and update alpha and lambda %f' %(time.time()-thist))
                thist = time.time()
                
        # Update alpha and lambda and fiber positions
        self._allFibers.updateLambdaAlpha(lamalph,XsforNL);
        maxX = self._allFibers.updateAllFibers(dt,XsforNL);
        if (verbose > 0):
            print('Update fiber time %f' %(time.time()-thist))
            thist = time.time()
        
        stressArray = np.zeros(3);
        if (stress):
            lamStress, ElasticStress = self._allFibers.FiberStress(XforNL,self._allFibers.getLambdas(),Dom)
            # Stress due to CLs 
            Dom.setg(self._gForStress);  # Make sure the shift in the CL links is at time n+1/2
            CLstress = 0
            if (self._CLNetwork is not None):
                CLstress = self._CLNetwork.CLStress(self._allFibers,XforNL,Dom);
            stressArray = np.array([lamStress ,ElasticStress, CLstress]);     
                        
        # Update the network (second time)
        if (updateNet):
            self.NetworkUpdate(Dom,(iT+1)*dt,self.getSecondNetworkStep(dt,iT,numSteps),fixedg);
            if (verbose > 0):
                print('Time to update network (second time) %f' %(time.time()-thist))
                thist = time.time()  
        # Turnover fibers (second time)
        if (turnoverFibs):
            bornFibs = self._allFibers.FiberBirthAndDeath(self.getSecondNetworkStep(dt,iT,numSteps),self._allFibersPrev);
            self._CLNetwork.deleteLinksFromFibers(bornFibs)
            #lBorn+=len(bornFibs);
            if (verbose > 0):
                print('Time to turnover fibers (second time) %f' %(time.time()-thist))
                thist = time.time() 
                   
        if (write):
            self._allFibers.writeFiberLocations(outfile);
        return maxX, itsneeded, stressArray;    
     
    
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
        else: # t = 0
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

    


