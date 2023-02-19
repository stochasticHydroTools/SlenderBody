import numpy as np
import copy
import time
from scipy.sparse.linalg import LinearOperator
from functools import partial
from mykrypy.linsys import LinearSystem, Gmres # mykrypy is krypy with modified linsys.py
from mykrypy.utils import ConvergenceError     # mykrypy is krypy with modified linsys.py
from warnings import warn
from fiberCollection import fiberCollection, SemiflexiblefiberCollection

# Definitions 
itercap = 100; # cap on GMRES iterations if we converge all the way
GMREStolerance=1e-3; # larger than GPU tolerance
verbose = -1;

# Documentation last updated: 03/12/2021

class TemporalIntegrator(object):

    """
    Class to do temporal integration. 
    There are three possibilities implemented:
    1) Backward Euler - first order accuracy for deterministic fibers
    2) Crank Nicolson - "second order" accuracy for deterministic fibers. However, 
    this temporal integrator is very sensitive because it uses linear multistep formulas
    to get higher-order accuracy. We do not use it any more for this reason.
    3) Midpoint Drift integrator - for filaments with semiflexible bending fluctuations. 
    Abstract class: does first order explicit
    """
    
    def __init__(self,fibCol,CLNetwork=None):
        self._allFibers = fibCol;
        self._allFibersPrev = copy.deepcopy(self._allFibers); # initialize previous fiber network object
        self._CLNetwork = CLNetwork;
        self._impco = 0; #coefficient for linear solves
        self._maxGMIters = itercap;
    
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
    
                 
    def SolveForFiberAlphaLambda(self,XforNL,XsforNL,iT,dt,tvalSolve,forceExt,lamStar,Dom,Ewald):
        """
        This method solves the saddle point system for lambda and alpha. It is for DETERMINISTIC
        fibers, and is based on solving a block-diagonal system first, then using GMRES to solve
        for a residual lambda and alpha. See Section 4.5 here: https://arxiv.org/pdf/2007.11728.pdf 
        for more details on this. 
        """
        thist = time.time();
        ExForce, UNonLoc = self._allFibers.formBlockDiagRHS(XforNL,tvalSolve,forceExt,lamStar,Dom,Ewald);
        if (verbose > 0):
            print('Time to form RHS %f' %(time.time()-thist))
            thist = time.time()
            
        # Apply preconditioner to get (delta lambda,delta alpha) from the block diagonal solver
        self._allFibers.FactorizePreconditioner(XforNL,XsforNL,self._impco,dt);
        BlockDiagAnswer = self._allFibers.BlockDiagPrecond(UNonLoc,ExForce);
        if (verbose > 0):
            print('Time to apply preconditioner %f' %(time.time()-thist))
            thist = time.time()
        
        # Set answer = block diagonal or proceed with GMRES
        giters = self.getMaxIters(iT)-1; # subtract the first application of mobility/preconditioner
        if (giters==0 or isinstance(self._allFibers,SemiflexiblefiberCollection)):
            # Block diagonal acceptable 
            lamalph = BlockDiagAnswer;
            itsneeded = 0;
        else:
            # GMRES set up and solve
            RHS = self._allFibers.calcResidualVelocity(BlockDiagAnswer,XforNL,XsforNL,dt*self._impco,lamStar, Dom, Ewald);
            systemSize = self._allFibers.getSysDimension();
            BlockOne = self._allFibers.getBlockOneSize();
            BlockTwo = systemSize - BlockOne;
            A = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.Mobility,impcodt=self._impco*dt,\
                X=XforNL, Xs=XsforNL, Dom=Dom,RPYEval=Ewald),dtype=np.float64);
            Pinv = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.BlockDiagPrecond, \
                   ExForces=np.zeros(BlockOne)),dtype=np.float64);
            SysToSolve = LinearSystem(A,RHS,Mr=Pinv);
            Solution = TemporalIntegrator.GMRES_solve(SysToSolve,tol=GMREStolerance,maxiter=giters)
            lamalphT = Solution.xk;
            itsneeded = len(Solution.resnorms)
            if (verbose > 0): 
                resno = Solution.resnorms
                print(resno)
                print('Number of iterations %d' %len(resno))
                print('Last residual %f' %resno[len(resno)-1])
            if (itsneeded == itercap):
                resno = Solution.resnorms
                print('WARNING: GMRES did not actually converge. The error at the last residual was %f' %resno[len(resno)-1])
            lamalph=np.reshape(lamalphT,len(lamalphT))+BlockDiagAnswer;
            #res=self._allFibers.CheckResiduals(lamalph,self._impco*dt,XforNL,XsforNL,Dom,Ewald,tvalSolve,ExForces=ExForce);
            if (verbose > 0):
                print('Time to solve GMRES and update alpha and lambda %f' %(time.time()-thist))
                thist = time.time()
        return lamalph, itsneeded, XforNL;

    def updateAllFibers(self,iT,dt,numSteps,Dom,Ewald=None,gravden=0.0,outfile=None,write=1,\
        updateNet=False,turnoverFibs=False,BrownianUpdate=False,fixedg=None,stress=False):
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
            bornFibs = self._allFibers.FiberBirthAndDeath(dt);
            self._CLNetwork.deleteLinksFromFibers(bornFibs)
            if (verbose > 0):
                print('Time to turnover fibers (first time) %f' %(time.time()-thist))
                thist = time.time()    
        # Update the network (first time)
        if (updateNet):
            self.NetworkUpdate(Dom,iT*dt,dt,fixedg);
            if (verbose > 0):
                print('Time to update network (first time) %f' %(time.time()-thist))
                thist = time.time()
        # Brownian update (always first order)
        thist = time.time() 
        if (BrownianUpdate):
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
        self._gForStress = Dom.getg(); # Will get reset during hydro!
        #if (stress):
        #    print('g for stress is %f' %self._gForStress)
        
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
        
        # Solve for the (alpha, lambda) to update the fibers
        thist = time.time()
        # Solve the block diagonal way
        if (turnoverFibs): # reinitialize lambda for fibers that were turned over
            self._allFibers.initializeLambdaForNewFibers(bornFibs,forceExt,tvalSolve,dt,self._impco,self._allFibersPrev)    
        Dom.roundg(); # for nonlocal hydro, round g to [-1/2,1/2] (have to do this after computing CL forces)
        lamStar = self.getLamNonLoc(iT); # lambdas
        self._allFibersPrev = copy.deepcopy(self._allFibers); # copy old info over
        
        lamalph, itsneeded, XWithLam = self.SolveForFiberAlphaLambda(XforNL,XsforNL,iT,dt,tvalSolve,forceExt,lamStar,Dom,Ewald)
        #np.savetxt('ChebPts.txt',XforNL)
        #np.savetxt('TanVecs.txt',XsforNL)
        #np.savetxt('FCL.txt',forceExt)
        #np.savetxt('LambdaAlpha.txt',lamalph)

        # Update alpha and lambda and fiber positions
        self._allFibers.updateLambdaAlpha(lamalph);
        maxX = self._allFibers.updateAllFibers(dt);
        if (verbose > 0):
            print('Update fiber time %f' %(time.time()-thist))
            thist = time.time()
        
        stressArray = np.zeros(3);
        if (stress):
            ElasticStress, lamStress = self._allFibers.FiberStress(XforNL,XWithLam,Dom.getVol())
            # Stress due to CLs 
            Dom.setg(self._gForStress); 
            CLstress = 0
            if (self._CLNetwork is not None):
                CLstress = self._CLNetwork.CLStress(self._allFibers,XforNL,Dom);
            stressArray=np.array([lamStress[0,1],ElasticStress[0,1],CLstress[0,1]]);
            if (verbose > 0):
                print('Stress time %f' %(time.time()-thist))
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
 
    def __init__(self,fibCol,CLNetwork=None):
        super().__init__(fibCol,CLNetwork);
        self._impco = 1; #coefficient for linear solves
        if (isinstance(fibCol,SemiflexiblefiberCollection) and not isinstance(self,MidpointDriftIntegrator)):
            raise TypeError('The midpoint drift integrator (not backward Euler) must be used for fibers with bending fluctuations')

class CrankNicolson(TemporalIntegrator):

    """
    Crank Nicolson temporal discretization. The same as the abstract parent class, but 
    with a different implicit coefficient in the solves. 
    See parent class for method descriptions 
    """
    
    def __init__(self,fibCol,CLNetwork=None):
        super().__init__(fibCol,CLNetwork);
        self._impco = 0.5; # coefficient for implicit solves
        warn('Crank-Nicolson not tested in new discretization. Backward Euler recommended.')
      
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

class MidpointDriftIntegrator(BackwardEuler):

    """
    This is the midpoint drift integrator that is intended for use 
    with semiflexible bending fluctuations. Basically, it is based on generating a
    Brownian velocity, then inverting K to step to the midpoint, then solving the 
    saddle point system we usually solve at the midpoint.
    More details can be found in Section 3 here: https://arxiv.org/pdf/2301.11123.pdf
    """ 

    def __init__(self,fibCol,CLNetwork=None):
        super().__init__(fibCol,CLNetwork);
        self._nLanczos=[];
        self._ModifyBE=True;
        if (not isinstance(fibCol,SemiflexiblefiberCollection)):
            raise TypeError('The midpoint drift integrator is for fibers with bending fluctuations only!')
        
    def SolveForFiberAlphaLambda(self,XforNL,XsforNL,iT,dt,tvalSolve,forceExt,lamStar,Dom,Ewald):
        """
        This is the method that gives alpha and lambda on the fibers using the midpoint solve. 
        The steps are outlined in the columns below. 
        """
        thist = time.time()  
        # Compute the external flow and force we treat explicitly (this includes the bending force 
        # at time n)
        ExForce, U0 = self._allFibers.formBlockDiagRHS(XforNL,tvalSolve,forceExt,lamStar,Dom,Ewald);
        if (verbose > 0):
            print('Time to form RHS %f' %(time.time()-thist))
            thist = time.time()  
        # Compute M^(1/2)*W for later use
        MHalfEta, MMinusHalfEta = self._allFibers.MHalfAndMinusHalfEta(XforNL,Ewald,Dom);
        #self._nLanczos.append(nLanczos);
        if (verbose > 0):
            print('Time to do M^1/2 %f' %(time.time()-thist))
            thist = time.time()  
        # Use M^(1/2)*W to step to the midpoint
        TauMidtime, MPMidtime, XMidtime = self._allFibers.StepToMidpoint(MHalfEta,dt);
        self._allFibers.FactorizePreconditioner(XMidtime,TauMidtime,self._impco,dt);
        if (verbose > 0):
            print('Time to step to MP %f' %(time.time()-thist))
            thist = time.time()  
        # Compute the additional velocity for the saddle point system. This includes 1) M^(1/2)*W (the 
        # normal Brownian velocity) + Drift terms + terms for modified backward Euler.
        UBrown = self._allFibers.DriftPlusBrownianVel(XMidtime, MHalfEta, MMinusHalfEta,dt,self._ModifyBE,Dom,Ewald);
        if (verbose > 0):
            print('Time to calc drift %f' %(time.time()-thist))
            thist = time.time()
        if (self._allFibers._nonLocal==1): 
            # GMRES set up and solve
            # Set the shear equal to the midtime shear
            Dom.setg(self._allFibers.getg((iT+0.5)*dt)) 
            systemSize = self._allFibers.getSysDimension();
            BlockOne = self._allFibers.getBlockOneSize();
            BlockTwo = systemSize - BlockOne;
            # Add velocity from external forcing
            UExForce = self._allFibers.ComputeTotalVelocity(XMidtime,ExForce,Dom,Ewald);
            RHS = np.concatenate((UBrown+U0+UExForce,np.zeros(BlockTwo)));
            A = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.Mobility,impcodt=self._impco*dt,\
                X=XMidtime, Xs=TauMidtime, Dom=Dom,RPYEval=Ewald),dtype=np.float64);
            Pinv = LinearOperator((systemSize,systemSize), matvec=partial(self._allFibers.BlockDiagPrecond, \
                   ExForces=np.zeros(BlockOne)),dtype=np.float64);
            SysToSolve = LinearSystem(A,RHS,Mr=Pinv);
            Solution = TemporalIntegrator.GMRES_solve(SysToSolve,tol=GMREStolerance,maxiter=itercap)
            lamalph = np.reshape(Solution.xk,systemSize);
            itsneeded = len(Solution.resnorms)
            if (verbose > 0): 
                resno = Solution.resnorms
                print(resno)
                print('Number of iterations %d' %len(resno))
                print('Last residual %1.1E' %resno[len(resno)-1])
            if (itsneeded == itercap):
                resno = Solution.resnorms
                print('WARNING: GMRES did not actually converge. The error at the last residual was %1.1E' %resno[len(resno)-1])
            #res=self._allFibers.CheckResiduals(lamalph,self._impco*dt,XMidtime,TauMidtime,Dom,Ewald,tvalSolve,UBrown,ExForce);
            #print(np.amax(abs(res)))
        else:
            # Just apply block-diag precond
            lamalph = self._allFibers.BlockDiagPrecond(UBrown+U0,ExForce);
            itsneeded=0;
        if (verbose > 0):
            print('Time to solve system %f' %(time.time()-thist))
        return lamalph, itsneeded, XMidtime;

