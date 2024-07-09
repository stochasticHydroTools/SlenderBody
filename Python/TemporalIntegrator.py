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
itercap = 10; # cap on GMRES iterations if we converge all the way
GMREStolerance=1e-3; # larger than GPU tolerance
verbose = 1;

class TemporalIntegrator(object):

    """
    Class to do temporal integration. 
    There are three possibilities implemented:
    
    1) Backward Euler - first order accuracy for deterministic fibers
    2) Crank Nicolson - "second order" accuracy for deterministic fibers. However,
       this temporal integrator is very sensitive because it uses linear multistep formulas
       to get higher-order accuracy. We do not use it any more for this reason.
    3) Midpoint Drift integrator - for filaments with semiflexible bending fluctuations.

    This abstract class is a first order explicit method. Because of the stiffness
    of the bending force, such a method is not really practical, and so the point of the 
    abstract class is to declare a list of methods, as well as implement methods that 
    are general to a bending force argument of the form $(1-c)X^n+cX^{n+1}$. In this 
    abstract class, $c=0$. 
    
    For more details on temporal integration, see Sections 6.4 and 7.3 (deterministic fibers)
    and 8.2 (fluctuating fibers) of Maxian's PhD thesis.
    
    Members
    --------
    self._allFibers: fiberCollection object
        This allows us to perform updates on the entire set of fibers
    self._CLNetwork: EndedCrossLinkedNetwork object
        This allows us to perform updates on the set of dynamic cross linkers
    """
    
    def __init__(self,fibCol,CLNetwork=None):
        """
        Constructor. Initialize the objects that we will be working with. 
        
        Parameters
        ----------
        fibCol: fiberCollection object
        CLNetwork: EndedCrossLinkedNetwork object, optional
            Only input if we are considering fibers with cross linkers. If CLNetwork
            is None, the code will only update the fibers.
        """
        self._allFibers = fibCol;
        self._allFibersPrev = copy.deepcopy(self._allFibers); # initialize previous fiber network object
        self._CLNetwork = CLNetwork;
        self._impco = 0; #coefficient for linear solves
        self._maxGMIters = itercap;
    
    def getXandXsNonLoc(self):
        """
        Get the positions and tangent vectors for the nonlocal hydro.
        Specifically, depending on the temporal integrator, sometimes 
        we evaluate $M_{NL}\\left(X_{NL}\\right)$, where $X_{NL}$ are 
        the positions we use for nonlocal hydrodynamics. In a first-order
        method, we simply use $X_{NL}=X^n$ (the current positions), so this
        is really nothing fancy.
        
        Returns
        --------
        (array, array)
            The positions $X$ and tangent vectors $\\tau$ that we will
            use for nonlocal hydrodynamics.
        """
        return self._allFibers.getX().copy(), self._allFibers.getXs().copy();

    def getLamNonLoc(self,iT):
        """
        Get the constraint forces $\\Lambda$ for the nonlocal hydro. In some
        of our deterministic methods, we time-lag the forces $\\Lambda$ to
        input to the nonlocal hydrodynamics. In a first order method, this 
        simply gets the $\\Lambda$ from the end of the previous time step.
        
        Parameters
        -----------
        iT: int
            The time step index
    
        Returns
        --------
        array
            The constraint forces $\\Lambda$ that we use as an initial guess
            for the nonlocal hydrodynamics.
        """
        return self._allFibers.getLambdas().copy();
        
    def setMaxIters(self,nIters):
        """
        Pass in the maximum number of GMRES iterations.
        
        Parameters
        -----------
        nIters: int
            Set to 1 to do block diagonal solves only. Every number
            larger than 1 then gives an additional GMRES iteration.
        """
        self._maxGMIters = nIters;

    def getMaxIters(self,iT):
        """
        The maximum number of GMRES iterations at a given time step.
        In a first-order method, at the first time step
        we set the maximum number of iterations 
        to a large number, so that we converge GMRES all the way at $t=0$. 
        After that, we set the cap at self._maxGMIters
        
        Parameters
        -----------
        iT: int
            The time step index (really we just need to know if it's zero or not)
        
        Returns
        ---------
        int
            The maximum number of GMRES iterations we perform at this time step
        """
        if (iT==0):
            return itercap; # converge all the way
        return self._maxGMIters;
           
    def gettval(self,iT,dt):
        """
        Get time value for this time step. This is useful when we have
        functions of time, like strain and background flow.
        
        Parameters
        ----------
        iT: int
            Time step index
        dt: double
            Time step size
            
        Returns
        --------
        double
            The time argument for functions of time. In a first order method,
            this is simply iT*dt.
        """
        return iT*dt;
        
    def NetworkUpdate(self,Dom,t,tstep,fixedg=None):
        """
        This is the method to update the CL network by calling
        the corresponding update method in self._CLNetwork.
        
        Parameters
        -----------
        Dom: Domain object
            To set the periodicity of the cross linking
        t: double
            The current simulation time
        tstep: double
            Time step to take for the network update
        fixedg: double, optional
            If set, it will fix the strain of the network at the input
            value. Otherwise, the strain is set by what the background 
            flow at time $t$ dictates.
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
        This method solves the saddle point system for lambda and alpha. It is for deterministic
        fibers, and is based on first solving the saddle point system
        $$
        \\begin{pmatrix}
        -M_L & K + c \\Delta t M_L L K \\\\
        K^T & 0 
        \\end{pmatrix}
        \\begin{pmatrix}
        \\widetilde{\\Lambda} \\\\
        \\widetilde{\\alpha}
        \\end{pmatrix} = 
        $$
        $$
        \\begin{pmatrix}
        M_L \\left(-LX^n + F_{ext}\\right)+M_{NL} \\left(\\Lambda^*-LX^* + F_{ext}\\right)+U_0 \\\\
        0
        \\end{pmatrix}
        $$
        This represents a system where the nonlocal hydrodynamics uses a time-lagged "guess"
        for $\\Lambda=\\Lambda^*$ and $X=X^*$. Thus the nonlocal hydrodynamics is handled 
        explicitly. In the case when all of the hydrodynamics is local, i.e., $M=M_L$, the solution
        of this sytem is the dynamics we are interested in. So we set $\\alpha=\\widetilde{\\alpha}$
        and move on.
        
        When there is nonlocal hydrodynamics, it is possible (for denser suspensions)
        that the above method will not be stable. In this case we instead want to treat the 
        nonlocal hydrodynamic bending force implicitly and solve the saddle point system
        $$
        \\begin{pmatrix}
        -M & K + c \\Delta t M L K \\\\
        K^T & 0 
        \\end{pmatrix}
        \\begin{pmatrix}
        \\Lambda \\\\
        \\alpha
        \\end{pmatrix} = 
        $$
        $$
        \\begin{pmatrix}
        M \\left(-LX^n + F_{ext}\\right)+U_0 \\\\
        0
        \\end{pmatrix}
        $$
        where $M=M_L+M_{NL}$, i.e., there is no difference here between the local and nonlocal
        mobilities. Because the nonlocal mobility is involved in a linear system, this system 
        has to be solved iteratively via GMRES. To do this, we subtract the system above 
        from the system preceding it, so that the "residual" system we are solving is given by
        $$
        \\begin{pmatrix}
        -M & K + c \\Delta t M L K \\\\
        K^T & 0 
        \\end{pmatrix}
        \\begin{pmatrix}
        \\Delta \\Lambda \\\\
        \\Delta \\alpha
        \\end{pmatrix} = 
        $$
        $$
        \\begin{pmatrix}
        M_{NL} \\left(-L\\left(X^n +c K \\widetilde{\\alpha} - X^*\\right)+\\widetilde{\\Lambda}-\\Lambda^*\\right) \\\\
        0
        \\end{pmatrix}
        $$
        where $\\Delta \\alpha = \\alpha - \\widetilde{\\alpha}$ and likewise for $\\Lambda$.
        What we do in practice is to perform a fixed number of GMRES iterations to estimate
        $\\Delta \\Lambda$ and $\\Delta \\alpha$ (necessary for stability and not accuracy). 
        The number of iterations is set using the getMaxIters method above. 
        
        Parameters
        -----------
        XforNL: array
            The positions $X$ that are arguments for the mobilities. In the equations above, 
            we have $M=M(X)$. This argument specifies the $X$ that we evaluate $M$ at. Typically, 
            in a first order method this is just $X=X^n$. 
        XsforNL: array
            The tangent vectors $\\tau$ that are arguments for the kinematic matrix $K$. In the 
            equations above, $K=K(\\tau)$, and this argument specifies what $\\tau$ we evaluate $K$ at.
            Typically, in a first order method this is just $\\tau=\\tau^n$. 
        iT: int
            The time step index
        dt: double
            The time step size
        tvalSolve: double
            The time we use to evaluate any functions of time like the background flow. See the 
            method gettval() above for an explanation. 
        forceExt: array
            The explicit forcing $F_{ext}$ that enters the saddle point system above
        lamStar: array
            The guess for the constraint force $\\Lambda^*$ that appears in the first saddle 
            point system and the RHS of the residual saddle point system
        Dom: Domain object
            The periodic domain where we perform the nonlocal velocity evaluations
        Ewald: RPYVelocityEvaluator object
            The object that evaluates the nonlocal fluid velocity
            
        Returns
        --------
        (array, int, array)
            The first argument returned is the solution of the saddle point 
            system, $\\left(\\Lambda,\\alpha\\right)=\\left(\\widetilde{\\Lambda}+\\Delta \\Lambda, \widetilde{\\alpha}+\\Delta \\alpha \\right)$.
            The second argument is the number of GMRES iterations required/used to solve the saddle 
            point system.
            The third argument is the positions $X$ used to solve the saddle point system. These are
            needed later when computing the stress due to the constraint forces.
        
        """
        thist = time.time();
        ExForce, UNonLoc = self._allFibers.formBlockDiagRHS(XforNL,self._allFibers.getX(),tvalSolve,forceExt,lamStar,Dom,Ewald);
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
        if (giters==0):
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
            gtol = GMREStolerance;
            Solution = TemporalIntegrator.GMRES_solve(SysToSolve,tol=gtol,maxiter=giters)
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
            #print('RHS max %f' %np.amax(np.abs(RHS)))
            res=self._allFibers.CheckResiduals(lamalph,self._impco*dt,XforNL,XsforNL,Dom,Ewald,tvalSolve,ExForces=ExForce);
            #print(np.amax(np.abs(res)))
            #resno = Solution.resnorms
            #print(resno)
            #print('Last residual %1.2E' %resno[len(resno)-1])
            #np.savetxt('LamAlph.txt',lamalph)
            #import sys
            #sys.exit()
            if (verbose > 0):
                print('Time to solve GMRES and update alpha and lambda %f' %(time.time()-thist))
                thist = time.time()
        return lamalph, itsneeded, XforNL;

    def updateAllFibers(self,iT,dt,numSteps,Dom,Ewald=None,gravden=0.0,outfile=None,write=False,\
        updateNet=False,turnoverFibs=False,BrownianUpdate=False,fixedg=None,stress=False,StericEval=None):
        """
        This is the main update method which updates the fiber collection and cross linked network. 
        The method proceeds in the following order:
        
        1) Turnover the fibers over time step $\\Delta t$
        2) Update the network of dynamic cross linkers over time step $\\Delta t$
        3) Perform a rigid body diffusion of the fibers over time step $\\Delta t$
        4) Obtain the arguments $X^*$ and $\\tau^*$ for the solve, and evaluate the 
           external forcing (cross linking, sterics, and/or gravity) at those arguments.
        5) Initialize $\\Lambda$ for newly turned-over fibers, then evaluate the 
           argument $\\Lambda^*$ for the guess constraint forces
        6) Solve for the fiber evolution velocity, paramerized by $\\alpha$ (see method
           SolveForFiberAlphaLambda) 
        7) Update all fibers by performing rotation of their tangent vectors and integrating
           to obtain the positions
        8) Compute the stress in the system and return that
        9) Write the locations at the end of the time step to a file

        Of course, there are options in this method to turn each of these updates off, as follows.
        
        Parameters
        ----------
        iT: int
            The time step index
        dt: double
            The time step size
        numSteps: int
            The maximum number of steps we will take
        Dom: Domain object
            The (periodic) domain we are solving on
        Ewald: RPYVelocityEvaluator object, optional
            The object that evaluates nonlocal flows. Defaults to none (local flows only)
        gravden: double, optional
            Uniform gravitational force strength in the z direction. Defaults to zero.
        outfile: string, optional
            Name of the output file to write the fiber locations to. Defaults to none.
        write: bool, optional
            Whether to write the locations to a file. Defaults to true.
        updateNet: bool, optional
            Whether to update the network of dynamic linkers. Defaults to false.
        turnoverFibs: bool, optional
            Whether to turn over the fibers. Defaults to false.
        BrownianUpdate: bool, optional
            Whether to perform rigid body translations and rotations as a separate update
            on the fiber locations. Defaults to false.
        fixedg: double, optional
            If we want to perform the solve on a domain with a fixed strain $g$, rather than
            that dictated by the background flow. Defaults to None, in which case the value
            of $g$ is dictated by the background flow.
        stress: bool, optional
            Whether to compute the stress in the suspension
        StericEval: StericForceEvaluator object
            the object used to compute the steric forces. Default is None, in which case
            steric forces will not be included in the calculation
        
        Returns
        --------
        (double, int, array, int)
            The first returned argument is the maximum absolute position of the fibers at the end
            of the time step (this is used to check stability). The second is the number of iterations
            needed to solve the GMRES system at that time step. The third is the $3 \\times 3$ array of
            the stress (due to the fibers only, not counting the background fluid stress) in the 
            suspension, if it is computed (otherwise it returns all zeros). The last output is the 
            number of contacts between the fibers, as measured by resampling to uniform points. This 
            last output is zero unless StericEval is provided to compute steric forces and evaluate 
            contacts.
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
        
        nContacts = 0;
        if (StericEval is not None):
            Touching, _ = StericEval.CheckContacts(XforNL,Dom, excludeSelf=True);
            nContacts, _ = Touching.shape;
            forceExt +=StericEval.StericForces(XforNL,Dom);
            #print('Number of contacts %d' %nContacts)
            if (verbose > 0):
                print('Time to calc steric force %f' %(time.time()-thist))
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
        
        stressArray = np.zeros(4);
        if (stress):
            ElasticStress, lamStress, DriftStress = self._allFibers.FiberStress(XforNL,XWithLam,XsforNL,Dom.getVol())
            # Stress due to CLs 
            Dom.setg(self._gForStress); 
            CLstress = np.zeros((3,3));
            if (self._CLNetwork is not None):
                CLstress = self._CLNetwork.CLStress(self._allFibers,XforNL,Dom);
            stressArray=np.array([lamStress[0,1],ElasticStress[0,1],DriftStress[0,1],CLstress[0,1]]);
            if (verbose > 0):
                print('Stress time %f' %(time.time()-thist))
                thist = time.time()
                
                   
        if (write):
            self._allFibers.writeFiberLocations(outfile);
        return maxX, itsneeded, stressArray, nContacts;    
     
    
    @staticmethod
    def GMRES_solve(linear_system, **kwargs):
        """
        Run GMRES solver and catch convergence errors. This uses
        mykrypy, a modified version of the python package krypy, 
        which can be found in the Dependencies folder.
        """
        try:
            return Gmres(linear_system, **kwargs)
        except ConvergenceError as e:
            # use solver of exception
            return e.solver    

class BackwardEuler(TemporalIntegrator):

    """
    Backward Euler temporal discretization. The same as the abstract parent class, but 
    with a different implicit coefficient in the solves. Specifically, this 
    class has all of the same methods as the abstract class, but sets the 
    implicit coefficient $c=1$.
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
    Specifically, this class has all of the same methods as the abstract class, but sets the
    implicit coefficient $c=1/2$. There are also some other modifications as documented below.
    """
    
    def __init__(self,fibCol,CLNetwork=None):
        super().__init__(fibCol,CLNetwork);
        self._impco = 0.5; # coefficient for implicit solves
      
    def getXandXsNonLoc(self):
        """
        In our second-order discretization, the arguments we use for the solve are
        $X^*=1.5X^n-0.5X^{n-1}$ and $\\tau^*=1.5\\tau^n-0.5\\tau^{n-1}$. This method
        overwrites the base class implementation to return those arguments.
        """
        X = 1.5*self._allFibers.getX() - 0.5*self._allFibersPrev.getX();
        Xs = 1.5*self._allFibers.getXs() - 0.5*self._allFibersPrev.getXs();
        return X, Xs;
    
    def getLamNonLoc(self,iT):
        """
        In our second-order discretization, the constraint force guess for nonlocal
        hydrodynamics is given by $\\Lambda^*=2\\Lambda^{n-1/2}-\\Lambda^{n-3/2}$ 
        (the constraint forces are obtained at the midpoint in the second-order scheme)
        This method returns this $\\Lambda^*$, unless we are at the first or second time
        step, in which case we don't have access to the previous $\\Lambda$ and we will
        solve the GMRES system fully, making the initial guess unimportant.
        """
        if (iT > 1):
            lamarg = 2.0*self._allFibers.getLambdas()-1.0*self._allFibersPrev.getLambdas();
        else: # t = 0
            lamarg = self._allFibers.getLambdas().copy();
        return lamarg;

    def getMaxIters(self,iT):
        """
        In the second-order scheme, we solve the GMRES system fully in the first two 
        time steps, then converge partially after that, when we can rely on our
        extrapolations to help us get second order accuracy.
        """
        if (iT < 2):
            return itercap; # converge all the way
        return self._maxGMIters;

    def gettval(self,iT,dt):
        """
        In the second-order scheme, the argument for any functions of time 
        is $(n+1/2)\\Delta t$. This evaluates the functions at the midpoint of the time 
        step
        """
        return (iT+0.5)*dt;


class MidpointDriftIntegrator(BackwardEuler):

    """
    This is the midpoint drift integrator that is intended for use
    with semiflexible bending fluctuations. It overwrites the method
    SolveForFiberAlphaLambda to account for the fluctuations in the 
    filament tangent vectors, as well as include the proper drift terms
    in the overdamped Langevin dynamics
    """ 

    def __init__(self,fibCol,CLNetwork=None):
        """
        The constructor is the same as in the abstract class. 
        In addition to setting the fiberCollection and CLNetwork objects,
        in this constructor we can also specify whether to use a modified backward
        Euler method for the Brownian velocity (i.e., whether to add a term in the
        velocity that is O(1) (compared to O(1/$\\sqrt{\\Delta t}$)) to make the 
        covariance of the fluctuations be more accurate -- see Section 8.2.1 in
        Maxian's PhD thesis.
        """
        super().__init__(fibCol,CLNetwork);
        self._ModifyBE=True;
        if (not isinstance(fibCol,SemiflexiblefiberCollection)):
            raise TypeError('The midpoint drift integrator is for fibers with bending fluctuations only!')
        
    def SolveForFiberAlphaLambda(self,XforNL,XsforNL,iT,dt,tvalSolve,forceExt,lamStar,Dom,Ewald):
        """
        This is the method that gives $\\alpha$ and $\\Lambda$ on the fibers using a "midpoint"
        temporal integrator to correctly capture the fluctuations and drift terms. The specific 
        order of steps is given in Section 8.3 of Maxian's thesis. What we do is to solve the 
        saddle point system
        $$
        \\begin{pmatrix}
        -M & K + c \\Delta t M L K \\\\
        K^T & 0 
        \\end{pmatrix}^{n+1/2,*}
        \\begin{pmatrix}
        \\Lambda \\\\
        \\alpha
        \\end{pmatrix}^{n+1/2,*} =
        $$
        $$
        \\begin{pmatrix}
        M^{n+1/2,*} \\left(-LX^n + F_{ext}^n\\right)+U_0^n +U_B^n+U_{MD}^n\\\\
        0
        \\end{pmatrix}
        $$
        for $\\alpha$ and $\\Lambda$. If there is nonlocal (inter-fiber) hydrodynamics, 
        we solve this system by converging GMRES to tolerance GMREStolerance (given
        at the top of this file) NOT running a fixed number of iterations. If there is 
        only intra-fiber (local) hydrodynamics, we solve the saddle point system by dense
        linear algebra on each fiber separately.
        
        In the saddle point solve, the mobility $M$ and kinematic matrix $K$ are evaluated at $X^{n+1/2,*}$,
        which is a guess for the midpoint positions, obtained by computing
        $$
        \\alpha^{n,*}=\\sqrt{\\frac{2k_BT}{\\Delta t}} \\left(K^n\\right)^\\dagger \\left(M^n\\right)^{1/2}W^n
        $$
        and updating the fiber by $\\Delta t/2 \\alpha^{n,*}=\\Delta t/2 \\left(\\Omega^{n,*},U_{mp}^{n,*}\\right)$,
        i.e., rotate the tangent vectors by $\\Delta t/2\\Omega^{n,*}$ 
        and translate the midpoint by $\\Delta t/2 U_{mp}^{n,*}$. 
        
        In the saddle point system, there are two additional terms to specify. The first is the 
        Brownian velocity, which in the modified backward Euler method is given by
        $$
        U_B^n =\\sqrt{\\frac{2k_BT}{\\Delta t}}\\left(\\left(M^n\\right)^{1/2}W+\\sqrt{\\frac{\\Delta t}{2}} M^n L^{1/2} W_2\\right),
        $$
        where $L$ is the bending energy matrix.
        
        The formula for the drift term $U_{MD}^n$ depends on the type of hydrodynamics being considered.
        If we are considering only LOCAL hydrodynamics (no inter-fiber communication), it is 
        given by
        $$U_{MD}^n=\\sqrt{\\frac{2 k_B T}{\\Delta t}} \\left(M^{n+1/2,*}-M^n\\right)\\left(M^n\\right)^{-T/2}\\eta,$$
        where $\\eta \\sim$randn(0,1) and this formula is computed using dense linear algebra on
        each fiber separately. 
        In the case when there is inter-fiber hydrodynamics, this resistance problem becomes 
        expensive, and so we use an alternative approach, computing
        $$U_{MD}^n = \\frac{k_B T}{\\delta L} \\left(M\\left(\\tau^{(RFD)}\\right)-M\\left(\\tau^n\\right)\\right)\\eta$$
        where the RFD for $\\tau$ is obtained by computing $\\mu=K^\dagger \\eta$ and rotating 
        $\\tau^n$ by the oriented angle $\\delta L \\mu$.
        
        Parameters
        -----------
        XforNL: array
            The positions $X$ that are arguments for the mobilities. In the equations above, 
            we have $M=M(X)$. This argument specifies the $X$ that we evaluate $M$ at.
            In this method $X=X^n$.
        XsforNL: array
            The tangent vectors $\\tau$ that are arguments for the kinematic matrix $K$. In the 
            equations above, $K=K(\\tau)$, and this argument specifies what $\\tau$ we evaluate $K$ at.
            In this method $\\tau=\\tau^n$.
        iT: int
            The time step index
        dt: double
            The time step size
        tvalSolve: double
            The time we use to evaluate any functions of time like the background flow. In this
            first order method it's equal to iT*dt
        forceExt: array
            The explicit forcing $F_{ext}$ that enters the saddle point system above
        lamStar: array
            Not used in this method (only in deterministic base class).
        Dom: Domain object
            The periodic domain where we perform the nonlocal velocity evaluations
        Ewald: RPYVelocityEvaluator object
            The object that evaluates the nonlocal fluid velocity
            
        Returns
        --------
        (array, int, array)
            The first argument returned is the solution of the saddle point 
            system, $\\left(\\Lambda,\\alpha\\right)=\\left(\\widetilde{\\Lambda}+\\Delta \\Lambda, \widetilde{\\alpha}+\\Delta \\alpha \\right)$.
            The second argument is the number of GMRES iterations required/used to solve the saddle 
            point system.
            The third argument is the positions $X$ used to solve the saddle point system. These are
            needed later when computing the stress due to the constraint forces.
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
        #np.savetxt('MHalfEta.txt',MHalfEta);
        MHalfEta = np.loadtxt('MHalfEta.txt');
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
            UExForce = self._allFibers.nonLocalVelocity(XMidtime,ExForce,Dom,Ewald,subSelf=False);
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
            if (itsneeded == itercap+1):
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
        np.savetxt('LamAlph.txt',lamalph);
        return lamalph, itsneeded, XMidtime;

