import numpy as np
from FiberDiscretization import FiberDiscretization

class DiscretizedFiber(object):
    """
    Object that stores the points and forces for each fiber. 
    That is X_n, X_{n-1}, Xs_n, Xs_{n-1}, lambda_{n-1/2}, lambda_{n-3/2}
    are the only things stored here.
    """
    def __init__(self,Discretization,X=None,Xs=None):
        """
        Constructor. Object variables are:
        Discretization = FiberDiscretization object that has all the 
        Chebyshev information, X, Xs = (N x 3) arrays that contain 
        the initial X and Xs (otherwise they are initialized randomly
        as a straight fiber).
        """
        self._fibDisc = Discretization;
        self._X = X;
        self._Xs = Xs;
        self._lambdas = np.zeros((3*self._fibDisc.getN()));

    ## METHODS FOR INITIALIZATION AND PRECOMPUTATION
    def initFib(self,Lengths):
        """
        Initialize straight fiber locations and tangent 
        vectors on [0,Lx] x [0,Ly] x [0,Lz]
        """
        # Choose a random start point
        start = np.random.rand(3)*Lengths;
        u=1-2*np.random.rand();
        v=np.sqrt(1-u*u);
        w=2*np.pi*np.random.rand();
        tangent = np.array([v*np.cos(w), v*np.sin(w), u]);
        N = self._fibDisc.getN();
        self._Xs = np.tile(tangent,N);
        self._X = np.reshape(start+np.reshape(self._fibDisc._s,(N,1))*tangent,3*N);

    def initPastVariables(self):
        """
        Initialize the previous values. This has to be done 
        after self._X and self._Xs are initialzied
        """
        self._lambdaprev = self._lambdas.copy();
        self._Xprev = self._X.copy();
        self._Xsprev = self._Xs.copy();

    ## GETTER METHODS
    def getLocs(self):
        return self._X;
    
    def getFibDisc(self):
        """
        Return the FiberDiscretization object
        """
        return self._fibDisc;

    def getXandXs(self,tint=2):
        """
        Get the position arguments necessary for the non-local solves.
        Inputs: tint = temporal integrator (2 for Crank-Nicolson, 
        1 for backward Euler, -1 for forward Euler), that tells
        the method whether to do LMM combinations for Xs and X)
        Outputs: X and Xs needed for the nonlocal solves.
        """
        N = self._fibDisc.getN(); # to save writing
        addxs = 0.5*(np.abs(tint)-1);
        Xsarg = np.reshape((1+addxs)*self._Xs - addxs*self._Xsprev,(N,3));
        Xarg = np.reshape((1+addxs)*self._X - addxs*self._Xprev,(N,3));
        return Xarg, Xsarg;
    
    def getfeandlambda(self,tint=2,addlam=1):
        """
        Get the force arguments necessary for the non-local solves.
        Inputs: tint = temporal integrator (2 for Crank-Nicolson, 
        1 for backward Euler, -1 for forward Euler), that tells
        the method whether to do LMM combinations for fE)
        addlam = whether to do the LMM for lambda (1 to do the LMM, 
        0 otherwise)
        """
        N = self._fibDisc.getN(); # to save writing
        addxs = 0.5*(np.abs(tint)-1);
        fEarg = np.reshape((1+addxs)*self._fibDisc.calcfE(self._X) - \
            addxs*self._fibDisc.calcfE(self._Xprev),(N,3));
        lamarg = np.reshape((1+addlam)*self._lambdas- \
            addlam*self._lambdaprev,(N,3));
        return fEarg, lamarg;
    
    ## METHODS TO UPDATE THE FIBER POSITIONS AND TANGENT VECTORS
    def alphaLambdaSolve(self,dt,tint,iters,nLvel,exF):
        """
        This method solves the linear system for 
        lambda and alpha for a given RHS. 
        Inputs: dt = timestep, tint = type of temporal integrator (2 for
        Crank-Nicolson, 1 for backward Euler, -1 for forward Euler), 
        iters= iteration count (really necessary to copy the previous lambda at the 
        right point), nLvel = non-local velocity as a 3N vector 
        (comes from the fiberCollection class), exF = external forces as a 3N vector
        (comes from the ExForces class)
        """
        addxs = 0.5*(np.abs(tint)-1);
        Xsarg = (1+addxs)*self._Xs - addxs*self._Xsprev;
        if (iters==0):
            self._lambdaprev = self._lambdas.copy();
        self._lambdasm1 = self._lambdas.copy();
        self._alphaU, self._vel, self._lambdas =\
            self._fibDisc.alphaLambdaSolve(self._X,Xsarg,dt,tint,nLvel,exF);
    
    def notConverged(self,tol):
        """
        Check if fixed point iteration is converged with tolerance tol. 
        Return 1 if NOT converged and 0 if converged. 
        """
        diff = self._lambdas-self._lambdasm1;
        # The 1.0 is here so that if lambda = 0 the iteration will converge. 
        reler = np.linalg.norm(diff)/max(1.0,np.linalg.norm(self._lambdas));
        return (reler > tol);
    
    def updateXsandX(self,dt,tint,exactinex=1):
        """ 
        Update the fiber position and tangent vectors
        Inputs: dt = timestep and tint = temporal integrator.
        If exactinex = 1, the inextensibility will be done
        EXACTLY according to the Rodriguez rotation formula. 
        Otherwise, it will be inexact. 
        """
        N = self._fibDisc.getN(); # to save writing
        if (not exactinex): # inexact update, just add the velocity*dt to X
            self._Xprev = self._X.copy();
            self._X+= dt*self._vel;
            self._Xsprev = self._Xs.copy();
            self._Xs = np.reshape(np.dot(self._fibDisc._Dmat,\
                        np.reshape(self._X,(N,3))),3*N);
            return;
        # For exact inextensbility
        addxs = 0.5*(np.abs(tint)-1);
        XsForOmega = (1+addxs)*self._Xs - addxs*self._Xsprev;
        # Update positions from tangent vectors
        Xnp1, Xsp1 = self._fibDisc.XFromXs(self._Xs, XsForOmega, self._alphaU, dt);
        # Add the constant velocity so that point 0 is the same
        Xnp1+= -Xnp1[0,:]+self._X[0:3]+dt*self._vel[0:3]
        self._Xprev = self._X.copy();
        self._Xsprev = self._Xs.copy();
        self._X = np.reshape(Xnp1,3*N);
        self._Xs = np.reshape(Xsp1,3*N);
    
    def writeLocs(self,of):#outFile,wa='a'):
        """
        Write the locations to a file object of. 
        """
        #of = open(outFile,wa);
        N = self._fibDisc.getN(); # to save writing
        for i in range(N):
            for j in range(3):
                of.write('%14.15f     ' % self._X[3*i+j]);
            of.write('\n');
        #of.close();
        
    
