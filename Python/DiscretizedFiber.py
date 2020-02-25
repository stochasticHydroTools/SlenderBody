import numpy as np

class DiscretizedFiber(object):
    """
    Object that stores the points and forces for each fiber. 
    That is X and Xs are the only things stored here, as well
    as the discretization of the fiber. 
    """
    ## ===============================================
    ##          METHODS FOR INITIALIZATION
    ## ===============================================
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

    ## ===============================================
    ##              PUBLIC METHODS
    ## ===============================================
    def initFib(self,Lengths):
        """
        Initialize straight fiber locations and tangent 
        vectors on [0,Lx] x [0,Ly] x [0,Lz]. 
        Input: Lengths = 3 vector [Lx,Ly,Lz]
        """
        # Choose a random start point
        start = np.random.rand(3)*Lengths;
        # Tangent vector (constant along fiber initially)
        u=1-2*np.random.rand();
        v=np.sqrt(1-u*u);
        w=2*np.pi*np.random.rand();
        tangent = np.array([v*np.cos(w), v*np.sin(w), u]);
        N = self._fibDisc.getN();
        # Initial Xs and X
        self._Xs = np.tile(tangent,N);
        self._X = np.reshape(start+np.reshape(self._fibDisc._s,(N,1))*tangent,3*N);

    def getXandXs(self):
        """
        Get the position and tangent vector as reshaped N x 3 arrays. 
        """
        N = self._fibDisc.getN(); # to save writing
        return np.reshape(self._X,(N,3)), np.reshape(self._Xs,(N,3));
    
    def updateXsandX(self,dt,velocity,XsForOmega,alpha,exactinex=1):
        """ 
        Update the fiber position and tangent vectors
        Inputs: dt = timestep, velocity = 3N vector velocity of X (used to 
        keep track of the constant), XsForOmega = 3N array of the Xs argument
        to construct the rotation angle for Xs, alpha = 2N-2 array of the alphas 
        from the solve, exactinex = 1 to preserve exact inextensibility
        or 0 to just update X and compute Xs from X. 
        """
        N = self._fibDisc.getN(); # to save writing
        if (not exactinex): # inexact update, just add the velocity*dt to X
            self._X+= dt*velocity;
            self._Xs = np.reshape(np.dot(self._fibDisc._Dmat,\
                        np.reshape(self._X,(N,3))),3*N);
            return;
        # For exact inextensbility
        # Update positions from tangent vectors
        Xnp1, Xsp1 = self._fibDisc.XFromXs(self._Xs, XsForOmega, alpha, dt);
        # Add the constant velocity so that point 0 is the same
        Xnp1+= -Xnp1[0,:]+self._X[0:3]+dt*velocity[0:3]
        self._X = np.reshape(Xnp1,3*N);
        self._Xs = np.reshape(Xsp1,3*N);
    
    def writeLocs(self,of):
        """
        Write the locations to a file object of. 
        """
        N = self._fibDisc.getN(); # to save writing
        for i in range(N):
            for j in range(3):
                of.write('%14.15f     ' % self._X[3*i+j]);
            of.write('\n');

        
    
