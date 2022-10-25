import numpy as np

# Documentation last updared: 03/12/2021

class DiscretizedFiber(object):
    """
    Object that stores the points and forces for each fiber. 
    That is X and Xs are the only things stored here, as well
    as the discretization of the fiber. 
    """
    ## ===============================================
    ##          METHODS FOR INITIALIZATION
    ## ===============================================
    def __init__(self,Discretization,Xs=None,XMP=None):
        """
        Constructor. Object variables are:
        Discretization = FiberDiscretization object that has all the 
        Chebyshev information, X, Xs = (N x 3) arrays that contain 
        the initial X and Xs (otherwise they are initialized randomly
        as a straight fiber).
        """
        self._fibDisc = Discretization;
        self._Xs = Xs;
        self._XMP = XMP;
        try:
            self._X = self._fibDisc.calcXFromXsAndMP(Xs,XMP);
        except:
            self._X = None;

    ## ===============================================
    ##              PUBLIC METHODS
    ## ===============================================
    def initFib(self,Lengths,Xs=None,XMP=None):
        """
        Initialize straight tangent 
        vectors on [0,Lx] x [0,Ly] x [0,Lz]. 
        Input: Lengths = 3 vector [Lx,Ly,Lz]
        """
        if (Xs is not None):
            raise ValueError('If you pass X, need to pass Xs')
            self._Xs = np.reshape(Xs,(3*N,1));
        else:
            # Initialize straight fiber
            # Tangent vector (constant along fiber initially)
            u=1-2*np.random.rand();
            v=np.sqrt(1-u*u);
            w=2*np.pi*np.random.rand();
            tangent = np.array([v*np.cos(w), v*np.sin(w), u]);
            # Initial Xs and X
            self._Xs = np.tile(tangent,self._fibDisc._Ntau);
        if (XMP is not None):
            self._XMP = XMP;
        else:
            self._XMP = Lengths*np.random.rand(3);
        self._X = self._fibDisc.calcXFromXsAndMP(self._Xs,self._XMP);
   
    def passXsandX(self,Xs,XMP):
        """
        Update X and Xs just by passing in the configurations 
        """
        self._Xs = np.reshape(Xs,3*self._fibDisc._Ntau);
        self._XMP = XMP;
        self._X = self._fibDisc.calcXFromXsAndMP(Xs,XMP);       
    
