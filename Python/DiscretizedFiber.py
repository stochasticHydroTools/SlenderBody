import numpy as np

class DiscretizedFiber(object):
    """

    Object that stores the points and forces for each fiber. 
    That is $X$, $\\tau$, and $X_{mp}$ are the only things stored here, 
    as well as the discretization of the fiber. 

    """

    ## ===============================================
    ##          METHODS FOR INITIALIZATION
    ## ===============================================
    def __init__(self,Discretization,Xs=None,XMP=None):
        """
        Constructor. 

        Parameters
        -----------
        Discretization: FibCollocationDiscretization object
           Stores all the Chebyshev information, 
        Xs: array, optional
           $N \\times 3$ array of tangent vectors. Only set
           if you want to instantiate a specific fiber
        XMP: array
           $3 \\times 1$ array with the fiber midpoint. Only
           set if you want to instantiate a specific fiber.
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
        Initialize a straight fiber with constant tangent 
        vector. This method uses the method of Archimedes to
        sample uniformly on the unit sphere. The steps are

        1) Set $u=1-2r_1$, where $r_1 \\sim U(0,1)$
        2) Set $v=\\sqrt{1-u^2}$
        3) Set $w=2 \\pi r_2$, where $r_2 \\sim U(0,1)$
        4) Set $\\tau=\\left(v\\cos{w}, v \\sin{w}, u\\right)$, which is the 
           tangent vector we use for the straight fiber.

        In addition, we also compute a random midpoint for 
        the fiber, setting $X_{mp}=\\left(r_x L_x, r_y L_y, r_z L_z\\right)$, 
        where $r_x$, $r_y$, and $r_z$ are all random draws from
        $U(0,1)$. 

        Parameters
        ----------
        Lengths: 3-array
           The length of the periodic domain in each direction, $\\left(L_x, L_y, L_z\\right)$
        Xs: array, optional
           $N \\times 3$ array of tangent vectors. Only set
           if you want to instantiate a specific fiber
        XMP: array
           $3 \\times 1$ array with the fiber midpoint. Only
           set if you want to instantiate a specific fiber.
        
        """
        if (Xs is not None):
            self._Xs = np.reshape(Xs,(3*self._fibDisc._Ntau,1));
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
        
        Parameters
        ----------
        Xs: array, optional
           $N \\times 3$ array of tangent vectors. Only set
           if you want to instantiate a specific fiber
        XMP: array
           $3 \\times 1$ array with the fiber midpoint. Only
           set if you want to instantiate a specific fiber.
        """
        self._Xs = np.reshape(Xs,3*self._fibDisc._Ntau);
        self._XMP = XMP;
        self._X = self._fibDisc.calcXFromXsAndMP(Xs,XMP);       
    
    def getPositions(self):
        """
        Get the position and tangent vector as reshaped N x 3 arrays. 

        Returns
        --------
        (array, array,array)
           Three arrays: the first is the $N_x \\times 3$ array of the 
           fiber positions. The second is the $N \\times 3$ array of the
           fiber tangent vectors. The last is the $3 \\times 1$ array of the 
           fiber midpoint.
        """
        return np.reshape(self._X,(self._fibDisc._Nx,3)), np.reshape(self._Xs,(self._fibDisc._Ntau,3)), self._XMP;
    
