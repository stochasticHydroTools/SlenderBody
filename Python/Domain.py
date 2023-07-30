import numpy as np
from DomainC import DomainC as DomCpp
from math import sqrt

# Documentation last updated: 03/12/2021

class Domain(object):
    """
    This class handles the domain and the relevant BCs
    (periodic domain is a child of general domain).
    In the abstract, there is a domain $D$ that is just a regular rectangular box 
    of size $L_x \\times L_y \\times L_z$. Then, there is some mapping from $\mathbb{R}^3$
    to this domain. This can involve periodic replication in some directions and shearing/stretching
    """
    ## ===================================
    ##      METHODS FOR INITIALIZATION
    ## ===================================
    def __init__(self,Lx,Ly,Lz):
        """
        Constructor. In the abstract the length inputs are NOT
        necessarily periodic length, but give the domain a characteristic volume 
        we can use for stress calculations. 
        
        Parameters
        -----------
        Lx: double
            The length in the $x$ direction
        Ly: double 
            The length in the $y$ direction
        Lz: double 
            The length in the $z$ direction
        """
        self._Lx = Lx;
        self._Ly = Ly;
        self._Lz = Lz;
        self._Lens = np.array([self._Lx,self._Ly,self._Lz]);
        self._g=0;

    ## ===================================
    ##          PUBLIC METHODS 
    ## ===================================
    def safetyfactor(self):
        return 1.0;
       
    def getLens(self):
        return self._Lens;
    
    def getVol(self):
        return 1.0*self._Lx*self._Ly*self._Lz;

    def getPeriodicLens(self):
        return [None,None,None];
    
    def calcShifted(self,dvec):
        return dvec;

    def MinPrimeShiftInPrimeCoords(self,rprime):
        return rprime;

    def ZeroLShiftInPrimeCoords(self,rprime):
        return rprime;

    def primecoords(self,ptsxyz):
        return ptsxyz;
    
    def unprimecoords(self,ptsprime):
        return ptsprime;
    
    def primeWaveNumbersFromUnprimed(self,kxUP,kyUP,kzUP):
        kxPrime = kxUP;
        kyPrime = kyUP;
        kzPrime = kzUP;
        return kxPrime, kyPrime, kzPrime;
    
    def setg(self,ing):
        if (ing > 0):
            raise ValueError('Cannot set g (strain) >0 in a non periodic domain');
        self._g = 0;
        
    def roundg(self):
        return self._g; # will be zero
    
    def getg(self):
        return self._g;
        
class PeriodicShearedDomain(Domain):
    
    """
    Child class of Domain that handles periodic BCs and shear.
    In this implementation the domain deformation is limited to be simple shear.
    The domain is assumed to be a potentially deformed (non-orthogonal) parallelepiped. 
    Unprimed coordinates denote positions in the undeformed (orthogonal) domain and 
    primed in the deformed domain. 
    
    Letting $g$ denote the strain in the coordinate system, the mapping between 
    coordinates is given by
    $$
    p':=\\begin{pmatrix} x' \\\\ y' \\\\ z' \\end{pmatrix}
    = \\begin{pmatrix} 1 & -g & 0 \\\\ 0 & 1 & 0  \\\\ 0 & 0 & 1 \\end{pmatrix}
    \\begin{pmatrix} x \\\\ y \\\\ z \\end{pmatrix}:=Lp
    $$
    and we will make use of the matrix $L$ extensively in the methods below.  
    
    In this implementation, the strain $g$ (self._g) is a private member of this class.
    
    """
    
    ## ===================================
    ##      METHODS FOR INITIALIZATION
    ## ===================================
    def __init__(self,Lx,Ly,Lz):
        """
        Constructor.
        
        Parameters
        -----------
        Lx: double
            The periodic length in the $x$ direction
        Ly: double 
            The periodic length in the $y$ direction
        Lz: double 
            The periodic length in the $z$ direction
        """
        super().__init__(Lx,Ly,Lz);
        self._cppDom = DomCpp([Lx,Ly,Lz]); # initialize C++ variables
        self._g = 0; # Deformation factor due to the shear

    ## ===================================
    ##          PUBLIC METHODS 
    ## ===================================
    def safetyfactor(self):
        """
        Compute the safety factor for measuring distance
        in the primed coordinates. If two points are at most $r_c$ apart
        in real coordinates, then they are at most $r_c \\times S(g)$
        apart in shear coordinates. For a sheared domain, the safety factor 
        can be found by bounding the distance in Eulerian coordinates by 
        the "distance" in sheared coordinates to obtain
        $$
        ||x|| = \\sqrt{ (x')^T L^{-T} L^{-1} x'}\\geq \\left(1+ \\frac{1}{2} \\left(g^2 + \\sqrt{g^2 \\left(g^2+4\\right)}\\right)\\right)^{-1}:=S^{-1} ||x'||.
        $$
        Thus, points that are $S r_c$ or more apart using the (wrong) Euclidean metric in 
        primed coordinates are at least $r_c$ or more apart in orthogonal coordinates.
        
        Returns
        -------
        double
            The safety factor. 
        """
        g = self._g;
        return 1+0.5*g*g+0.5*sqrt(g*g*(g*g+4.0));
    
    def setg(self,ing):
        self._g = ing;
    
    def roundg(self):
        """
        Shift g so it's on [-1/2,1/2]
        """
        self._g-=round(self._g);
    
    def getg(self):
        return self._g;

    def calcShifted(self,dvec):
        """
        Method to compute the nearest periodic image
        of a point. The periodicity is in the primed directions. What this method
        is doing is to take a vector and slide it along the PERIODIC axes until 
        it is on $[-L/2,L/2]^3$ in undeformed space (this is
        necessary to estimate the minimum norm)
        
        Parameters
        ----------
        dvec: 3-vector
            Vector in unprimed coordinates
        
        Returns
        --------
        vector (3)
            The shifted vector
        """
        # Just call the C++ function (the python was unbelievably slow!)
        newvec = self._cppDom.calcShifted(dvec, self._g);
        if (False):
            shift = 0*dvec;
            ## Shift in oblique y and z
            shift[1] = round(dvec[1]/(1.0*self._Ly));
            shift[2] = round(dvec[2]/(1.0*self._Lz));
            # Shift the vector
            dvec[0]-=g*self._Ly*shift[1];
            dvec[1]-=self._Ly*shift[1];
            dvec[2]-=self._Lz*shift[2];
            # Shift in x
            shift[0] = round(dvec[0]/self._Lx);
            dvec[0]-=shift[0]*self._Lx;
        return np.array(newvec);
        
    def EliminatePairsOfPointsOutsideRange(self,pairs,pts,rcut):
        """
        This method takes a list of pairs of points that are potentially less than 
        rcut apart. It then removes any erroneous pairs that come from using a safety
        factor in the sheared coordinates. 
        
        Parameters
        -----------
        pairs: list
            List of pairs (of integer indices) of potentially interacting points
            that were picked up by neighbor search with the safety factor.
        pts: array 
            Array that has the positions of the points
        rcut: double
            The cutoff distance we want to compute interactions below
        
        Returns
        --------
        list
            List of pairs (of integer indices) of interacting points
            that are indeed a distance less than rcut in Euclidean distance (with 
            periodicity measured using sheared coordinates)
        """
        return self._cppDom.EliminatePairsOfPointsOutsideRange(pairs,pts,rcut,self._g);

    def MinPrimeShiftInPrimeCoords(self,rprime):
        """
        Method to shift input deformed coordinates into
        their minimum periodic image on $[-L/2, L/2]$. 
        
        Parameters
        -----------
        rprime: array
            $N \\times 3$ array of $N$ vectors in deformed $(x',y',z')$ coordinates. 
        
        Returns
        --------
        array
            The DEFORMED coordinates with a periodic shift so that their
            locations are on $[-L/2,L/2]$
        """
        rprime-= np.round(rprime/self._Lens)*self._Lens;
        return rprime;

    def ZeroLShiftInPrimeCoords(self,rprime):
        """
        Method to shift input deformed coordinates into
        their minimum periodic image on $[0,L]$. This is necessary
        if an external method requires them on this box.
        
        Parameters
        -----------
        rprime: array
            $N \\times 3$ array of $N$ vectors in deformed $(x',y',z')$ coordinates. 
        
        Returns
        --------
        array
            The DEFORMED coordinates with a periodic shift so that their
            locations are on $[0,L]$
        """
        shifted = np.mod(rprime,self._Lens);
        shifted[shifted==self._Lens] = 0;
        return shifted;

    def getPeriodicLens(self):
        """
        Returns
        -------
        Array (3)
            The length in each direction
        """
        return self._Lens;

    def primecoords(self,ptsxyz):
        """
        Method to calculate the coordinates of a vector 
        in the shifted coordinate system by applying the matrix
        $L$ (see constructor)
        
        Parameters
        ----------
        ptsxyz: array
            $N \\times 3$ array of points in the $(x,y,z)$ coordinate system
        
        Returns
        --------
        array
            $N \\times 3$ array of points in the $(x',y',z')$ coordinate system
        """
        L = np.array([[1,-self._g,0],[0,1,0],[0,0,1]]);
        ptsprime = np.dot(L,ptsxyz.T).T;
        return ptsprime;
    
    def unprimecoords(self,ptsprime):
        """
        Method to calculate the coordinates of a vector 
        in the shifted coordinate system by applying the matrix
        $L^{-1}$ (see constructor)
        
        Parameters
        ----------
        ptsprime: array
            $N \\times 3$ array of points in the $(x',y',z')$ coordinate system
        
        Returns
        --------
        array
            $N \\times 3$ array of points in the $(x,y,z)$ coordinate system
        """
        Linv = np.array([[1,self._g,0],[0,1,0],[0,0,1]]);
        pts = np.dot(Linv,ptsprime.T).T;
        return pts;
    
    def primeWaveNumbersFromUnprimed(self,kxUP,kyUP,kzUP):
        """
        Compute the wave numbers on a sheared grid from the 
        wave numbers on an unsheared grid. The wave numbers are discussed
        in (86) of `this paper.
        <https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.014102>`_
        
        Parameters
        ----------
        kxUP: array
            $x$ wave numbers on standard unprimed coordinate system $(x,y,z)$. 
        kyUP: array
            $y$ wave numbers on standard unprimed coordinate system $(x,y,z)$. 
        kzUP: array
            $z$ wave numbers on standard unprimed coordinate system $(x,y,z)$. 
            
        Returns
        -------
        (array,array,array)
            Wave numbers on the primed deformed coordindate system $(x',y',z')$. 
            These are the same as the $(x,y,z)$ system, except the $y$ wave number
            becomes $k_y'=k_y - g k_x$
        """
        kxPrime = kxUP;
        kyPrime = kyUP - self._g*kxUP;
        kzPrime = kzUP;
        return kxPrime, kyPrime, kzPrime;

