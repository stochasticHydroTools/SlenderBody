import numpy as np
import EwaldUtils as ewc
from math import sqrt

class Domain(object):
    """
    This class handles the domain and the relevant BCs
    (periodic domain is a child of general domain)
    """
    def __init__(self,Lx,Ly,Lz):
        """
        Constructor. Initialize the domain
        Input variables: Lx, Ly, Lz = periodic length in the x, y, and z directions.
        """
        self._Lx = Lx;
        self._Ly = Ly;
        self._Lz = Lz;
        self._Lens = np.array([self._Lx,self._Ly,self._Lz]);
        self._g = 0;

    def safetyfactor(self):
        """
        Compute the safety factor for measuring distance
        in the primed coordinates. If two points are at most rcut apart
        in real coordinates, then they are at most rcut*safetyfactor(g)
        apart in shear coordinates
        """
        return 1+0.5*self._g*self._g+0.5*sqrt(self._g*self._g*(self._g*self._g+4.0));
    
    def setg(self,ing):
        ing-=round(ing); # Now g is in [-1/2,1/2]
        self._g = ing;
    
    def getg(self):
        return self._g;
    
    def getLens(self):
        return self._Lens;
    
    def getminLen(self):
        return np.amin(self._Lens);
    
    def getVol(self):
        return 1.0*self._Lx*self._Ly*self._Lz;

    def isPeriodic(self):
        return 0;
    
    def calcShifted(self,dvec):
        """
        Method to compute the nearest PERIODIC image
        of a point/vector in UNDEFORMED coordinates.
        Inputs: dvec is a 3 vector in (x,y,z)
        undeformed coordinates
        For a general free space domain, do nothing.
        """
        return dvec;

    def MinPrimeShiftInPrimeCoords(self,rprime):
        """
        Method to shift input DEFORMED coordiantes into
        their minimum periodic image on [-L/2, L/2]. 
        Input: rprime = N x 3 list of N vectors in deformed 
        (x',y',z') coordinates. 
        For a general domain, do nothing.
        """
        return rprime;

    def ZeroLShiftInPrimeCoords(self,rprime):
        """
        Method to shift input DEFORMED coordiantes into
        their minimum periodic image on [0,L].
        Input: rprime = N x 3 list of N vectors in deformed 
        (x',y',z') coordinates. 
        For a general domain, check that the coordinates are in 
        the right boundaries and then stop.
        """
        if (np.any(rprime > self._Lens) or np.any(rprime < 0) > 0):
            raise ValueError('Points have gone out of the NON periodic domain');
        return rprime;

    def primecoords(self,ptsxyz):
        """
        Method to calculate the coordinates of a vector 
        in the shifted coordinate system. Inputs = ptsxyz (pts in
        the x,y,z coordinate system).
        Output is the vector pts', i.e. in the x',y',z' coordinate system
        """
        Linv = np.array([[1,-self._g,0],[0,1,0],[0,0,1]]);
        ptsprime = np.dot(Linv,ptsxyz.T).T;
        return ptsprime;
    
    def unprimecoords(self,ptsprime):
        """
        Method to compute the coordinates of a point in the (x,y,z)
        coordinate system from the coordinates in the (x',y',z') coordinate
        system. Inputs: the prime coordinates.
        Outpts: the coordinates in the (x,y,z) space
        """
        L = np.array([[1,self._g,0],[0,1,0],[0,0,1]]);
        pts = np.dot(L,ptsprime.T).T;
        return pts;

    def __del__(self):
        # Destructor
        print('Domain object destroyed');
        

class PeriodicDomain(Domain):
    
    """
    Child class of Domain that handles periodic BCs
    """
    
    def __init__(self,Lx,Ly,Lz):
        super().__init__(Lx,Ly,Lz);

    def calcShifted(self,dvec):
        """
        Method to compute the nearest periodic image
        of a point. The periodicity is in (x',y',z').
        Inputs: vector (not shifted) dvec.
        Output: vector shifted so it's on [-L/2,L/2]^3 (this is 
        necessary to estimate the minimum norm)
        """
        # Just call the C++ function (the python was unbelievably slow!)
        newvec = ewc.calcShifted(dvec, self._g, self._Lx, self._Ly, self._Lz);
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

    def MinPrimeShiftInPrimeCoords(self,rprime):
        """
        Shift input to [-L/2, L/2] (see super docstring)
        """
        rprime-= np.round(rprime/self._Lens)*self._Lens;
        return rprime;

    def ZeroLShiftInPrimeCoords(self,rprime):
        """
        Shift input to [0,L] (see super docstring)
        """
        return np.mod(rprime,self._Lens);

    def isPeriodic(self):
        return 1;

