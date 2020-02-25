import numpy as np
import EwaldUtils as ewc
from math import sqrt

class Domain(object):
    """
    This class handles the domain and the relevant BCs
    (periodic domain is a child of general domain)
    There is a domain D that is just a regular rectangular box 
    of size Lx * Ly * Lz. Then, there is some mapping from this to R^3. 
    This can involve periodic replication in some directions and shearing/stretching
    Donev: Is this true for a general domain -- where is it used? The domain is assumed to be a potentially deformed (non-orthogonal) parallelopiped
    Unprimed coordinates denote positions in the undeformed (orthogonal) domain and 
    primed in the deformed domain
    Donev: This needs to go inside the child class: In this implementation the domain deformation is limited to be simple shear
    Putting shear here in the parent class completely destroys the abstraction of a domain as a general mapping from a rectangle
    to a subdomain of R^3 (it is not all of R^3 btw but could be if all periodic).
    """
    def __init__(self,Lx,Ly,Lz):
        """
        Constructor. Initialize the domain
        Input variables: Lx, Ly, Lz = length in the x, y, and z directions (this is NOT
        necessarily periodic length). But we need the domain to have some volume 
        for stress calculations.
        """
        self._Lx = Lx;
        self._Ly = Ly;
        self._Lz = Lz;
        self._Lens = np.array([self._Lx,self._Ly,self._Lz]);
        self._g = 0; # Donev: Deformation factor due to the shear

    def safetyfactor(self):
        """
        Compute the safety factor for measuring distance
        in the primed coordinates. If two points are at most rcut apart
        in real coordinates, then they are at most rcut*safetyfactor(g)
        apart in shear coordinates
        """
        # Donev: Why don't you declare a local variable g=self._g to save typing?
        # This is why I hate self!!!
        g = self._g;
        return 1+0.5*g*g+0.5*sqrt(g*g*(g*g+4.0));
    
    def setg(self,ing):
        ing-=round(ing); # Now g is in [-1/2,1/2]
        self._g = ing;
    
    def getg(self):
        return self._g;
    
    def getLens(self):
        # Get lengths of the bounded domain
        return self._Lens;
    
    def getVol(self):
        return 1.0*self._Lx*self._Ly*self._Lz;

    def getPeriodicLens(self):
        # Get periodic lengths (none for general domain)
        return [None,None,None];
    
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
    
    def primeWaveNumbersFromUnprimed(self,kxUP,kyUP,kzUP):
        """
        Compute the wave numbers on a sheared grid from the 
        wave numbers on an unsheared grid. This applies in arbitrary
        periodic directions. (This is here so Ewald doesn't know about
        the sheared domain). 
        Inputs: kxUP, kyUP, kzUP = wave numbers on the standard unprimed
        coordinate system (x,y,z). These could be arrays of any form
        Outputs: kxPrime, kyPrime, kzPrime = wabe numbers on the primed
        deformed coordindate system (x',y',z'). Format of output and input 
        are the same.
        """
        kxPrime = kxUP;
        kyPrime = kyUP - self._g*kxUP;
        kzPrime = kzUP;
        return kxPrime, kyPrime, kzPrime;

    def __del__(self):
        # Destructor
        print('Domain object destroyed');
        
class PeriodicDomain(Domain):
    
    """
    Child class of Domain that handles periodic BCs
    Donev: Move shear here not in Domain.
    """
    
    def __init__(self,Lx,Ly,Lz):
        super().__init__(Lx,Ly,Lz);

    def calcShifted(self,dvec):
        """
        Method to compute the nearest periodic image
        of a point. The periodicity is in (x',y',z').
        Inputs: vector (not shifted) dvec in UNPRIMED COORDS.
        Output: UNPRIMED vector shifted periodicially in the primed directions
        so it's on [-L/2,L/2]^3 in undeformed space (this is
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

    def getPeriodicLens(self):
        return self._Lens;

