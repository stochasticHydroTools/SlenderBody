import numpy as np
from scipy.spatial import cKDTree
import EwaldUtils as ewc
from math import sqrt

class Domain(object):
    """
    This class handles the domain and the periodic BCs
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
    
    def makekDTree(self,points):
        """
        Make a kD tree. 
        Inputs: list of points to make the tree on as an N x 3 array. 
        Return: the tree
        """
        ptsprime = self.primecoords(points);
        # Mod the points so they are on the right bounds [0,Lx] x [0,Ly] x [0,Lz]
        ptsprime = np.mod(ptsprime,self._Lens);
        # Establish KD tree
        return cKDTree(ptsprime,boxsize=self._Lens);
    
    def kDTreeNeighbors(self,pointTree,rcut):
        """
        Get the neighbors within an Eulerian distance rcut within the
        ckD tree pointTree. 
        Inputs: cKdTree pointTree, distance rcut
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*self.safetyfactor(); # add the safety factor
        return pointTree.query_pairs(rwsafety,output_type='ndarray');
    
    def TwokDTreesNeighbors(self,pointTree,NeighborTree,rcut):
        """
        Return a list of neighbors on the ckDtree NeighborTree that are 
        within a distance rcut from the points on ckDTree PointTree.
        Inputs: cKD tree of the points pointTree, neighbors NeighborTree, 
        distance rcut
        Outputs: nPoints list of neighbors for each point
        """
        rwsafety=rcut*self.safetyfactor(); # add the safety factor
        return pointTree.query_ball_tree(NeighborTree,rwsafety);
    
    def calcShifted(self,dvec):
        """
        Method to compute the nearest periodic image
        of a point. The periodicity is in (x',y',z') 
        where the strain in the coordinate system is given
        by g. 
        Inputs: vector (not shifted), and g
        Output: vector shifted so it's on [-L/2,L/2]^3 (this is 
        necessary to estimate the minimum norm)
        """
        # Just call the C++ function (the python was unbelievably slow!)
        newvec = ewc.calcShifted(dvec, self._g, self._Lx, self._Ly, self._Lz);
        #shift = 0*dvec;
        ## Shift in oblique y and z
        #shift[1] = round(dvec[1]/(1.0*self._Ly));
        #shift[2] = round(dvec[2]/(1.0*self._Lz));
        # Shift the vector
        #dvec[0]-=g*self._Ly*shift[1];
        #dvec[1]-=self._Ly*shift[1];
        #dvec[2]-=self._Lz*shift[2];
        # Shift in x
        #shift[0] = round(dvec[0]/self._Lx);
        #dvec[0]-=shift[0]*self._Lx;
        #print dvec - cppres
        return np.array(newvec);

    def primecoords(self,ptsxyz):
        """
        Method to calculate the coordinates of a vector 
        in the shifted coordinate system. Inputs = ptsxyz (pts in
        the x,y,z coordinate system), and the strain g. 
        Output is the vector pts', i.e. in the x',y',z' coordinate system
        """
        L = np.array([[1,self._g,0],[0,1,0],[0,0,1]]);
        pts = np.linalg.solve(L,ptsxyz.T).T;
        return pts;
    
    def unprimecoords(self,ptsprime):
        """
        Method to compute the coordinates of a point in the (x,y,z)
        coordinate system from the coordinates in the (x',y',z') coordinate
        system. Inputs: the prime coordinates and the strain g. 
        Outpts: the coordinates in the (x,y,z) space
        """
        L = np.array([[1,self._g,0],[0,1,0],[0,0,1]]);
        pts = np.dot(L,ptsprime.T).T;
        return pts;

    #
    # Stuff for binning the points (not used in the final version of the
    # code, we are using kD trees instead)
    #
    def calcnBins(self,rcut):
        """
        Calculate the number of bins in each direction for 
        a given bin edge length rcut.
        """
        rwsafety=rcut*self.safetyfactor();
        nx = int(float(self._Lx)/rwsafety);
        ny = int(float(self._Ly)/rwsafety);
        nz = int(float(self._Lz)/rwsafety);
        return nx, ny, nz;
    
    def binPoints(self,Npts,pts,nxBin,nyBin,nzBin):
        """ 
        Bin the points.
        Inputs = number of pts, locations of pts (in (x,y,z) space), 
        nxBin, nyBin, nzBin = number of bins in each direction.
        Returns the 2 linked lists first (for the first point in each bin) and
        next (for the next point in the bin).
        """
        bins = self.binsbyP(pts,nxBin,nyBin,nzBin);
        sbins = bins[:,0]+nxBin*bins[:,1]+nxBin*nyBin*bins[:,2];
        # Form the linked lists
        bfirst = -np.ones(nxBin*nyBin*nzBin,dtype=np.int);
        pnext = -np.ones(Npts,dtype=np.int);
        for iPt in range(Npts):
            if (bfirst[sbins[iPt]] == -1):
                bfirst[sbins[iPt]] = iPt;
            else:
                jPt = bfirst[sbins[iPt]];
                while (jPt !=-1):
                    jPtprev = jPt;
                    jPt = pnext[jPt];
                pnext[jPtprev] = iPt;
        return bfirst, pnext;
    
    def binsbyP(self,pts,nxBin,nyBin,nzBin):
        """
        Get the bins for each point. 
        Inputs: pts = Npts x 3 array of the pts in (x,y,z) space,
        nxBin, nyBin, nzBin = number of bins in each direction. 
        Return: the bin number for each point as an array
        """
        coords = self.primecoords(pts);
        # Shift all coordinates so they are on [0,L]^3
        coords = coords - np.floor(coords/self._Lens)*self._Lens;
        # Get the bin for each pt
        dperbin = np.array([[float(nxBin)/self._Lx, float(nyBin)/self._Ly,\
                                float(nzBin)/self._Lz]]);
        bins = np.int_(coords*dperbin);
        bins = np.mod(bins,[nxBin, nyBin, nzBin]); # takes care of any rounding issues
        return bins;
    
    @staticmethod
    def neighborBins(tbin,nxBin,nyBin,nzBin):
        """
        Neighbor bins for each bin
        Input: the bin as a 3 array (iBin,jBin,kBin)
        Output: list of at most 27 (possibly less if there are less
        than 3 bins in a direction) neighbor bins as a 27 array. 
        """
        neighbors = tbin+np.int_([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],\
                        [1,1,0],[-1,1,0],[0,-1,0],[1,-1,0],[-1,-1,0],[0,0,1],\
                        [1,0,1],[-1,0,1],[0,1,1],[1,1,1],[-1,1,1],[0,-1,1],\
                        [1,-1,1],[-1,-1,1],[0,0,-1],[1,0,-1],[-1,0,-1],\
                        [0,1,-1,],[1,1,-1],[-1,1,-1],[0,-1,-1],[1,-1,-1],\
                        [-1,-1,-1]]);
        neighbors = np.mod(neighbors,[nxBin, nyBin, nzBin]);
        _, idx = np.unique(neighbors,axis=0,return_index=True);
        neighbors = neighbors[np.sort(idx)];
        sN = neighbors[:,0]+nxBin*neighbors[:,1]+nxBin*nyBin*neighbors[:,2];
        return sN;

        


