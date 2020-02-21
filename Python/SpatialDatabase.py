import numpy as np
from scipy.spatial import cKDTree

# Donev: This looks good to me with some minor points below
class SpatialDatabase(object):

    """
    Donev changed this to say what I think it is:
    Spatial database object. 
    This class is a spatial index of a collection of points in a (periodic sheared) Domain (see Domain.py)
    Its purpose is to search for neighboring points efficiently
    """
    
    def __init__(self,pts,Dom):
        """
        Constructor. Inputs are the set of points and the Domain 
        which are used to initialize the ptsprime variable
        Donev: Essentially all arguments here take a Domain as an argument
        Would it make sense to store here a pointer to Domain?
        Way to answer this is to think whether it would make sense to pass in Domain argument to one of the routines
        that is *different* from the one you used inside updateSpatialStructures?
        """
        self._Npts,_ = pts.shape;
        self.updateSpatialStructures(pts,Dom);

    def updateSpatialStructures(self,pts,Dom):
        """
        Donev: Explain what pts is: An array of unprimed coordinates of the points?
        Update all relevant spatial structures. In the general case, 
        this is just updating the deformed coordinates of the points.
        """
        self._ptsprime = Dom.primecoords(pts);
        
    def selfNeighborList(self,rcut,Dom):
        """
        Compute a list of neighbors. Compute pairs
        of pts (i,j) that are a distance r w/safety factor apart in 
        the primed (deformed) norm.
        General method is a quadratic loop
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        # Quadratic loop
        neighbors = [];
        for iPt in range(self._Npts):
            for jPt in range(iPt+1,self._Npts):
                rvecprime = self._ptsprime[iPt,:]-self._ptsprime[jPt,:];
                # Shift rvec so it's on [-L/2, L/2]^3
                rvecprime = Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                if (np.linalg.norm(rvecprime) < rwsafety):
                    neighbors.append([iPt,jPt]);
        return np.array(neighbors);
            
    def otherNeighborsList(self,other,rcut,Dom):
        """
        Compute a list of neighbors between 2 different point sets.
        Pairs of pts (i,j) that are a distance r w/safety factor apart in
        the primed norm, where i is in self._pts and j is in other._pts
        General method is a quadratic loop
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        # Quadratic loop
        neighbors = [];
        for iPt in range(self._Npts):
            iNeighbors=[];
            for jPt in range(other._Npts):
                rvecprime = self._ptsprime[iPt,:]-other._ptsprime[jPt,:];
                # Shift rvec so it's on [-L/2, L/2]^3
                rvecprime = Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                if (np.linalg.norm(rvecprime) < rwsafety):
                    iNeighbors.append(jPt);
            neighbors.append(iNeighbors);
        return neighbors;

    def __del__(self):
        # Destructor
        print('SpatialDatabase object destroyed');
     

class ckDSpatial(SpatialDatabase):

    """
    Child of SpatialDatabase that uses ckD trees to compute 
    neighbors efficiently.
    """
    
    def __init__(self,pts,Dom):
        """
        Constructor. Initialize the kD tree.
        """
        self.updateSpatialStructures(pts,Dom);
        # Donev: It is typical for child classes in init to call the initializer of the parent class
        # Is this automatic in python? That is, has the line
        # self._Npts,_ = pts.shape;
        # been executed also somehow?
    
    def updateSpatialStructures(self,pts,Dom):
        """
        Update the kD tree using the set of points pts (an N x 3)
        array, and Domain object Dom.
        """
        ptsprime = Dom.primecoords(pts);
        # Mod the points so they are on the right bounds [0,Lx] x [0,Ly] x [0,Lz]
        # (needed for the call to kD tree)
        ptsprime = Dom.ZeroLShiftInPrimeCoords(ptsprime);
        # The domain can be periodic or free space. If periodic, pass
        # that information to the kD tree.
        BoxLens = None;
        if (Dom.isPeriodic()):
            BoxLens = Dom.getLens();
        # Update the KD tree
        self._myKDTree = cKDTree(ptsprime,boxsize=BoxLens);

    def selfNeighborList(self,rcut,Dom):
        """
        Get the neighbors within an Eulerian distance rcut (same 
        as rcut*safety factor in the deformed norm) within the
        ckD tree pointTree. 
        Inputs: distance rcut, domain object Dom where this is happening
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        neighbors = self._myKDTree.query_pairs(rwsafety,output_type='ndarray');
        return neighbors;

    def otherNeighborsList(self,other,rcut,Dom):
        """
        Return a list of neighbors on another ckDtree that are
        within a distance rcut from the points on self._mkDTree.
        Inputs: other ckDSpatial object with its kD tree, Eulerian 
        distance rcut, Domain object Dom
        Outputs: nPoints list of neighbors for each point
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        return self._myKDTree.query_ball_tree(other._myKDTree,rwsafety);

class LinkedListSpatial(SpatialDatabase):

    """
    Child of SpatialDatabase that uses linked lists to do neighbor 
    searches. 
    """
    
    def __init__(self,pts,Dom):
        super().__init__(pts,Dom);
    
    def selfNeighborList(self,rcut,Dom):
        """
        Get the neighbors within an Eulerian distance rcut using
        the linked list construction. 
        Inputs: rcut, Domain object
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        neighbors = [];
        nbins = self.calcnBins(rwsafety,Dom);
        ptbins = self.binsbyP(nbins,Dom);
        #print('Number of bins (%d, %d, %d)' %(nbins[0], nbins[1], nbins[2]));
        # First and next linked lists
        bfirst, pnext = self.binPoints(nbins,Dom);
        # Loop over points. For each point, check neighboring bins
        # for other points.
        for iPt in range(self._Npts): # loop over points
            tbin=[ptbins[iPt,:]];
            neighBins = LinkedListSpatial.neighborBins(tbin,nbins);
            for iSn in range(len(neighBins)): # loop over neighboring bins
                jPt = bfirst[neighBins[iSn]];
                while (jPt !=-1):
                    rvecprime = self._ptsprime[iPt,:]-self._ptsprime[jPt,:];
                    # Shift rvec so it's on [-L/2, L/2]^3
                    rvecprime = Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                    # Only include points that are close enough (and include
                    # each pair only once)
                    if (jPt > iPt and np.linalg.norm(rvecprime) < rwsafety):
                        neighbors.append([iPt,jPt]);
                    jPt = pnext[jPt];
        neighbors = np.array(neighbors);        
        return neighbors;

    def otherNeighborsList(self,other,rcut,Dom):
        """
        Get the neighbors of self within the object other
        for an Eulerian distance rcut. Using linked lists, 
        we iterate
        Inputs: other set of points, rcut, Domain object
        Output: Npts list of neighbors for each point
        """
        rwsafety=rcut*Dom.safetyfactor(); # add the safety factor
        neighbors = [];
        nbins = self.calcnBins(rwsafety,Dom);
        # Bin the self points
        ptbins = self.binsbyP(nbins,Dom);
        print('Number of bins (%d, %d, %d)' %(nbins[0], nbins[1], nbins[2]));
        # First and next linked lists for the OTHER points
        bfirst, pnext = other.binPoints(nbins,Dom);
        # Loop over points. For each point, check neighboring bins
        # for points in other.
        for iPt in range(self._Npts): # loop over points
            iNeighbors=[];
            tbin=[ptbins[iPt,:]];
            neighBins = LinkedListSpatial.neighborBins(tbin,nbins);
            for iSn in range(len(neighBins)): # loop over neighboring bins
                jPt = bfirst[neighBins[iSn]]; # point in OTHER
                while (jPt !=-1):
                    rvecprime = self._ptsprime[iPt,:]-other._ptsprime[jPt,:];
                    # Shift rvec so it's on [-L/2, L/2]^3
                    rvecprime = Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                    # Only include points that are close enough
                    if (np.linalg.norm(rvecprime) < rwsafety):
                        iNeighbors.append(jPt);
                    jPt = pnext[jPt];
            neighbors.append(iNeighbors);
        return neighbors;

    def calcnBins(self,binlen,Dom):
        """
        Calculate the number of bins in each direction for 
        a given bin edge length rcut.
        """
        nbins = np.int_(Dom.getLens()/binlen);
        return nbins;
    
    def binsbyP(self,nbins,Dom):
        """
        Get the bins for each point. 
        Inputs: nbins (3 array with number of bins in each direction), 
        Domain object that computation is done on.
        Return: the bin number for each point as an array as an N x 3 
        array where the columns are the bin numbers in each direction.
        """
        # Shift all coordinates so they are on [0,L]^3 (assuming this
        # will only be necessary for a periodic domain
        coords = Dom.ZeroLShiftInPrimeCoords(self._ptsprime);
        # Get the bin for each pt
        dperbin = Dom.getLens()/nbins;
        bins = np.int_(coords/dperbin);
        bins = np.mod(bins,nbins); # takes care of any rounding issues
        return bins;    

    def binPoints(self,nbins,Dom):
        """ 
        Bin the points, i.e. create the linked lists first and next.
        Inputs = nbins (3 array with number of bins in each direction), 
        Domain object that computation is done on.
        Returns the 2 linked lists first (for the first point in each bin) and
        next (for the next point in the bin).
        """
        bins = self.binsbyP(nbins,Dom);
        sbins = bins[:,0]+nbins[0]*bins[:,1]+nbins[0]*nbins[1]*bins[:,2];
        # Form the linked lists
        bfirst = -np.ones(nbins[0]*nbins[1]*nbins[2],dtype=np.int);
        pnext = -np.ones(self._Npts,dtype=np.int);
        for iPt in range(self._Npts):
            if (bfirst[sbins[iPt]] == -1):
                bfirst[sbins[iPt]] = iPt;
            else:
                jPt = bfirst[sbins[iPt]];
                while (jPt !=-1):
                    jPtprev = jPt;
                    jPt = pnext[jPt];
                pnext[jPtprev] = iPt;
        return bfirst, pnext;
    
    @staticmethod
    def neighborBins(tbin,nbins):
        """
        Neighbor bins for each bin
        Input: the bin as a 3 array (iBin,jBin,kBin)
        Output: list of at most 27 (possibly less if there are less
        than 3 bins in a direction) neighbor bins as a 27 array. 
        Note: it is ok to have periodic neighbors for a general free
        space domain since those points will just get dropped anyway. 
        """
        neighbors = tbin+np.int_([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],\
                        [1,1,0],[-1,1,0],[0,-1,0],[1,-1,0],[-1,-1,0],[0,0,1],\
                        [1,0,1],[-1,0,1],[0,1,1],[1,1,1],[-1,1,1],[0,-1,1],\
                        [1,-1,1],[-1,-1,1],[0,0,-1],[1,0,-1],[-1,0,-1],\
                        [0,1,-1,],[1,1,-1],[-1,1,-1],[0,-1,-1],[1,-1,-1],\
                        [-1,-1,-1]]);
        neighbors = np.mod(neighbors,nbins);
        _, idx = np.unique(neighbors,axis=0,return_index=True);
        neighbors = neighbors[np.sort(idx)];
        sN = neighbors[:,0]+nbins[0]*neighbors[:,1]+nbins[0]*nbins[1]*neighbors[:,2];
        return sN;
        



    
