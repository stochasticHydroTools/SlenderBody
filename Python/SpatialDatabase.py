import numpy as np
from scipy.spatial import cKDTree

# Documentation last updated: 03/12/2021

class SpatialDatabase(object):

    """
    Spatial database object.
    This class is a spatial index of a collection of points in a (sheared) Domain (see Domain.py)
    Its purpose is to search for neighboring points efficiently. 
    Children: cKDSpatial (uses ckD trees to find neighbors), LinkedListSpatial (uses linked list which
    rn are written in native python). 
    """
    
    def __init__(self,pts,Dom,nThr=1):
        """
        Constructor. Inputs are the set of points and the Domain 
        which are used to initialize the ptsprime variable
        """
        self._Npts,_ = pts.shape;
        self.updateSpatialStructures(pts,Dom);

    def updateSpatialStructures(self,pts,Dom):
        """
        Update all relevant spatial structures. In the general case,
        this is just updating the deformed coordinates of the points
        and the pointer to the domain object.
        Inputs: pts is an N x 3 array of pints in unprimed coordinates, 
        Dom is the Domain object that we'll be doing the computation on
        """
        self._ptsPrime = Dom.primecoords(pts);
        self._Dom = Dom;
        
    def selfNeighborList(self,rcut,numperfiber=1):
        """
        Compute a list of neighbors. Compute pairs
        of pts (i,j) that are a distance r w/safety factor apart in 
        the primed (deformed) norm.
        General method is a quadratic loop
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        # Quadratic loop
        neighbors = [];
        for iPt in range(self._Npts):
            for jPt in range(iPt+1,self._Npts):
                rvecprime = self._ptsPrime[iPt,:]-self._ptsPrime[jPt,:];
                # Shift rvec so it's on [-L/2, L/2]^3
                rvecprime = self._Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                if (np.linalg.norm(rvecprime) < rwsafety):
                    neighbors.append([iPt,jPt]);
        return np.array(neighbors);
            
    def otherNeighborsList(self,other,rcut):
        """
        Compute a list of neighbors between 2 different point sets.
        Pairs of pts (i,j) that are a distance r w/safety factor apart in
        the primed norm, where i is in self._pts and j is in other._pts
        General method is a quadratic loop
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        # Quadratic loop
        neighbors = [];
        for iPt in range(self._Npts):
            iNeighbors=[];
            for jPt in range(other._Npts):
                rvecprime = self._ptsPrime[iPt,:]-other._ptsPrime[jPt,:];
                # Shift rvec so it's on [-L/2, L/2]^3
                rvecprime = self._Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                if (np.linalg.norm(rvecprime) < rwsafety):
                    iNeighbors.append(jPt);
            neighbors.append(iNeighbors);
        return neighbors;    

class ckDSpatial(SpatialDatabase):

    """
    Child of SpatialDatabase that uses ckD trees to compute 
    neighbors efficiently.
    """
    
    def __init__(self,pts,Dom,nThr=1):
        """
        Constructor. Initialize the kD tree.
        """
        super().__init__(pts,Dom,nThr);
        # The super constructor will then call THIS child method to
        # updateSpatialStructures
    
    def updateSpatialStructures(self,pts,Dom):
        """
        Update the kD tree using the set of points pts (an N x 3)
        array, and Domain object Dom.
        """
        ptsprime = Dom.primecoords(pts);
        # Mod the points so they are on the right bounds [0,Lx] x [0,Ly] x [0,Lz]
        # (needed for the call to kD tree)
        ptsprime = Dom.ZeroLShiftInPrimeCoords(ptsprime);
        self._ptsPrime = ptsprime;
        # The domain can be periodic or free space. If periodic, pass
        # that information to the kD tree.
        # Update the KD tree
        self._Dom = Dom;
        self._myKDTree = cKDTree(ptsprime,boxsize=Dom.getPeriodicLens());

    def selfNeighborList(self,rcut,numperfiber=1):
        """
        Get the neighbors within an Eulerian distance rcut (same 
        as rcut*safety factor in the deformed norm) within the
        ckD tree pointTree. 
        Inputs: distance rcut
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        neighbors = self._myKDTree.query_pairs(rwsafety,output_type='ndarray');
        return neighbors;

    def otherNeighborsList(self,other,rcut):
        """
        Return a list of neighbors on another ckDtree that are
        within a distance rcut from the points on self._mkDTree.
        Inputs: other ckDSpatial object with its kD tree, Eulerian 
        distance rcut
        Outputs: nPoints list of neighbors for each point
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        return self._myKDTree.query_ball_tree(other._myKDTree,rwsafety);

class RaulLinkedList(SpatialDatabase):

    """
    Multi-threaded CPU version
    """
    
    def __init__(self,pts,Dom,nThr=1):
        """
        Constructor. Initialize
        """
        import NeighborSearch
        super().__init__(pts,Dom,nThr);
        # The super constructor will then call THIS child method to
        # updateSpatialStructures
        self._precision = np.float32;
        self._Nlist = NeighborSearch.NList(nThr)
    
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
        # Update ptsprime
        self._Dom = Dom;
        self._ptsPrime = ptsprime;

    def selfNeighborList(self,rcut,numperfiber=1):
        """
        Get the neighbors within an Eulerian distance rcut (same 
        as rcut*safety factor in the deformed norm) within the
        ckD tree pointTree. 
        Inputs: distance rcut
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        self._Nlist.updateList(pos=self._ptsPrime.copy(), Lx=self._Dom._Lx, Ly=self._Dom._Ly, Lz=self._Dom._Lz,\
                 numberParticles=self._Npts,rcut=rwsafety,useGPU=False,NperFiber=numperfiber)
        # Post process list to return 2D array
        neighbors = self._Nlist.pairList;
        AllNeighbors = np.reshape(neighbors,(len(neighbors)//2,2))
        return AllNeighbors;

    def otherNeighborsList(self,other,rcut):
        """
        """
        raise NotImplementedError('Other neighbors not implemented in Rauls linked list implementation')

class LinkedListSpatial(SpatialDatabase):

    """
    Child of SpatialDatabase that uses linked lists to do neighbor 
    searches. 
    """
    
    def __init__(self,pts,Dom,nThr):
        super().__init__(pts,Dom,nThr);
    
    def selfNeighborList(self,rcut,numperfiber=1):
        """
        Get the neighbors within an Eulerian distance rcut using
        the linked list construction. 
        Inputs: rcut
        Output: pairs of neighbors as an nPairs x 2 array
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        neighbors = [];
        nbins = self.calcnBins(rwsafety);
        ptbins = self.binsbyP(nbins);
        #print('Number of bins (%d, %d, %d)' %(nbins[0], nbins[1], nbins[2]));
        # First and next linked lists
        bfirst, pnext = self.binPoints(nbins);
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
                    rvecprime = self._Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                    # Only include points that are close enough (and include
                    # each pair only once)
                    if (jPt > iPt and np.linalg.norm(rvecprime) < rwsafety):
                        neighbors.append([iPt,jPt]);
                    jPt = pnext[jPt];
        neighbors = np.array(neighbors);        
        return neighbors;

    def otherNeighborsList(self,other,rcut):
        """
        Get the neighbors of self within the object other
        for an Eulerian distance rcut. Using linked lists, 
        we iterate
        Inputs: other set of points, rcut
        Output: Npts list of neighbors for each point
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        neighbors = [];
        nbins = self.calcnBins(rwsafety);
        # Bin the self points
        ptbins = self.binsbyP(nbins);
        print('Number of bins (%d, %d, %d)' %(nbins[0], nbins[1], nbins[2]));
        # First and next linked lists for the OTHER points
        bfirst, pnext = other.binPoints(nbins);
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
                    rvecprime = self._Dom.MinPrimeShiftInPrimeCoords(rvecprime);
                    # Only include points that are close enough
                    if (np.linalg.norm(rvecprime) < rwsafety):
                        iNeighbors.append(jPt);
                    jPt = pnext[jPt];
            neighbors.append(iNeighbors);
        return neighbors;

    def calcnBins(self,binlen):
        """
        Calculate the number of bins in each direction for 
        a given bin edge length rcut.
        """
        nbins = np.int_(self._Dom.getLens()/binlen);
        return nbins;
    
    def binsbyP(self,nbins):
        """
        Get the bins for each point. 
        Inputs: nbins (3 array with number of bins in each direction)
        Return: the bin number for each point as an array as an N x 3 
        array where the columns are the bin numbers in each direction.
        """
        # Shift all coordinates so they are on [0,L]^3 (assuming this
        # will only be necessary for a periodic domain
        coords = self._Dom.ZeroLShiftInPrimeCoords(self._ptsprime);
        # Get the bin for each pt
        dperbin = self._Dom.getLens()/nbins;
        bins = np.int_(coords/dperbin);
        bins = np.mod(bins,nbins); # takes care of any rounding issues
        return bins;    

    def binPoints(self,nbins):
        """ 
        Bin the points, i.e. create the linked lists first and next.
        Inputs = nbins (3 array with number of bins in each direction), 
        Returns the 2 linked lists first (for the first point in each bin) and
        next (for the next point in the bin).
        """
        bins = self.binsbyP(nbins);
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
        



    
