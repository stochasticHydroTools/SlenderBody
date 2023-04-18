import numpy as np
from scipy.spatial import cKDTree

# Documentation last updated: 03/12/2021

class SpatialDatabase(object):

    """
    Spatial database object.
    This class is a spatial index of a collection of points in a (sheared) Domain (see Domain.py)
    Its purpose is to search for neighboring points efficiently. 
    Children: CellLinkedList, which is a linked cell list implementation in mixed CPU/GPU format
    See https://github.com/stochasticHydroTools/SlenderBody/blob/master/Python/Dependencies/NeighborSearch/nlist_py.cu

    Other children (not used): cKDSpatial (uses ckD trees to find neighbors), 
    LinkedListSpatial (uses linked list in native python), 
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
        self._pts = pts;
        self._ptsPrime = Dom.primecoords(pts);
        self._Dom = Dom;
        
    def TrimListEuclideanDist(self,rcut,neighbors):
        return self._Dom.EliminatePairsOfPointsOutsideRange(neighbors,self._pts,rcut);
                
        
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
        TrimNeighbors = self.TrimListEuclideanDist(rcut,np.array(neighbors)) ; 
        return TrimNeighbors;
            
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
    
    def updateSpatialStructures(self,pts,Dom):
        """
        Update the kD tree using the set of points pts (an N x 3)
        array, and Domain object Dom.
        """
        ptsprime = Dom.primecoords(pts);
        # Mod the points so they are on the right bounds [0,Lx] x [0,Ly] x [0,Lz]
        # (needed for the call to kD tree)
        ptsprime = Dom.ZeroLShiftInPrimeCoords(ptsprime);
        self._pts = pts;
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
        TrimNeighbors = self.TrimListEuclideanDist(rcut,neighbors)
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

class CellLinkedList(SpatialDatabase):

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
        self._Nlist = NeighborSearch.NList(nThr) # WARNING: parallel version returns different order of list every time -> loss of reproducibility
    
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
        self._pts = pts;

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
        TrimNeighbors = self.TrimListEuclideanDist(rcut,AllNeighbors)
        return TrimNeighbors;

    def otherNeighborsList(self,other,rcut):
        """
        """
        raise NotImplementedError('Other neighbors not implemented in Rauls linked list implementation')
