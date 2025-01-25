import numpy as np
from scipy.spatial import cKDTree


class SpatialDatabase(object):

    """
    The purpose of this class is to take a set of points and find all pairs of points 
    that are $r_c$ or less apart in the Euclidean metric. 
    
    There are three implementations: a quadratic loop (the base class used for checking), 
    a class which uses the built-in python methods associated with ckDTree (a scipy class
    that implements kD trees in C), and a class that uses linked lists and is built for multi-threaded
    CPUs. This latter class is the one that is the default for all neighbor searches
    (initialized `on these lines <https://github.com/stochasticHydroTools/SlenderBody/blob/c48b3b5df1d75e6847b5714a3a2ab49199bae6d9/Python/fiberCollection.py#L203>`_).
    """
    
    def __init__(self,pts,Dom,nThr=1):
        """
        Constructor. 
        
        Parameters
        ----------
        pts: array
            A set of points to initialize the arrays as an $N \\times 3$ array, 
            where $N$ is the number of points
        Dom: Domain object 
            The domain where we do the search (usually sheared periodic)
        nThr: int
            Number of threads; not applicable in this base class.
        """
        self._Npts,_ = pts.shape;
        self.updateSpatialStructures(pts,Dom);
        

    def updateSpatialStructures(self,pts,Dom):
        """
        Update all relevant spatial structures. In the general case,
        this is just updating the coordinates of the points
        and the pointer to the domain object. In particular, because the neighbor
        searches we do are Euclidean distance in periodic sheared coordinates, we 
        first use the Domain object to obtain the coordinates in deformed coordinates, 
        which we save as a member self._pts. 
        
        Parameters
        ----------
        pts: array
            The points in regular orthogonal Cartesian coordinates where we want neighbors
        Dom: Domain object
            The Domain object that we'll be doing the computation on
        """
        self._pts = pts;
        self._ptsPrime = Dom.primecoords(pts);
        self._Dom = Dom;
        
    def TrimListEuclideanDist(self,rcut,neighbors,rcutLow=-1.0):
        """
        This method will trim the list of neighbors to eliminate any superfluous ones.
        In particular, because the neighbor search happens on the periodic sheared domain,
        it's necessary to include a safety factor $S(g)$ so that we catch all pairs of points
        that are $r_c$ apart in the Euclian metric. However, this can result in extra 
        neighbors that are within $r_c(1+S(g))$ in the sheared coordinates, 
        but not actually within $r_c$ in the Euclidean metric in regular coordinates. 
        The purpose of this method is to eliminate those points. 
        
        Parameters
        ----------
        rcut: double
            The cutoff distance below which we want all neighbors
        neighbors: list
            List of pairs $(i,j)$ of points that are within $r_c(1+S(g))$ of each other when 
            the search is done in sheared coordinates with a safety factor. 
        
        Returns
        -------
        list
            List of pairs $(i,j)$ of points that are actually within $r_c$ of each other in
            the Euclidean metric, with periodicity in the sheared directions.
        """
        return self._Dom.EliminatePairsOfPointsOutsideRange(neighbors,self._pts,rcut,rcutLow);
                
        
    def selfNeighborList(self,rcut,rcutLow=-1.0,numperfiber=1):
        """
        This is the main method that computes the list of pairs of points $(i,j)$ 
        that are within $r_c$ of each other. This search will compute all pairs of 
        points that are within $r_c(1+S(g))$ of each other in the sheared coordinates.
        Then it will call the method TrimListEuclideanDist above to trim the list to 
        pairs of points that are actually $r_c$ or less apart in the Euclidean metric.  
        In this general class, we use a quadratic loop. 
        
        Parameters
        ----------
        rcut: double
            The cutoff distance $r_c$
        numperfiber: int, optional
            The number of points per fiber. Does not do anything in the base class. 
        
        Returns
        --------
        list
            List of pairs $(i,j)$ of points that are within $r_c$ of each other in
            the Euclidean metric, with periodicity in the sheared directions.  
        """
        if (rcutLow>=1):
            print('Probably a bug with arguments to self neighbor list')
            import sys
            sys.exit()
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
        TrimNeighbors = self.TrimListEuclideanDist(rcut,np.array(neighbors),rcutLow) ; 
        return TrimNeighbors;
            
    def otherNeighborsList(self,other,rcut):
        """
        Compute a list of neighbors between 2 different point sets.
        
        Parameters
        ----------
        other: array
            Array of "other" points that we want to search against. Specifically, 
            we want neighbors of the array stored in this class that are in the set other. 
        rcut: double
            The cutoff distance $r_c$
        Returns
        --------
        list
            The list has $N_i$ entries, where $N_i$ is the number of points in
            self._pts (the points stored in this class). The $i$th entry of the list
            is the neighbors (indexed by their location in other) of point $i$.
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
    neighbors efficiently. In this case, this class is essentially a wrapper to 
    the scipy class cKD tree.
    """
    
    def __init__(self,pts,Dom,nThr=1):
        """
        Constructor. 
        
        Parameters
        ----------
        pts: array
            A set of points to initialize the arrays as an $N \\times 3$ array, 
            where $N$ is the number of points
        Dom: Domain object 
            The domain where we do the search (sheared periodic)
        nThr: int
            Number of threads; not applicable in this base class.
        """
        super().__init__(pts,Dom,nThr);
    
    def updateSpatialStructures(self,pts,Dom):
        """
        Update all relevant spatial structures. In this case, we update the
        kD tree with the sheared point positions. 
        
        Parameters
        ----------
        pts: array
            The points in regular orthogonal Cartesian coordinates where we want neighbors
        Dom: Domain object
            The Domain object that we'll be doing the computation on
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

    def selfNeighborList(self,rcut,rcutLow=-1.0,numperfiber=1):
        """
        This is the main method that computes the list of pairs of points $(i,j)$ 
        that are within $r_c$ of each other. This search will compute all pairs of 
        points that are within $r_c(1+S(g))$ of each other in the sheared coordinates.
        Then it will call the method TrimListEuclideanDist above to trim the list to 
        pairs of points that are actually $r_c$ or less apart in the Euclidean metric.  
        
        Parameters
        ----------
        rcut: double
            The cutoff distance $r_c$
        numperfiber: int, optional
            The number of points per fiber. Does not do anything in the kD tree class. 
        
        Returns
        --------
        list
            List of pairs $(i,j)$ of points that are within $r_c$ of each other in
            the Euclidean metric, with periodicity in the sheared directions.  
        """
        if (rcutLow>=1):
            print('Probably a bug with arguments to self neighbor list')
            import sys
            sys.exit()
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        neighbors = self._myKDTree.query_pairs(rwsafety,output_type='ndarray');
        TrimNeighbors = self.TrimListEuclideanDist(rcut,neighbors,rcutLow)
        return neighbors;

    def otherNeighborsList(self,other,rcut):
        """
        Compute a list of neighbors between 2 different point sets.
        
        Parameters
        ----------
        other: array
            Array of "other" points that we want to search against. Specifically, 
            we want neighbors of the array stored in this class that are in the set other. 
        rcut: double
            The cutoff distance $r_c$
        Returns
        --------
        list
            The list has $N_i$ entries, where $N_i$ is the number of points in
            self._pts (the points stored in this class). The $i$th entry of the list
            is the neighbors (indexed by their location in other) of point $i$. 
        """
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        return self._myKDTree.query_ball_tree(other._myKDTree,rwsafety);

class CellLinkedList(SpatialDatabase):

    """
    Child of SpatialDatabase that uses multi-threaded CPU linked lists to compute 
    neighbors efficiently. This class is a wrapper to a CUDA/C++ code that computes the neighbors. 
    The raw code for this class can be found 
    `on github here <https://github.com/stochasticHydroTools/SlenderBody/tree/master/Python/Dependencies/NeighborSearch>`_.
    Importantly, the constructor of this class determines whether you use a GPU for the neighbor search or CPU. 
    Typically, this is such an easy computation that it is not worth using GPU resources, so the default is to 
    NOT use the GPU and use a specified number of threads on the CPU. IMPORTANT: if you want to use the GPU, 
    make sure you compile the NeighborSearch module in GPU mode (see the makefile there for more specifics)!
    """
    
    def __init__(self,pts,Dom,nThr=1,useGPU=False):
        """
        Constructor. 
        
        Parameters
        ----------
        pts: array
            A set of points to initialize the arrays as an $N \\times 3$ array, 
            where $N$ is the number of points
        Dom: Domain object 
            The domain where we do the search (sheared periodic)
        nThr: int, optional
            Number of threads for the multi-threaded-CPU version (defaults to 1).
        useGPU: bool, optional
            Whether to use GPU for the neighbor search (see class docstring). Defaults to false.
        """
        import NeighborSearch
        super().__init__(pts,Dom,nThr);
        # The super constructor will then call THIS child method to
        # updateSpatialStructures
        self._useGPU = useGPU;
        self._Nlist = NeighborSearch.NList(nThr) # WARNING: parallel version returns different order of list every time -> loss of reproducibility
    
    def updateSpatialStructures(self,pts,Dom):
        """
        Update all relevant spatial structures. In this case,
        this is just updating the coordinates of the points
        and the pointer to the domain object. In particular, because the neighbor
        searches we do are Euclidean distance in periodic sheared coordinates, we 
        first use the Domain object to obtain the coordinates in deformed coordinates, 
        which we save. 
        
        Parameters
        ----------
        pts: array
            The points in regular orthogonal Cartesian coordinates where we want neighbors
        Dom: Domain object
            The Domain object that we'll be doing the computation on
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

    def selfNeighborList(self,rcut,rcutLow=-1.0,numperfiber=1):
        """
        This is the main method that computes the list of pairs of points $(i,j)$ 
        that are within $r_c$ of each other. This search will compute all pairs of 
        points that are within $r_c(1+S(g))$ of each other in the sheared coordinates.
        Then it will call the method TrimListEuclideanDist above to trim the list to 
        pairs of points that are actually $r_c$ or less apart in the Euclidean metric.  
        
        Parameters
        ----------
        rcut: double
            The cutoff distance $r_c$
        numperfiber: int, optional
            The number of points per fiber. In this class, if numperfiber > 1, it will 
            ignore neighbors on the same fiber. So, for example, if numperfiber=100 and 
            points 5 and 6 are within $r_c$ of each other, it will not include those
            points in the master list.
        
        Returns
        --------
        list
            List of pairs $(i,j)$ of points that are within $r_c$ of each other in
            the Euclidean metric, with periodicity in the sheared directions.  
        """
        if (rcutLow>=1):
            print('Probably a bug with arguments to self neighbor list')
            import sys
            sys.exit()
        rwsafety=rcut*self._Dom.safetyfactor(); # add the safety factor
        self._Nlist.updateList(pos=self._ptsPrime.copy(), Lx=self._Dom._Lx, Ly=self._Dom._Ly, Lz=self._Dom._Lz,\
                 numberParticles=self._Npts,rcut=rwsafety,useGPU=self._useGPU,NperFiber=numperfiber)
        # Post process list to return 2D array
        neighbors = self._Nlist.pairList;
        AllNeighbors = np.reshape(neighbors,(len(neighbors)//2,2))
        TrimNeighbors = self.TrimListEuclideanDist(rcut,AllNeighbors,rcutLow)
        return TrimNeighbors;

    def otherNeighborsList(self,other,rcut):
        """
        Not implemented for the CPU/GPU version. A workaround is to just combine the lists
        of points into one.
        """
        raise NotImplementedError('Other neighbors not implemented in Rauls linked list implementation')
