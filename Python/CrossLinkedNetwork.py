import numpy as np
from CrossLinkForceEvaluator import CrossLinkForceEvaluator as CForceEvalCpp
from warnings import warn 
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import time

verbose = -1;
kbT = 4.1e-3;

class CrossLinkedNetwork(object):

    """
    This class keeps track of cross linking information.
    
    In our model, cross linkers form and break at $N_u$ uniformly-spaced 
    "sites" along the fiber centerline. The positions of these sites are 
    obtained by resampling the Chebyshev interpolant $X(s)$ of the fiber centerline. 
    This abstract class is not aware of the dynamics of link formation and
    breakage at the sites (those are left to the child class DoubleEndedCrossLinkedNetwork).
    This class is, however, aware of the number of links, and contains members which 
    list the site number which is the "head" of each link and the site number which is the 
    "tail." It also stores a list of periodic shifts associated with each link, which 
    can also be thought of as keeping track of what periodic copy of fiber $i$ is linked 
    to what periodic copy of fiber $j$. 
    
    This class also implements methods for computing force and stress,
    and information about the network morphology (number of bundles, etc.)
    As far as computing the force, we model each CL as a linear spring between the two
    sites.
    """
    
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,smoothForce,nThreads=1):
        """
        Initialize the CrossLinkedNetwork. 
        
        Parameters
        -----------
        nFib: int
            Number of fibers
        N: int
            Number of Chebyshev collocation points for fiber POSITION (note that
            this differs from the number of tangent vectors, which we typically
            refer to using $N$).
        Nunisites: int 
            Number of uniformly placed CL binding sites per fiber
        Lfib: double
            Length of each fiber
        kCL: double
            Cross linking spring constant (for the linear spring model)
        rl: double
            Rest length of the CLs (for the linear spring model)
        Dom: Domain object
            Periodic domain on which the calculation is being carried out
        fibDisc: FibCollocationDiscretization object
            The discretization of each fiber's centerline
        smoothForce: boolean
            Whether to smooth out the forcing or use an actual spring between
            the uniformly spaced points. See the method CLForce for details on 
            what formulas this switches between
        nThreads: int
            Number of threads for openMP calculation of CL forces and stress
        """
        self._Npf = N; 
        self._NsitesPerf = Nunisites;
        self._Lfib = Lfib;
        self._nFib = nFib;
        self._kCL = kCL;
        self._rl = rl;
        self._smoothForce = smoothForce;
        
        self._nDoubleBoundLinks = 0;             # number of links taken
        self._TotNumSites = self._NsitesPerf*self._nFib;
        
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length of the fiber.
        self._sigma = self.sigmaFromNL(self._Npf,self._Lfib)
            
        # Cutoff bounds to form the CLs
        try:
            self._deltaL = min(np.sqrt(kbT/self._kCL),0.5*self._rl);
        except: 
            self._deltaL = min(2*np.sqrt(kbT/10),0.5*self._rl);
        # Add fudge factor because of discrete sites
        self._ds = self._Lfib/(self._NsitesPerf-1);
        #if (self._ds > 2*self._deltaL):
        #    raise ValueError('The number of uniform pts (ds = %f) is too small to resolve \
        #        the delta binding distance %f' %(self._ds, self._deltaL));
        
        # Uniform binding locations
        self._su = np.linspace(0.0,self._Lfib,self._NsitesPerf,endpoint=True);
        self._ChebStartByFib=np.arange(0,self._Npf*(self._nFib+1),self._Npf);
        self._allL = np.ones(self._nFib)*self._Lfib;
        self._wCheb = np.tile(fibDisc.getw(),nFib);
        # Seed C++ random number generator and initialize the C++ variables for cross linking 
        self._DLens = Dom.getPeriodicLens();
        for iD in range(len(self._DLens)):
            if (self._DLens[iD] is None):
                self._DLens[iD] = 1e99;
        FibFromSiteIndex = np.repeat(np.arange(nFib,dtype=np.int64),self._NsitesPerf);
        NChebs = self._Npf*np.ones(nFib,dtype=np.int64);
        self._CForceEvaluator = CForceEvalCpp(self._NsitesPerf,np.tile(self._su,nFib),fibDisc._MatfromNtoUniform[0::3,0::3],\
            fibDisc._stackWTilde_Nx,FibFromSiteIndex,NChebs,np.tile(fibDisc.gets(),nFib),\
            self._wCheb,self._sigma*np.ones(nFib),self._kCL,self._rl,nThreads);
    
    def sigmaFromNL(self,N,L):
        """
        In the case when we smooth the fiber cross linking force (see method
        CLForce), the forcing is smoothed over a Gaussian with standard deviation
        $\sigma$. This method gives $\sigma$ as a function of the number of 
        Chebyshev points $N$ and length $L$. 
        
        Parameters
        ----------
        N: int
            Number of Chebyshev collocation points for fiber POSITION (note that
            this differs from the number of tangent vectors, which we typically
            refer to using $N$).
        L: double
            Length of the fibers
            
        Returns
        --------
        double
            The smoothing lengthscale $\sigma$
        """
        sigma = 0.1*L;
        if (N >=50):
            sigma = 0.005*L; # just to check point forces with more rigid fibers
        elif (N >=32):
            sigma = 0.05*L;
        elif (N>=24):
            sigma = 0.075*L
        return sigma;
    
    ## ==================================================================
    ## METHODS FOR NETWORK UPDATE & FORCE (CALLED BY TEMPORAL INTEGRATOR)
    ## ==================================================================
    def updateNetwork(self,fiberCol,Dom,tstep):
        raise NotImplementedError('No update implemented for abstract network')
    
    def CLForce(self,uniPoints,chebPoints,Dom,fibCollection):
        """
        This method computes the total cross linking FORCE (not force density)
        in one of two ways. To establish some notation, let the link be 
        between uniform point $p$ on fiber $X^{(i)}$ and uniform point $q$ on fiber
        $X^{(j)}$. Let the distance between the uniform points be $R$.
        
        If the variable self._smoothForce is set to true, then we first compute a 
        force density
        $$f^{(i)}(s) = -K \\left(1-\\frac{\\ell}{R}\\right) \\delta_h(s-s_i^*)\\int_0^L \\left(X^{(i)}(s)-X^{(j)}(s')\\right) \\delta_h(s'-s_j^*) ds'$$
        on fiber $i$, with the corresponding formula for fiber $j$, where $\\delta_h$ is a Gaussian
        density with standard deviation $\\sigma$. We then obtain force by multiplying by the 
        $L^2$ inner product weights matrix $\\widetilde{W}$. More details on this are in 
        Section 9.1.3.2 in Maxian's PhD thesis.
        
        If, on the other hand, self._smoothForce is set to false, then we compute forces
        at the uniform points using the standard formulas for a spring, then map those forces to the 
        Chebyshev points by multiplying by the corresponding row of the uniform resampling matrix. 
        The exact formulas can be found in Section 9.1.3.1 in Maxian's PhD thesis.
        
        Parameters
        ----------
        uniPoints: array
            Uniform points for all fibers
        chebPoints: array
            Chebyshev points on all fibers
        Dom: Domain object
            The (potentially strained) periodic domain on which we due the calculation
            
        Returns
        -------
        array
            nPts*3 one-dimensional array of the forces at the Chebyshev points due to the CLs
        """
        if (self._nDoubleBoundLinks==0): # if no CLs don't waste time
            return np.zeros(3*self._Npf*self._nFib);
        # Call the C++ function to compute forces
        shifts = Dom.unprimecoords(self._PrimedShifts[:self._nDoubleBoundLinks,:]);
        if (self._smoothForce):
            Clforces = self._CForceEvaluator.calcCLForces(self._HeadsOfLinks[:self._nDoubleBoundLinks], \
                self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPoints,chebPoints);
        else:
            Clforces = self._CForceEvaluator.calcCLForcesEnergy(self._HeadsOfLinks[:self._nDoubleBoundLinks], \
                    self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPoints,chebPoints);
        return Clforces;
        
 
    def CLStress(self,fibCollection,chebPts,Dom):
        """
        Compute the cross-linker contribution to the stress. If a single 
        cross linker connects fibers $i$ and $j$, the formula for the stress
        due to this CL is given by
        $$\\sigma_{CL} = \\sum_p X^{(i)}_p F^{(i)}_p + X^{(j)}_p F^{(j)}_p$$
        where the product between $X$ and $F$ is an outer product and the 
        positions $X$ are the same periodic copies of the fibers that are linked
        by the CL. 
        
        At present, this method is only implemented for nonsmooth forcing (actual springs
        between the fibers). As such, it returns an error if you try to call it with
        smooth forcing.
        
        Parameters
        ----------
        fibCollection: fiberCollection object
            The object that stores the collection of fibers (and is queried to get the 
            uniform points needed to compute forces)
        chebPts: array
            Chebyshev points of all fiber locations
        Dom: Domain object
            The periodic domain; needed to compute volume

        Returns
        -------
        Array
            The stress due to the cross linkers as a $3 \\times 3$ array.
        """
        if (self._nDoubleBoundLinks==0):
            return 0;
        uniPts = fibCollection.getUniformPoints(chebPts);
        shifts = Dom.unprimecoords(np.array(self._PrimedShifts));
        if (self._smoothForce):
            raise TypeError('Stress only implemented with nonsmooth (energy-based) CL forcing')
        stress = self._CForceEvaluator.calcCLStressEnergy(self._HeadsOfLinks[:self._nDoubleBoundLinks],\
            self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPts,chebPts);
        stress/=Dom.getVol();
        return stress;
    
    ## ==============================================
    ##     METHODS FOR NETWORK GEOMETRY INFO
    ## ==============================================   
    def numLinks(self):
        """
        Returns
        -------
        int
            The number of links connecting distinct fibers
        """
        return self._nDoubleBoundLinks;
    
    def mapSiteToFiber(self,sites):
        """
        Parameters
        ----------
        sites: array of ints
            Indices of the uniform sites
        
        Returns
        -------
        int array
            The fiber(s) associated with a set of uniform site(s)
        """
        return sites//self._NsitesPerf;
    
    def mapFibToStartingSite(self,LinkedFibs):
        """
        The first uniform site on a given fiber(s)
        
        Parameters
        -----------
        LinkedFibs: array of ints
            The fibers we are interested in
        
        Returns
        --------
        int array
            The index of the first uniform site associated with 
            the fibers in question
        """
        return LinkedFibs*self._NsitesPerf;
    
    def numLinksOnEachFiber(self):
        """
        Returns
        --------
        array
            One-dimensional array of length nFibers, which gives the number
            of links bound to each fiber. The total number of links in the system
            is thus half the sum of this array
        """
        iFibs = self.mapSiteToFiber(self._HeadsOfLinks[:self._nDoubleBoundLinks])
        jFibs = self.mapSiteToFiber(self._TailsOfLinks[:self._nDoubleBoundLinks])
        NumOnEachFiber = np.zeros(self._nFib,dtype=np.int64)
        for iLink in range(len(iFibs)):
            NumOnEachFiber[iFibs[iLink]]+=1;
            NumOnEachFiber[jFibs[iLink]]+=1;
        return NumOnEachFiber; # Sum is 2 x the number of links
        
    def calcLinkStrains(self,uniPoints,Dom):
        """
        Method to compute the strain in each link. 
        We define (signed) strain as $r-\\ell$, where $r$ is the length of 
        the link at present and $\\ell$ is the rest length. This statistic is 
        useful to tell if the simulation is unstable.
        
        Parameters
        ----------
        uniPoints: array
            Uniform points, as an nUni*nFib x 3 two-dimensional numpy array
        Dom: Domain object 
            Periodic domain
        
        Returns
        --------
        array
            An array of nLinks size that has the signed strain for each link
        """
        linkStrains = np.zeros(self._nDoubleBoundLinks);
        shifts = Dom.unprimecoords(self._PrimedShifts);
        for iLink in range(self._nDoubleBoundLinks):
            ds = uniPoints[self._HeadsOfLinks[iLink],:]-uniPoints[self._TailsOfLinks[iLink],:]-shifts[iLink,:];
            linkStrains[iLink]=(np.linalg.norm(ds)-self._rl);#/self._rl;
        #print(np.amax(np.abs(linkStrains)))
        return linkStrains;
    
    def SpringCOM(self,uniPoints,Dom):
        """
        Method to compute the center of mass of each link.
        
        Parameters
        ----------
        uniPoints: array
            Uniform points, as an nUni*nFib x 3 two-dimensional numpy array
        Dom: Domain object 
            Periodic domain
        
        Returns
        --------
        array
            An array of nLinks x 3 size that has the center of each link (average 
            of the two linked points)
        """
        linkCOMs = np.zeros((self._nDoubleBoundLinks,3));
        shifts = Dom.unprimecoords(self._PrimedShifts);
        for iLink in range(self._nDoubleBoundLinks):
            linkCOMs[iLink,:]=(uniPoints[self._HeadsOfLinks[iLink],:]+(uniPoints[self._TailsOfLinks[iLink],:]+shifts[iLink,:]))/2;
        return linkCOMs;
    
    def getSortedLinks(self):
        """
        Sort the links so that the site with smaller index comes first
        """
        AllLinks = np.zeros((self._nDoubleBoundLinks,2),dtype=np.int64);
        for iLink in range(self._nDoubleBoundLinks):
            minsite = min(self._HeadsOfLinks[iLink], self._TailsOfLinks[iLink])
            maxsite = max(self._HeadsOfLinks[iLink], self._TailsOfLinks[iLink])
            AllLinks[iLink,0] = minsite;
            AllLinks[iLink,1] = maxsite;
        return AllLinks;
    
    def ConnectionMatrix(self,bundleDist=0):
        """
        Build an nFib x nFib matrix that gives the fibers connected to each other.
        
        Parameters
        ----------
        bundleDist: double
            If bundleDist = 0, it will give you the adjacency matrix with nonzero
            entries at (i,j) if there is one link between fibers i and j.
            If bundleDist > 0, the nonzero entries (i,j) have 2 links between them
            separated by a distance (in arclength) at least bundleDist
        
        Returns
        -------
        The fiber adjacency matrix
        """
        SortedLinks = self.getSortedLinks();
        LinkedFibs = self.mapSiteToFiber(SortedLinks)
        LinkedSites = SortedLinks - self.mapFibToStartingSite(LinkedFibs);
        AdjacencyMatrix = csr_matrix((np.ones(len(LinkedFibs[:,0]),dtype=np.int64),\
            (LinkedFibs[:,0],LinkedFibs[:,1])),shape=(self._nFib,self._nFib)); 
        if (bundleDist==0):
            return AdjacencyMatrix;
        (HeadFibs, TailFibs) = AdjacencyMatrix.nonzero();
        BundleConnections = lil_matrix((self._nFib,self._nFib),dtype=np.int64)
        for iConnection in range(len(HeadFibs)):
            iFib = HeadFibs[iConnection];
            jFib = TailFibs[iConnection];
            linkInds = np.where(np.logical_and(LinkedFibs[:,0]==iFib,LinkedFibs[:,1]==jFib));
            s_iSites = LinkedSites[linkInds[0],0]*self._ds;
            s_jSites = LinkedSites[linkInds[0],1]*self._ds;
            if (max(s_iSites)-min(s_iSites) > bundleDist and \
                max(s_jSites)-min(s_jSites) > bundleDist):
                # There are sites connected that are sufficiently well separated
                BundleConnections[iFib,jFib]=1;
        return BundleConnections;
     
    def FindBundles(self,bunddist):
        """
        Uses the adjacency matrix of the previous method to sort the fibers into 
        bundles based on the connectivity.
        
        Parameters
        ----------
        bunddist: double
            Used in the previous method to make the connectivity matrix
        
        Returns
        -------
        (int, int array)
            The method first returns the number of bundles in the system, followed 
            by an integer array of length nFib that records what bundle each fiber is in
        """
        AdjacencyMatrix = self.ConnectionMatrix(bunddist);
        nBundles, whichBundlePerFib = connected_components(csgraph=AdjacencyMatrix, directed=False, return_labels=True)
        return nBundles, whichBundlePerFib;  
    
    def BundleOrderParameters(self,fibCollection,nBundles,whichBundlePerFib,minPerBundle=2):
        """
        Get the order parameter of each bundle. The order parameter for a bundle $B$ of F filaments 
        is defined as the maximum eigenvalue of the matrix 
        $$M = \\frac{1}{F L} \\sum_{i \\in B} \\int_0^L {\\tau^{(i)}(s)\\tau^{(i)}(s) ds}$$
        where we discretize the integral by direct Clenshaw-Curtis quadrature.
        
        Parameters
        ----------
        fibCollecton: fiberCollection object
            The collection of fibers
        nBundles: int
            The number of bundles in the system (obtained from previous method)
        whichBundlePerFib: int-array
            An integer array of length nFib that records what bundle each fiber is in (obtained
            from previous method)
        minPerBundle: int
            The minimum number of fibers that qualify as a "bundle." The default is two, so that 
            we don't compute order parameters for "bundles" of one fiber.
        
        Returns
        --------
        (array, array, array)
            Three arrays are returned: the order parameters (1D array), the number of fibers per bundle
            (1D array), and the average tangent vector of the bundle (NBundle x 3 array). These arrays
            are resized to remove "bundles" that have less than minPerBundle fibers.
        """
        BundleMatrices = np.zeros((3*nBundles,3));
        NPerBundle = np.zeros(nBundles);
        OrderParams = np.zeros(nBundles);
        averageBundleTangents = np.zeros((nBundles,3));
        for iFib in range(self._nFib):
            # Find cluster
            clusterindex = whichBundlePerFib[iFib];
            NPerBundle[clusterindex]+=1;
            Xs = fibCollection._tanvecs;
            Ntau = int(len(Xs)/(self._nFib));
            iStart = Ntau*iFib;
            BundleMatrices[clusterindex*3:clusterindex*3+3,:]+=fibCollection._fiberDisc.averageXsXs(Xs[iStart:iStart+Ntau,:]);
            averageBundleTangents[clusterindex]+=fibCollection._fiberDisc.averageTau(Xs[iStart:iStart+Ntau,:]);
        for clusterindex in range(nBundles):
            B = 1/(NPerBundle[clusterindex])*BundleMatrices[clusterindex*3:clusterindex*3+3,:];
            EigValues = np.linalg.eigvalsh(B);
            OrderParams[clusterindex] = np.amax(np.abs(EigValues))
            averageBundleTangents[clusterindex]*=1/(NPerBundle[clusterindex]);           
        return OrderParams[NPerBundle >= minPerBundle], NPerBundle[NPerBundle >= minPerBundle], averageBundleTangents[NPerBundle >= minPerBundle];
     
    def avgBundleAlignment(self,BundleAlignmentParams,nPerBundle):
        """
        Calculate the weighted average of alignment across all bundles, 
        $$\\bar{q}= \\frac{\\sum q_i n_i}{\\sum n_i}$$
        where $q_i$ is the alignment parameter of bundle $i$ and $n_i$ is
        the number of fibers in bundle $i$.
        
        Parameters
        -----------
        BundleAlignmentParams: array
            The alignment parameters for each bundle, obtained from previous method.
        nPerBundle: int array
            The number of fibers in each bundle, obtained from previous method.
            
        Returns
        -------
        double
            The average bundle alignment parameter $\\bar{q}$ defined above.
        """
        if (np.sum(nPerBundle)==0):
            return 0;
        return np.sum(BundleAlignmentParams*nPerBundle)/np.sum(nPerBundle);
    
    def LocalOrientations(self,nEdges,fibCollection):  
        """
        Calculate the orientiation parameter for every fiber 
        within nEdges of a given fiber. The order parameter
        is defined as the maximum eigenvalue of the matrix 
        $$M^{(i)} = \\frac{1}{F L} \\sum_{j \\in N^{(i)}} \\int_0^L {\\tau^{(j)}(s)\\tau^{(j)}(s) ds}$$
        where we discretize the integral by direct Clenshaw-Curtis quadrature.
        Here the sum occurs over all fibers that are within a certain number of 
        "edges" (links) in the adjacency matrix from fiber $i$. We define this as 
        the set of "neighbors" $N^{(i)}$. 
        
        Parameters
        ------
        nEdges: int
            Number of conections we allow fibers to be separated by. For instance, 
            if nEdges=1, then the sum is over all fibers connected to fiber $i$. If 
            nEdges=2, then the sum is over all fibers connected directly to fiber $i$, 
            or connected to an intermediate fiber that is connected to fiber $i$. And so
            forth. 
        fibCollection: fiberCollection object
            The collecton of fibers we are operating on.
            
        Returns
        --------
        array
            A one-dimensional array of length nFib, where the $i$th entry is the 
            orientation parameter, the maximum eigenvalue of the matrix $M^{(i)}$ 
            defined above.
        """
        AdjacencyMatrix = self.ConnectionMatrix();
        NeighborOrderParams = np.zeros(self._nFib);
        numCloseBy = np.zeros(self._nFib);
        Distances = shortest_path(AdjacencyMatrix, directed=False, unweighted=True, indices=None)
        for iFib in range(self._nFib):
            OtherFibDistances = Distances[iFib,:];
            Fibs = np.arange(self._nFib);
            CloseFibs = Fibs[OtherFibDistances <= nEdges];
            numCloseBy[iFib] = len(CloseFibs);
            B = np.zeros((3,3));
            for jFib in CloseFibs:
                Xs = fibCollection._tanvecs;
                Ntau = int(len(Xs)/(self._nFib));
                iStart = Ntau*jFib;
                B+=fibCollection._fiberDisc.averageXsXs(Xs[iStart:iStart+Ntau,:]);
            B*= 1/numCloseBy[iFib];
            EigValues = np.linalg.eigvalsh(B);
            NeighborOrderParams[iFib] = np.amax(np.abs(EigValues))
        return NeighborOrderParams, numCloseBy-1;

    def SparseMatrixOfConnections(self):
        """
        Returns
        --------
        Sparse matrix
            Sparse matrix of size Nsite x Nsite, whose $(i,j)$ entry
            is the number of connections (links) between sites $i$ and $j$
        """
        PairConnections = lil_matrix((self._TotNumSites,self._TotNumSites),dtype=np.int64)
        # Figure out which sites have links  
        for iLink in range(self._nDoubleBoundLinks):
            head = self._HeadsOfLinks[iLink];
            tail = self._TailsOfLinks[iLink];
            row = min(head,tail);
            col = max(head,tail);
            PairConnections[row,col]+=1;
        return PairConnections;
    
    def nLinksAllSites(self,PossiblePairs):
        """
        Return the number of links between each pair of sites in the rows
        of input PossiblePairs
        """
        PairConnections = self.SparseMatrixOfConnections();   
        nLinksPerPair=[];
        for pair in PossiblePairs:
            if (pair[1] < pair[0]):
                raise ValueError('Your pair list has to have (head) < (tail) and no duplicates')
            nLinksPerPair.append(PairConnections[pair[0],pair[1]]);    
        return nLinksPerPair;
               

    ## ==============================================
    ##    TOOLS FOR SAVING AND LOADING
    ## ==============================================
    def setLinks(self,iPts,jPts,Shifts):
        """
        Set the links from input vectors of iPts, jPts, and Shifts  
        
        Parameters
        ----------
        iPts: int array
            Heads of links
        jPts: int array
            Tails of links
        Shifts: three-dimensional array
            The periodic shift associated with each link
        """
        self._HeadsOfLinks = iPts;
        self._TailsOfLinks = jPts;
        self._PrimedShifts = Shifts;
        self._nDoubleBoundLinks = len(iPts);
    
    def setLinksFromFile(self,FileName):
        """
        Set the links from a file name. The file has a list of iPts, 
        jPts (two ends of the links), and shift in zero strain coordinates.
        
        Parameters
        -----------
        FileName: string
            The name of the file
        """
        AllLinks = np.loadtxt(FileName);
        self._nDoubleBoundLinks = len(AllLinks)-1;
        self._HeadsOfLinks = AllLinks[1:,0].astype(np.int64);
        self._TailsOfLinks = AllLinks[1:,1].astype(np.int64);
        self._PrimedShifts = AllLinks[1:,2:];
               
    def writeLinks(self,of):
        """
        Write the links to a file object of.
        """
        of.write('%d \t %d \t %f \t %f \t %f \n' %(self._nDoubleBoundLinks,-1,-1,-1,-1));
        for iL in range(self._nDoubleBoundLinks):
            of.write('%d \t %d \t %f \t %f \t %f \n' \
                %(self._HeadsOfLinks[iL], self._TailsOfLinks[iL],self._PrimedShifts[iL,0],\
                    self._PrimedShifts[iL,1],self._PrimedShifts[iL,2]));
    
    def writeLinkLocations(self,uniPoints,Dom,FileName):
        """
        Write the locations of the link heads and tails
        """
        shifts = Dom.unprimecoords(self._PrimedShifts);
        with open(FileName,"ab") as f:
            for iLink in range(self._nDoubleBoundLinks):
                x=np.concatenate((uniPoints[self._HeadsOfLinks[iLink],:], uniPoints[self._TailsOfLinks[iLink],:]+shifts[iLink,:]));
                np.savetxt(f, np.reshape(np.array([x]),(1,6)));                    

class CrossLinkedSpeciesNetwork(CrossLinkedNetwork):

    """
    Cross linked network for fibers of different species. Child class of CrossLinkedNetwork
    The main difference is that there are new mappings for a site to fiber and the first site
    on each fiber
    """

    def __init__(self,FiberSpeciesCollection,Kspring,rl,Dom,nThreads=1):
        """
        Constructor
        Takes a FiberSpeciesCollection object, the spring constant Kspring, the CL rest length rl, 
        the Domain we are operating on, and the number of threads
        """
        
        fC = FiberSpeciesCollection;
        nSpecies = fC._nSpecies;
        ds = fC._LengthBySpecies/(fC._nUniformBySpecies-1);
        print(ds)
        self._SiteToFiberMap = fC._UniformPointIndexToFiberIndex;
        if (np.amax(ds)-np.amin(ds) > 1e-14):
            raise ValueError('Must have the same density of sites on all fibers!')    
        self._ds = ds[0];
        self._kCL = Kspring;
        self._rl = rl;
        self._deltaL = min(np.sqrt(4e-3/self._kCL),0.5*self._rl);
        
        self._nDoubleBoundLinks = 0;             # number of links taken
        self._TotNumSites = fC._UniformPointStartBySpecies[nSpecies];
        
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length of the fiber.
        self._sigma = np.zeros(np.sum(fC._NFibersBySpecies));
        self._allL = np.zeros(np.sum(fC._NFibersBySpecies));
        start=0;
        for iS in range(nSpecies):
            nFibs = fC._NFibersBySpecies[iS];
            self._sigma[start:start+nFibs]=self.sigmaFromNL(fC._nChebBySpecies[iS],fC._LengthBySpecies[iS])
            self._allL[start:start+nFibs]=fC._LengthBySpecies[iS];
            start+=nFibs;
        self._nFib=start;
        print('Number of fibers %d' %self._nFib)
        print('Sigma/L is')
        print(self._sigma/self._allL)
            
        # Uniform binding locations
        Uniforms = np.zeros(self._TotNumSites);
        for iSpecies in range(nSpecies):
            su = np.tile(np.linspace(0.0,fC._LengthBySpecies[iSpecies],fC._nUniformBySpecies[iSpecies],endpoint=True),fC._NFibersBySpecies[iSpecies]);
            Uniforms[fC._UniformPointStartBySpecies[iSpecies]:fC._UniformPointStartBySpecies[iSpecies+1]]=su;
            
        self._wCheb = fC._wtsCheb;
        
        # Seed C++ random number generator and initialize the C++ variables for cross linking 
        self._DLens = Dom.getPeriodicLens();
        for iD in range(len(self._DLens)):
            if (self._DLens[iD] is None):
                self._DLens[iD] = 1e99;
                
        print(self._SiteToFiberMap)
        ChebPerFib = fC.nChebByFib();
        UniPerFib = fC.nUniByFib();
        self._CForceEvaluator = CForceEvalCpp(Uniforms,self._SiteToFiberMap,ChebPerFib,fC._sCheb,\
            self._wCheb,self._sigma,self._kCL,self._rl,nThreads);
        self._ChebStartByFib = np.zeros(self._nFib+1,dtype=np.int64);
        self._UniStartByFib = np.zeros(self._nFib+1,dtype=np.int64);
        for i in range(len(ChebPerFib)):
            self._ChebStartByFib[i+1]= ChebPerFib[i]+ self._ChebStartByFib[i];
            self._UniStartByFib[i+1]= UniPerFib[i]+ self._UniStartByFib[i];
                
    def mapSiteToFiber(self,sites):
        return np.take(self._SiteToFiberMap,sites)
    
    def mapFibToStartingSite(self,LinkedFibs):
        return np.take(self._UniStartByFib,LinkedFibs)
    
    
