import numpy as np
from CrossLinkForceEvaluator import CrossLinkForceEvaluator as CForceEvalCpp
from warnings import warn 
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import time

verbose = -1;

# Documentation last updated: 03/09/2021

class CrossLinkedNetwork(object):

    """
    Object with all the cross-linking information. 
    The abstract parent class is a network of fibers that are all the same, and 
    implements methods for computing force and stress, and information about
    the network morphology (number of bundles, etc.)
    There is a child class below which makes some small modifications for a fiber
    collection
    """
    
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,smoothForce,nThreads=1):
        """
        Initialize the CrossLinkedNetwork. 
        Input variables: nFib = number of fibers, N = number of points per
        fiber, Nunisites = number of uniformly placed CL binding sites per fiber
        Lfib = length of each fiber
        kCL = cross linking spring constant, rl = rest length of the CLs, 
        Dom = domain object, fibDisc = fiber collocation discretization, Threads = number 
        of threads for openMP calculation of CL forces and stress
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
        print('Sigma/L is %f' %(self._sigma/self._Lfib))
            
        # Cutoff bounds to form the CLs
        self._deltaL = min(np.sqrt(4e-3/self._kCL),0.5*self._rl);
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
        Obtain the CL smoothness density sigma as a function of the 
        number of points on a fiber N and length of the fiber L
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
        """
        Method to update the network. 
        """
        raise NotImplementedError('No update implemented for abstract network')
    
    def CLForce(self,uniPoints,chebPoints,Dom,fibCollection):
        """
        Compute the total cross linking force.
        Inputs: uniPoints = uniform points for all fibers, chebPoints = Chebyshev points 
        on all fibers, Dom = Domain object. 
        Outputs: nPts*3 one-dimensional array of the forces at the 
        Chebyshev points due to the CLs
        """
        if (self._nDoubleBoundLinks==0): # if no CLs don't waste time
            return np.zeros(3*self._Npf*self._nFib);
        # Call the C++ function to compute forces
        shifts = Dom.unprimecoords(self._PrimedShifts[:self._nDoubleBoundLinks,:]);
        #thist=time.time();
        if (self._smoothForce):
            Clforces = self._CForceEvaluator.calcCLForces(self._HeadsOfLinks[:self._nDoubleBoundLinks], \
                self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPoints,chebPoints);
        else:
            Clforces = self._CForceEvaluator.calcCLForcesEnergy(self._HeadsOfLinks[:self._nDoubleBoundLinks], \
                    self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPoints,chebPoints);
        #print('First method time %f' %(time.time()-thist))
        return Clforces;
        
 
    def CLStress(self,fibCollection,chebPts,Dom):
        """
        Compute the cross-linker contribution to the stress.
        Inputs: fibCollection = fiberCollection object to compute
        the stress on, fibDisc = fiber Discretization object describing
        each fiber, Dom = Domain (needed to compute the volume)
        """
        if (self._nDoubleBoundLinks==0):
            return 0;
        uniPts = fibCollection.getUniformPoints(chebPts);
        shifts = Dom.unprimecoords(np.array(self._PrimedShifts));
        #np.savetxt('uniPts.txt',uniPts)
        #np.savetxt('shifts.txt',shifts)
        #np.savetxt('Heads.txt',self._HeadsOfLinks[:self._nDoubleBoundLinks])
        #np.savetxt('Tails.txt',self._TailsOfLinks[:self._nDoubleBoundLinks])
        #np.savetxt('Chebpts.txt',chebPts)
        if (self._smoothForce):
            raise TypeError('Stress only implemented with nonsmooth (energy-based) CL forcing')
        stress = self._CForceEvaluator.calcCLStressEnergy(self._HeadsOfLinks[:self._nDoubleBoundLinks],\
            self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPts,chebPts);
        #print('CPP stress %f' %stress)
        stress/=Dom.getVol();
        #np.savetxt('CLStress.txt',stress)
        return stress;
    
    ## ==============================================
    ##     METHODS FOR NETWORK GEOMETRY INFO
    ## ==============================================   
    def numLinks(self):
        return self._nDoubleBoundLinks;
    
    def mapSiteToFiber(self,sites):
        return sites//self._NsitesPerf;
    
    def mapFibToStartingSite(self,LinkedFibs):
        """
        The first uniform site on a given fiber
        """
        return LinkedFibs*self._NsitesPerf;
    
    def numLinksOnEachFiber(self):
        iFibs = self.mapSiteToFiber(self._HeadsOfLinks[:self._nDoubleBoundLinks])
        jFibs = self.mapSiteToFiber(self._TailsOfLinks[:self._nDoubleBoundLinks])
        NumOnEachFiber = np.zeros(self._nFib,dtype=np.int64)
        for iLink in range(len(iFibs)):
            NumOnEachFiber[iFibs[iLink]]+=1;
            NumOnEachFiber[jFibs[iLink]]+=1;
        return NumOnEachFiber; # Sum is 2 x the number of links
        
    def calcLinkStrains(self,uniPoints,Dom):
        """
        Method to compute the strain in each link
        Inputs: uniPoints = uniform points on the fibers to which the 
        links are bound, Dom = domain object for periodic shifts 
        Return: an array of nLinks sisze that has the signed strain ||link length|| - rl
        for each link 
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
        Method to compute the strain in each link
        Inputs: uniPoints = uniform points on the fibers to which the 
        links are bound, Dom = domain object for periodic shifts 
        Return: an array of nLinks sisze that has the signed strain ||link length|| - rl
        for each link 
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
        Build a matrix that gives the fibers connected to each other. 
        If bundleDist = 0, it will give you the adjacency matrix with nonzero 
        entries at (i,j) if there is one link between fibers i and j. 
        If bundleDist > 0, the nonzero entries (i,j) have 2 links between them
        separated by a distance at least bundleDist
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
        Find bundles in the network. We quantify a bundle as two fibers linked with links 
        at least bunddist apart on their axes. 
        This method returns the number of bundles and the labels (which bundle each fiber is in)
        """
        AdjacencyMatrix = self.ConnectionMatrix(bunddist);
        nBundles, whichBundlePerFib = connected_components(csgraph=AdjacencyMatrix, directed=False, return_labels=True)
        return nBundles, whichBundlePerFib;  
    
    def BundleOrderParameters(self,fibCollection,nBundles,whichBundlePerFib,minPerBundle=2):
        """
        Get the order parameter of each bundle. The order parameter for a bundle of F filaments 
        is defined as the maximum eigenvalue of the matrix 
        1/F *sum(1/L_i integral_0^L_i {Xs(s)Xs(s) ds}, where we discretize the integral by direct
        Clenshaw-Curtis quadrature
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
        Calculate the weighted average of alignment across all bundles. 
        Inputs are the alignment parameters
        for each bundle, and the number of fibers in each bundle.
        """
        if (np.sum(nPerBundle)==0):
            return 0;
        return np.sum(BundleAlignmentParams*nPerBundle)/np.sum(nPerBundle);
    
    def LocalOrientations(self,nEdges,fibCollection):  
        """
        Calculate the orientiation parameter for every fiber 
        within nEdges of every fiber. 
        Returns an nFib array of the orientations and an nFib array of the number
        of connected fibers for each fiber
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
        """
        self._HeadsOfLinks = iPts;
        self._TailsOfLinks = jPts;
        self._PrimedShifts = Shifts;
        self._nDoubleBoundLinks = len(iPts);
    
    def setLinksFromFile(self,FileName):
        """
        Set the links from a file name. The file has a list of iPts, 
        jPts (two ends of the links), and shift in zero strain coordinates 
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
       
    ## ==============================================
    ##     PRIVATE METHODS ONLY ACCESSED IN CLASS
    ## ==============================================
    def getPairsThatCanBind(self,fiberCol,Dom):
        """
        Generate list of events for binding of a doubly-bound link to both sites. 
        SpatialDatabase = database of uniform points (sites). This is a 
        method for debugging (not called in the final production code)
        """
        if (verbose > 0):
            thist = time.time();
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        uniNeighbs = SpatialDatabase.selfNeighborList(self._rl+self._deltaL,self._NsitesPerf);
        if (verbose > 0):
            print('ckD neighbor search time %f' %(time.time()-thist));
        uniNeighbs = uniNeighbs.astype(np.int64);
        thist = time.time();
        # Filter the list of neighbors to exclude those on the same fiber
        Fibs = uniNeighbs // self._NsitesPerf;
        delInds = np.arange(len(Fibs[:,0]));
        newLinks = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
        nPotentialLinks = len(newLinks[:,0]);
        PrimedShifts = np.zeros((nPotentialLinks,3));
        GoodInds = [];
        distances = np.zeros(nPotentialLinks)
        # Post process list to exclude pairs that are too close (or too far b/c of sheared transformation)
        for iMaybeLink in range(nPotentialLinks):
            iPt = newLinks[iMaybeLink,0];
            jPt = newLinks[iMaybeLink,1];
            rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
            r = np.linalg.norm(rvec);
            distances[iMaybeLink]=r;
            if (r < self._rl+self._deltaL and r > self._rl-self._deltaL):
                GoodInds.append(iMaybeLink);
                PrimedShifts[iMaybeLink,:] = Dom.primecoords(uniPts[iPt,:]-uniPts[jPt,:] - rvec);    
        newLinks = newLinks[GoodInds,:];
        PrimedShifts = PrimedShifts[GoodInds,:];
        #PrimedShifts = np.concatenate((PrimedShifts,-PrimedShifts));
        #newLinks = np.concatenate((newLinks,np.fliplr(newLinks)));
        return newLinks, PrimedShifts, distances[GoodInds];                        

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
    
    
