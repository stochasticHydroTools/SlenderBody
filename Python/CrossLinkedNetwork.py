import numpy as np
import CrossLinkForces as clinkforcecpp
from warnings import warn 
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import time

verbose = -1;

class CrossLinkedNetwork(object):

    """
    Object with all the cross-linking information. 
    The abstraction is just a class with some cross linking binding sites 
    in it, the children implement the update methods. 
    Currently, KMCCrossLinkedNetwork implements kinetic Monte Carlo
    to update the network
    """
    
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,nThreads=1):
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
        
        self._nDoubleBoundLinks = 0;             # number of links taken
        self._TotNumSites = self._NsitesPerf*self._nFib;
        
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length of the fiber.
        self._sigma = 0.10*self._Lfib;  # standard deviation of Gaussian for smooth points
        if (self._Npf >= 32):
            self._sigma = 0.05*self._Lfib;
        elif (self._Npf >= 24):
            self._sigma = 0.075*self._Lfib;
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
        self._wCheb = fibDisc.getw();
        # Seed C++ random number generator and initialize the C++ variables for cross linking 
        self._DLens = Dom.getPeriodicLens();
        for iD in range(len(self._DLens)):
            if (self._DLens[iD] is None):
                self._DLens[iD] = 1e99;
        clinkforcecpp.initCLForcingVariables(self._su,fibDisc.gets(),fibDisc.getw(),self._sigma,self._kCL,self._rl,nThreads);
    
    ## ==================================================================
    ## METHODS FOR NETWORK UPDATE & FORCE (CALLED BY TEMPORAL INTEGRATOR)
    ## ==================================================================
    def updateNetwork(self,fiberCol,Dom,tstep,of=None):
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
        #import time
        #thist=time.time();
        Clforces = clinkforcecpp.calcCLForces(self._HeadsOfLinks[:self._nDoubleBoundLinks], \
            self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPoints,chebPoints);
        return Clforces;
        #print('First method time %f' %(time.time()-thist))
        
        # Alternative implementation for arbitrary binding pts
        thist=time.time();
        iFibs = self._HeadsOfLinks[:self._nDoubleBoundLinks]//self._NsitesPerf;
        iUnis = self._su[np.array(self._HeadsOfLinks) % self._NsitesPerf];
        jFibs = self._TailsOfLinks[:self._nDoubleBoundLinks]//self._NsitesPerf;
        jUnis = self._su[np.array(self._TailsOfLinks) % self._NsitesPerf];
        fD = fibCollection._fiberDisc;
        # Compute coefficients of all fibers
        Coefficients = np.zeros((self._nFib*self._Npf,3));
        for iFib in range(self._nFib):
            iInds = fibCollection.getRowInds(iFib);
            Coefficients[iInds,:],_=fD.Coefficients(chebPoints[iInds,:]);
        Clforces2 = clinkforcecpp.calcCLForces2(iFibs.astype(int),jFibs.astype(int),iUnis,jUnis,shifts,chebPoints,Coefficients,self._Lfib);
        print('Second method time %f' %(time.time()-thist))
        print('Error %f e-8' %(1e8*np.amax(np.abs(Clforces-Clforces2))))
 
    def CLStress(self,fibCollection,chebPts,Dom):
        """
        Compute the cross-linker contribution to the stress.
        Inputs: fibCollection = fiberCollection object to compute
        the stress on, fibDisc = fiber Discretization object describing
        each fiber, Dom = Domain (needed to compute the volume)
        """
        stress = np.zeros((3,3));
        if (self._nDoubleBoundLinks==0):
            return stress;
        uniPts = fibCollection.getUniformPoints(chebPts);
        shifts = Dom.unprimecoords(np.array(self._PrimedShifts));
        if False: # python
            for iLink in range(self._nDoubleBoundLinks): # loop over links
                iFib = self._HeadsOfLinks[iLink] // self._NsitesPerf;
                jFib = self._TailsOfLinks[iLink] // self._NsitesPerf;
                iInds = fibCollection.getRowInds(iFib);
                jInds = fibCollection.getRowInds(jFib);
                # Force on each link 
                Clforces = np.reshape(clinkforcecpp.calcCLForces([self._HeadsOfLinks[iLink]], [self._TailsOfLinks[iLink]], shifts[iLink,:],\
                    uniPts,chebPts),(chebPts.shape));
                for iPt in range(self._Npf): # increment stress: use minus since force on fluid = - force on fibers
                    stress-=self._wCheb[iPt]*np.outer(chebPts[iInds[iPt],:],Clforces[iInds[iPt],:]);
                    stress-=self._wCheb[iPt]*np.outer(chebPts[jInds[iPt],:]+shifts[iLink,:],Clforces[jInds[iPt],:]);
        # C++ function call 
        stress = clinkforcecpp.calcCLStress(self._HeadsOfLinks[:self._nDoubleBoundLinks],\
            self._TailsOfLinks[:self._nDoubleBoundLinks],shifts,uniPts,chebPts);
        #print(stressCpp-stress)
        stress/=np.prod(Dom.getLens());
        return stress;
    
    ## ==============================================
    ##     METHODS FOR NETWORK GEOMETRY INFO
    ## ==============================================   
    def numLinks(self):
        return self._nDoubleBoundLinks;
    
    def numLinksOnEachFiber(self):
        iFibs = self._HeadsOfLinks[:self._nDoubleBoundLinks]//self._NsitesPerf;
        jFibs = self._TailsOfLinks[:self._nDoubleBoundLinks]//self._NsitesPerf;
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
            linkStrains[iLink]=(np.linalg.norm(ds)-self._rl)/self._rl;
        #print(np.amax(np.abs(linkStrains)))
        return linkStrains;
    
    def getSortedLinks(self):
        AllLinks = np.zeros((self._nDoubleBoundLinks,2),dtype=np.int64);
        for iLink in range(self._nDoubleBoundLinks):
            minsite = min(self._HeadsOfLinks[iLink], self._TailsOfLinks[iLink])
            maxsite = max(self._HeadsOfLinks[iLink], self._TailsOfLinks[iLink])
            AllLinks[iLink,0] = minsite;
            AllLinks[iLink,1] = maxsite;
        return AllLinks;
    
    def ConnectionMatrix(self,bundleDist=0):
        SortedLinks = self.getSortedLinks();
        LinkedFibs = SortedLinks//self._NsitesPerf;
        LinkedSites = SortedLinks - LinkedFibs*self._NsitesPerf;
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
        bunddist = distance the links must be separated by to form a bundle. 
        """
        AdjacencyMatrix = self.ConnectionMatrix(bunddist);
        nBundles, whichBundlePerFib = connected_components(csgraph=AdjacencyMatrix, directed=False, return_labels=True)
        return nBundles, whichBundlePerFib;  
    
    def BundleOrderParameters(self,fibCollection,nBundles,whichBundlePerFib,minPerBundle=2,flowOn=False):
        """
        Get the order parameter of each bundle
        """
        BundleMatrices = np.zeros((3*nBundles,3));
        NPerBundle = np.zeros(nBundles);
        OrderParams = np.zeros(nBundles);
        averageBundleTangents = np.zeros((nBundles,3));
        for iFib in range(self._nFib):
            # Find cluster
            clusterindex = whichBundlePerFib[iFib];
            NPerBundle[clusterindex]+=1;
            iInds = fibCollection.getRowInds(iFib);
            Xs = fibCollection._tanvecs[iInds,:];
            for p in range(self._Npf):
                BundleMatrices[clusterindex*3:clusterindex*3+3,:]+=np.outer(Xs[p,:],Xs[p,:])*self._wCheb[p];
                averageBundleTangents[clusterindex]+=Xs[p,:]*self._wCheb[p];
        for clusterindex in range(nBundles):
            B = 1/(NPerBundle[clusterindex]*self._Lfib)*BundleMatrices[clusterindex*3:clusterindex*3+3,:];
            EigValues = np.linalg.eigvalsh(B);
            OrderParams[clusterindex] = np.amax(np.abs(EigValues))
            averageBundleTangents[clusterindex]*=1/(NPerBundle[clusterindex]*self._Lfib);           
        return OrderParams[NPerBundle >= minPerBundle], NPerBundle[NPerBundle >= minPerBundle], averageBundleTangents[NPerBundle >= minPerBundle];
     
    def avgBundleAlignment(self,BundleAlignmentParams,nPerBundle):
        if (np.sum(nPerBundle)==0):
            return 0;
        return np.sum(BundleAlignmentParams*nPerBundle)/np.sum(nPerBundle);
    
    def LocalOrientations(self,nEdges,fibCollection):  
        """
        For every fiber within nEdges connections of me, what is the orientation
        Returns an nFib array
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
                jInds = fibCollection.getRowInds(jFib);
                Xs = fibCollection._tanvecs[jInds,:];
                for p in range(self._Npf):
                    B+=np.outer(Xs[p,:],Xs[p,:])*self._wCheb[p];
            B*= 1/(numCloseBy[iFib]*self._Lfib);
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
        Return the number of links between all possible sites
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
        raise NotImplementedError('Have not allowed for setting links yet')
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
               
    def writeLinks(self,of,uniPts):
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
        SpatialDatabase = database of uniform points (sites)
        """
        if (verbose > 0):
            thist = time.time();
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        uniNeighbs = SpatialDatabase.selfNeighborList(self._rl+self._deltaL);
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
        # Post process list to exclude pairs that are too close (or too far b/c of sheared transformation)
        for iMaybeLink in range(nPotentialLinks):
            iPt = newLinks[iMaybeLink,0];
            jPt = newLinks[iMaybeLink,1];
            rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
            r = np.linalg.norm(rvec);
            if (r < self._rl+self._deltaL and r > self._rl-self._deltaL):
                GoodInds.append(iMaybeLink);
                PrimedShifts[iMaybeLink,:] = Dom.primecoords(uniPts[iPt,:]-uniPts[jPt,:] - rvec);    
        newLinks = newLinks[GoodInds,:];
        PrimedShifts = PrimedShifts[GoodInds,:];
        #PrimedShifts = np.concatenate((PrimedShifts,-PrimedShifts));
        #newLinks = np.concatenate((newLinks,np.fliplr(newLinks)));
        return newLinks, PrimedShifts;                        

    
