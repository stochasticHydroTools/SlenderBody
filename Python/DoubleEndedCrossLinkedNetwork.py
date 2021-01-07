from CrossLinkedNetwork import CrossLinkedNetwork
import scipy.sparse as sp
import numpy as np

class DoubleEndedCrossLinkedNetwork(CrossLinkedNetwork):

    """
    This class is a child of CrossLinkedNetwork which implements a network 
    with cross links where each end matters seaprately
    There are 4 rates:
    Rate of one end to attach (self._kon) 
    Rate of the other end to attach (self._konSecond)
    Rate of one end to dettach (self._koff)
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,konsecond,CLseed,Dom,fibDisc,nThreads=1):
        """
        Constructor
        # In addition to iPts and jPts (lists of completed links), there is now a list of potential sites
        # that have a link attached that could bind to an available other site
        This list is self._FreeLinkBound. All other information is samea as super 
        """
        super().__init__(nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads);
        # We need the following data structures
        # 1) A count of the number of free links bound to each site
        # 2) A count of the number of links connecting each pair
        # For 2) we pre-allocate a sparse matrix of size Nsite x Nsite
        # self._PairConnections[i,j] is the number of connections where the left-end of the link is bound to site i 
        # and the right end is bound to site j
        self._TotNumSites = self._NsitesPerf*self._nFib;
        self._SiteIndices = np.arange(self._TotNumSites,dtype=np.int64);
        self._FreeLinkBound = np.zeros(self._TotNumSites,dtype=np.int64); # says whether or not a link is bound to a site with a free end
        self._PairConnections = sp.lil_matrix((self._TotNumSites,self._TotNumSites),dtype=np.int64)
        self._konSecond = konsecond;
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
    
    def getnLinksBetween2Sites(self,Pair):
        return self._PairConnections[Pair[0],Pair[1]]+self._PairConnections[Pair[1],Pair[0]];
        
    def nLinksAllSites(self,fiberCol,Dom):
        """
        Return the number of links between all possible sites
        """
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = newLinks.shape;
        BothEndBindings = newLinks.astype(np.int64);
        PossiblePairs = [];
        nLinksPerPair = [];
        for iPair in range(nPairs):            
            if (self.calcKonOne(BothEndBindings[iPair,0],BothEndBindings[iPair,1],uniPts,Dom) > 0):
                nLinksPerPair.append(self.getnLinksBetween2Sites(BothEndBindings[iPair,:]));
                PossiblePairs.append(BothEndBindings[iPair,:]);
        print(nLinksPerPair)
        return nLinksPerPair, PossiblePairs;
    
    def updateNetwork(self,fiberCol,Dom,tstep,of=None):
        """
        Update the network using Kinetic MC.
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        This is an event-driven algorithm. See comments throughout, but
        the basic idea is to sample times for each event, then update 
        those times as the state of the network changes. 
        """
        if (self._nCL==0): # do nothing if there are no CLs
            return; 
        # Obtain Chebyshev points and uniform points, and get
        # the neighbors of the uniform points
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        
        # Compute the list of neighbors (constant for this step)
        nPairs, BindingPairs = self.getPairsThatCanBind(SpatialDatabase)
        UniquePairs = BindingPairs[0:nPairs,:];
        PairIndices = np.arange(2*nPairs,dtype=np.int64)
        
        # Precompute total number quantities (to be incremented)
        self._nFreeEnds = np.sum(self._FreeLinkBound);
        self._nDoubleBoundLinks = self._PairConnections.sum();
        allRates = np.zeros(2*nPairs+3);
        allTimes = np.zeros(2*nPairs+3);
        
        # Initial rates and times
        systime = 0;
        allRates[0] = self.getOneBindingRate();  # always the same, based on # of sites
        allRates[1] = self.getOneUnbindingRate(); # based on # of currently bound single ends
        allRates[2] = self.getOneCLEndUnbindingRate(); # based on # of doubly-bound links
        allTimes[0:3] = -np.log(1-np.random.rand(3))/allRates[0:3];
        BaseCLRates = self.getSecondBindingRates(UniquePairs,uniPts,Dom);
        allRates[3:], allTimes[3:] = self.updateSecondBindingRate(BaseCLRates,allRates[3:],allTimes[3:],PairIndices,BindingPairs,systime)
        import time
        thist = time.time();
        while (systime < tstep):
            # Compute rates for binding, unbinding, one end unbinding, both ends binding
            event = np.argmin(allTimes);
            systime = allTimes[event];
            if (event==0): # end binding, choose random site and bind an end
                BoundEnd = int(np.random.rand()*self._TotNumSites);
                PlusOrMinusSingleBound=1;
            elif (event==1): # free end unbinding, choose random site with correct probability and unbind
                probabilities = self._FreeLinkBound/self._nFreeEnds;
                BoundEnd = np.random.choice(self._SiteIndices,p=probabilities);
                PlusOrMinusSingleBound=-1;
            elif (event==2): # one end of CL unbinding
                # Choose a CL (random entry from sparse matrix) 
                NonzeroEntries = sp.find(self._PairConnections)
                LeftEnds = NonzeroEntries[0];
                RightEnds = NonzeroEntries[1];
                probabilities = NonzeroEntries[2]/self._nDoubleBoundLinks;
                # Choose the link and end
                Link = np.random.choice(np.arange(len(LeftEnds),dtype=np.int64),p=probabilities);
                # Unbind it (remove from matrix)
                self._PairConnections[LeftEnds[Link],RightEnds[Link]]-=1;
                self._nDoubleBoundLinks-=1;
                # Add a free end at the other site. Always assume the remaining bound end is at the left
                BoundEnd = LeftEnds[Link];
                PlusOrMinusSingleBound=1;   
            else: # second end of CL binding
                # The index now determines which pair of sites the CL is binding to
                # In addition, we always assume the bound end is the one at the left 
                linkindex = event-3;
                Pair = BindingPairs[linkindex,:];
                self._PairConnections[Pair[0],Pair[1]]+=1;
                self._nDoubleBoundLinks+=1;
                BoundEnd = Pair[0];
                PlusOrMinusSingleBound=-1;
            # Recompute subset of rates and times
            self._nFreeEnds+= PlusOrMinusSingleBound;
            self._FreeLinkBound[BoundEnd]+= PlusOrMinusSingleBound
            allRates[1] = self.getOneUnbindingRate(); # based on # of currently bound single ends (this always changes)
            allRates[2] = self.getOneCLEndUnbindingRate(); # not always necessary but too easy
            allTimes[:3] = -np.log(1-np.random.rand(3))/allRates[:3]+systime;
            # For doubly-binding links, only the end where the one end is already bound needs to updated
            # (only need to search the first column)
            allRates[3:], allTimes[3:] = self.updateSecondBindingRate(BaseCLRates,allRates[3:],\
                allTimes[3:],PairIndices[BindingPairs[:,0]==BoundEnd],BindingPairs,systime)
            
        print('Time step time %f ' %(time.time()-thist))
            
    
    ## ======================================== ##
    ##    PRIVATE METHODS (INVOLVED IN UPDATE)
    ## ======================================== ##  
    def calcKonSecondRate(self,iPt,jPt,uniPts,Dom):
        """
        Calculate the binding rate for a single link. 
        Inputs: link endpoints iPt, jPt. 
        uniPts = (Nsites*Nfib)x3 array of uniform binding 
        sites on the fibers, Dom = Domain object
        Output: the rate for a single event of binding.
        """
        rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
        r = np.linalg.norm(rvec);
        if (r < (1+self._uppercldelta)*self._rl and r > (1-self._lowercldelta)*self._rl):
            return self._konSecond; # binds with probability 1 at every time step
        return 0;  
     
    def getOneBindingRate(self):
        """
        Compute rate for binding 
        """
        return self._kon*self._TotNumSites;
    
    def getOneUnbindingRate(self):
        """
        Compute rate for unbinding, which is the number of freely bound links x koff
        """
        return self._koff*self._nFreeEnds;
    
    def getOneCLEndUnbindingRate(self): 
        """
        Generate list of events for unbinding of one end of a doubly-bound link
        """  
        OneEndofCLUnbindingRate = 2*self._koff*self._nDoubleBoundLinks; # There are 2 ends (2 events per link)
        return OneEndofCLUnbindingRate;
   
    def getPairsThatCanBind(self,SpatialDatabase):
        """
        Generate list of events for binding of a doubly-bound link to both sites
        """
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        iPts = newLinks[:,0].astype(np.int64);
        jPts = newLinks[:,1].astype(np.int64);
        nPairs = len(iPts);
        BothEndBindings = np.zeros((2*nPairs,2),dtype=np.int64);
        BothEndBindings[0:nPairs,:] = newLinks;
        BothEndBindings[nPairs:,0] = jPts;
        BothEndBindings[nPairs:,1] = iPts;
        return nPairs, BothEndBindings;
    
    def getSecondBindingRates(self,BothEndBindings,uniPts,Dom):
        nPairs = len(BothEndBindings[:,0]);
        BothEndBindingRates = np.zeros(2*nPairs);
        for iPair in range(nPairs):
            for bE in [0,1]:       
                BoundEnd = BothEndBindings[iPair,bE];     
                UnboundEnd = BothEndBindings[iPair,1-bE];
                # Rate is k_on, second * number of free links bound at that end
                BothEndBindingRates[iPair+bE*nPairs] = self.calcKonSecondRate(BoundEnd,UnboundEnd,uniPts,Dom);
        return BothEndBindingRates;
    
    def updateSecondBindingRate(self,BaseRates,BothEndRates,BothEndBindingTimes,PairIndices,BothEndBindings,systime):
        for iPair in PairIndices:
            BoundEnd = BothEndBindings[iPair,0];   
            BothEndRates[iPair] = BaseRates[iPair]*self._FreeLinkBound[BoundEnd];
            BothEndBindingTimes[iPair] = -np.log(1-np.random.rand())/BothEndRates[iPair] + systime;
        return BothEndRates, BothEndBindingTimes;    
    
    
    
        

