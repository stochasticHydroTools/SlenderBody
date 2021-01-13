from CrossLinkedNetwork import CrossLinkedNetwork
import scipy.sparse as sp
import numpy as np
from CrossLinkingEvent import EventQueue as heap

class DoubleEndedCrossLinkedNetwork(CrossLinkedNetwork):

    """
    This class is a child of CrossLinkedNetwork which implements a network 
    with cross links where each end matters separately. 
    
    There are 3 reactions
    1) Binding of a floating link to one site (rate _kon)
    2) Unbinding of a link that is bound to one site to become free (reverse of 1, rate _koff)
    3) Binding of a singly-bound link to another site to make a doubly-bound CL (rate _konSecond)
    4) Unbinding of a double bound link in one site to make a single-bound CL (rate _koffSecond)
    5) Binding of both ends of a CL (rate _kDoubleOn)
    6) Unbinding of both ends of a CL (rate _kDoubleOff)
    
    This is the pure python version. See below object DoubleEndedCrossLinkedNetworkCPP for 
    implementation using C++ and fortran. 
    
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,konsecond,CLseed,Dom,fibDisc,nThreads=1):
        """
        Constructor
        """
        super().__init__(nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads);
        self._TotNumSites = self._NsitesPerf*self._nFib;
        self._FreeLinkBound = np.zeros(self._TotNumSites,dtype=np.int64); # number of free-ended links bound to each site
        self._konSecond = konsecond;
        self._nFreeEnds = 0;
        self._nDoubleBoundLinks = 0;
        self._MaxLinks = 2*int(konsecond/koff*kon/koff)*self._TotNumSites
        self._HeadsOfLinks = np.zeros(self._MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(self._MaxLinks,dtype=np.int64);
        self._kDoubleOn = 0;
        self._kDoubleOff = 0;
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
        
    def nLinksAllSites(self,fiberCol,Dom):
        """
        Return the number of links between all possible sites
        """
        PairConnections = sp.lil_matrix((self._TotNumSites,self._TotNumSites),dtype=np.int64)
        # Figure out which sites have links  
        for iLink in range(self._nDoubleBoundLinks):
            head = self._HeadsOfLinks[iLink];
            tail = self._TailsOfLinks[iLink];
            row = min(head,tail);
            col = max(head,tail);
            PairConnections[row,col]+=1;
        
        # Compute possible pairs
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
                PossiblePairs.append(BothEndBindings[iPair,:]);
        
        for pair in PossiblePairs:
            nLinksPerPair.append(PairConnections[pair[0],pair[1]]);    
        return nLinksPerPair, PossiblePairs;
    
    def updateNetwork(self,fiberCol,Dom,tstep,of=None):
        """
        Update the network using Kinetic MC.
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        This is an event-driven algorithm using a heap for fast processing.
        """
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
        BaseCLRates, FirstLinkPerSite,NextLinkByLink = self.getBaseSecondBindingRates(UniquePairs,uniPts,Dom);
        RatesSecondBind, TimesSecondBind = self.calcAllSecondBindingRates(nPairs,BaseCLRates,BindingPairs);
        TruePairs = PairIndices[BaseCLRates > 0];
        nTruePairs = len(TruePairs);
        # Initialize que
        queue = heap([],self._TotNumSites+4+2*nPairs);
        # Add double binding to heap
        indexstart = 0;
        for iEv in range(len(TimesSecondBind)):
            queue.heappush(indexstart,TimesSecondBind[iEv],tstep)
            indexstart+=1;
        
        # Rate and time for free link binding
        RateFreeBind = self._kon*self._TotNumSites;
        TimeFreeBind = self.logrand()/RateFreeBind;
        # Add binding to heap
        indexFreeBinding = indexstart;
        queue.heappush(indexFreeBinding,TimeFreeBind,tstep)
        indexstart+=1;
        
        # For free link unbinding
        nSites = self._TotNumSites;
        RatesFreeUnbind = self._koff*self._FreeLinkBound;
        TimesFreeUnbind = self.logrand(nSites)/RatesFreeUnbind;
        # Add to heap
        indexFreeUnbinding = indexstart;
        for iEv in range(len(TimesFreeUnbind)):
            queue.heappush(indexstart,TimesFreeUnbind[iEv],tstep)
            indexstart+=1;
        
        # For CL unbinding
        RateSecondUnbind = self._koff*self._nDoubleBoundLinks;
        TimeSecondUnbind = self.logrand()/RateSecondUnbind;
        # Add to heap
        indexSecondUnbind = indexstart;
        queue.heappush(indexstart,TimeSecondUnbind,tstep)
        
        # Events for double binding and unbinding
        RateDoubleBind = self._kDoubleOn*nTruePairs;
        TimeDoubleBind = self.logrand()/RateDoubleBind;
        indexDoubleBind = indexstart+1;
        queue.heappush(indexDoubleBind,TimeDoubleBind,tstep)
        
        RateDoubleUnbind = self._kDoubleOff*self._nDoubleBoundLinks;
        TimeDoubleUnbind = self.logrand()/RateDoubleUnbind;
        indexDoubleUnbind = indexstart+2;
        queue.heappush(indexDoubleUnbind,TimeDoubleUnbind,tstep)
        
        import time
        thist = time.time();
        systime=0;
        while (systime < tstep):
            # Compute rates for binding, unbinding, one end unbinding, both ends binding
            theEvent = queue.heappop();
            if (theEvent.time() < systime):
                raise ValueError('You went backwards in time!') 
            systime = theEvent.time();
            event = theEvent.index();
            linkChange = False;
            if (event==indexFreeBinding): # end binding, choose random site and bind an end
                #print('Index %d and binding' %event)
                BoundEnd = int(np.random.rand()*self._TotNumSites);
                PlusOrMinusSingleBound=1;
                # Update binding time in queue
                TimeFreeBind = self.logrand()/RateFreeBind+systime;
                queue.heappush(indexFreeBinding,TimeFreeBind,tstep)
            elif (event >= indexFreeUnbinding and event < indexSecondUnbind): 
                #print('Index %d and unbinding end' %event)
                BoundEnd = event-indexFreeUnbinding;
                PlusOrMinusSingleBound=-1;
            elif (event == indexSecondUnbind or event==indexDoubleUnbind): # CL unbinding
                linkNum = int(np.random.rand()*self._nDoubleBoundLinks);
                BoundEnd = self._HeadsOfLinks[linkNum];
                UnbindingEnd = self._TailsOfLinks[linkNum];
                # Unbind it (remove from lists)
                self._HeadsOfLinks[linkNum] = self._HeadsOfLinks[self._nDoubleBoundLinks-1];
                self._TailsOfLinks[linkNum] = self._TailsOfLinks[self._nDoubleBoundLinks-1];
                self._nDoubleBoundLinks-=1;
                linkChange = True;
                # Add a free end at the other site. Always assume the remaining bound end is at the left
                if (event == indexSecondUnbind):
                    #print('Index %d and unbinding CL end' %event)
                    PlusOrMinusSingleBound=1;
                else:
                    #print('Index %d and unbinding both CL end' %event)
                    PlusOrMinusSingleBound=0;
            else: # CL binding
                # The index now determines which pair of sites the CL is binding to
                # In addition, we always assume the bound end is the one at the left 
                if (event == indexDoubleBind):
                    #print('Index %d and double binding' %index)
                    pair = TruePairs[int(np.random.rand()*nTruePairs)];
                    BoundEnd = BindingPairs[pair,0];
                    UnboundEnd = BindingPairs[pair,1]; 
                    PlusOrMinusSingleBound=0;
                    TimeDoubleBind = self.logrand()/RateDoubleBind+systime;
                    queue.heappush(indexDoubleBind,TimeDoubleBind,tstep)   # push double binding
                else:
                    #print('Index %d and binding CL end' %event)
                    BoundEnd = BindingPairs[event,0];
                    UnboundEnd = BindingPairs[event,1];
                    PlusOrMinusSingleBound=-1;
                self._HeadsOfLinks[self._nDoubleBoundLinks] = BoundEnd;
                self._TailsOfLinks[self._nDoubleBoundLinks] = UnboundEnd;
                self._nDoubleBoundLinks+=1;
                linkChange = True;
                
            
            # Recompute subset of rates and times
            self._nFreeEnds+= PlusOrMinusSingleBound;
            self._FreeLinkBound[BoundEnd]+= PlusOrMinusSingleBound
            
            # Rates of CL binding change based on number of bound ends
            RatesSecondBind, TimesSecondBind, updatedLinks = self.updateSecondBindingRate(BoundEnd,BaseCLRates,\
                RatesSecondBind,TimesSecondBind,FirstLinkPerSite,NextLinkByLink,systime);
            
            #misses = np.setdiff1d(PairIndices[BindingPairs[:,0]==BoundEnd],updatedLinks);
            #if (np.any(BaseCLRates[misses])):
            #    raise ValueError('Link missed!')

            for iInd in (PairIndices[BindingPairs[:,0]==BoundEnd]):
                queue.heapupdate(iInd,TimesSecondBind[iInd],tstep);
                
            # Update unbinding event at BoundEnd (the end whose state has changed)
            RatesFreeUnbind[BoundEnd] = self._koff*self._FreeLinkBound[BoundEnd];
            TimesFreeUnbind[BoundEnd] = self.logrand()/RatesFreeUnbind[BoundEnd]+systime;
            queue.heapupdate(BoundEnd+indexFreeUnbinding,TimesFreeUnbind[BoundEnd],tstep)         
            
            # Update unbinding events (links)
            if (linkChange):
                RateSecondUnbind = self._koff*self._nDoubleBoundLinks;
                TimeSecondUnbind = self.logrand()/RateSecondUnbind+systime;
                queue.heapupdate(indexSecondUnbind,TimeSecondUnbind,tstep)
                RateDoubleUnbind = self._kDoubleOff*self._nDoubleBoundLinks;
                TimeDoubleUnbind = self.logrand()/RateDoubleUnbind+systime;
                queue.heapupdate(indexDoubleUnbind,TimeDoubleUnbind,tstep)
            
            #queue.checkIndices(indexSecondUnbind+2);
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
        
    def logrand(self,N=1):
        return -np.log(1-np.random.rand(N));
        
    def getPairsThatCanBind(self,SpatialDatabase):
        """
        Generate list of events for binding of a doubly-bound link to both sites. 
        SpatialDatabase = database of uniform points (sites)
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
    
    def getBaseSecondBindingRates(self,BothEndBindings,uniPts,Dom):
        """
        Calculate the base rates of binding for a pair of sites. 
        Really what this method is doing is checking that the sites
        are FAR enough apart (we already know they are close enough
        from neighbor search) to have a link form between them. In the
        python version, it also computes the link lists for fast searching
        through the links. 
        """
        nPairs = len(BothEndBindings[:,0]);
        BothEndBindingRates = np.zeros(2*nPairs);
        FirstLinkPerSite = np.zeros(self._TotNumSites,dtype=np.int64)-1;
        NextLinkByLink = np.zeros(2*nPairs,dtype=np.int64)-1;
        for iPair in range(nPairs):
            thiskOn = self.calcKonSecondRate(BothEndBindings[iPair,0],BothEndBindings[iPair,1],uniPts,Dom);
            for bE in [0,1]:       
                BoundEnd = BothEndBindings[iPair,bE];     
                UnboundEnd = BothEndBindings[iPair,1-bE];
                if (thiskOn > 0):
                    linkNum = iPair+bE*nPairs;
                    BothEndBindingRates[linkNum]=thiskOn;
                    # Build link lists
                    if (FirstLinkPerSite[BoundEnd]==-1):
                        FirstLinkPerSite[BoundEnd] = linkNum;
                    else:
                        newLink = FirstLinkPerSite[BoundEnd]; 
                        while (newLink > -1):
                            prevnewLink = newLink;
                            newLink = NextLinkByLink[newLink];   
                        NextLinkByLink[prevnewLink] = linkNum;
        return BothEndBindingRates, FirstLinkPerSite,NextLinkByLink;
    
    def calcAllSecondBindingRates(self,nPairs,BaseRates,BothEndBindings):
        """
        Update the binding rates for ALL links based on the number of free
        links bound at the left end of the link. Also computes times of binding
        for those links
        """
        BothEndRates = np.zeros(2*nPairs);
        BothEndBindingTimes = np.zeros(2*nPairs)
        for iPair in range(2*nPairs):
            BoundEnd = BothEndBindings[iPair,0];  
            BothEndRates[iPair] = BaseRates[iPair]*self._FreeLinkBound[BoundEnd];
            BothEndBindingTimes[iPair] =self.logrand()/BothEndRates[iPair];
        return BothEndRates, BothEndBindingTimes;    
    
    def updateSecondBindingRate(self,BoundEnd,BaseRates,BothEndRates,BothEndBindingTimes,FirstLink,NextLink,systime=0):
        """
        Update the binding rates for any link with leftend = BoundEnd. This is done by 
        going through the linked lists starting at first[BoundEnd]
        """
        ThisLink = FirstLink[BoundEnd];
        updatedLinks=[];
        while (ThisLink > -1):
            BothEndRates[ThisLink] = BaseRates[ThisLink]*self._FreeLinkBound[BoundEnd];
            eltime = self.logrand()/BothEndRates[ThisLink];
            BothEndBindingTimes[ThisLink] = eltime + systime;
            updatedLinks.append(ThisLink);
            ThisLink = NextLink[ThisLink];
        return BothEndRates, BothEndBindingTimes, updatedLinks;  
    

from EndedCrossLinkedNetwork import EndedCrossLinkedNetwork
class DoubleEndedCrossLinkedNetworkCPP(DoubleEndedCrossLinkedNetwork):

    """
    Version that uses C++ and fortran code
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
        super().__init__(nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,konsecond,CLseed,Dom,fibDisc,nThreads);
        # C++ initialize
        allRates = [self._kon,self._konSecond,self._koff,self._koff,self._kDoubleOn,self._kDoubleOff];
        self._cppNet = EndedCrossLinkedNetwork(self._TotNumSites, allRates, CLseed);
           
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._cppNet.getNBoundEnds();
            
    def nLinksAllSites(self,fiberCol,Dom):
        """
        Return the number of links between all possible sites
        """
        self._HeadsOfLinks = self._cppNet.getLinkHeadsOrTails(True);
        self._TailsOfLinks = self._cppNet.getLinkHeadsOrTails(False);
        self._nDoubleBoundLinks = len(self._HeadsOfLinks);
        return super().nLinksAllSites(fiberCol,Dom);
            
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
        BaseCLRates = self.getBaseSecondBindingRates(UniquePairs,uniPts,Dom);
        TruePairs = PairIndices[BaseCLRates > 0];
        import time
        thist = time.time();
        self._cppNet.updateNetwork(tstep,BindingPairs[TruePairs,:]);
        
        print('Time step time %f ' %(time.time()-thist))       
    
    def getBaseSecondBindingRates(self,BothEndBindings,uniPts,Dom):
        """
        Calculate the base rates of binding for a pair of sites. 
        Really what this method is doing is checking that the sites
        are FAR enough apart (we already know they are close enough
        from neighbor search) to have a link form between them. 
        """
        nPairs = len(BothEndBindings[:,0]);
        BothEndBindingRates = np.zeros(2*nPairs);
        FirstLinkPerSite = np.zeros(self._TotNumSites,dtype=np.int64)-1;
        NextLinkByLink = np.zeros(2*nPairs,dtype=np.int64)-1;
        for iPair in range(nPairs):
            thiskOn = self.calcKonSecondRate(BothEndBindings[iPair,0],BothEndBindings[iPair,1],uniPts,Dom);
            for bE in [0,1]:       
                if (thiskOn > 0):
                    linkNum = iPair+bE*nPairs;
                    BothEndBindingRates[linkNum]=thiskOn;
        return BothEndBindingRates;
        

