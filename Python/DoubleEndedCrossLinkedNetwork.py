from CrossLinkedNetwork import CrossLinkedNetwork
import scipy.sparse as sp
import numpy as np
from CrossLinkingEvent import EventQueue as heap
from CrossLinkingEvent import CrossLinkingEvent as cle 

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
        self._konSecond = konsecond;
        self._nFreeEnds = 0;
        self._nDoubleBoundLinks = 0;
        self._PairConnections = sp.lil_matrix((self._TotNumSites,self._TotNumSites),dtype=np.int64)
        self._MaxLinks = 2*int(konsecond/koff*kon/koff)*self._TotNumSites
        self._MaxEnds = 2*int(kon/koff)*self._TotNumSites
        self._MaxEventsOtherThanBinding = 2*self._MaxLinks+self._MaxEnds+1;
        self._HeadsOfLinks = np.zeros(self._MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(self._MaxLinks,dtype=np.int64);
        
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
        BaseCLRates, FirstLinkPerSite,NextLinkByLink = self.getSecondBindingRates(UniquePairs,uniPts,Dom);
        RatesSecondBind, TimesSecondBind = self.calcAllSecondBindingRates(nPairs,BaseCLRates,BindingPairs);
        # Initialize que
        queue = heap([],self._MaxEventsOtherThanBinding+2*nPairs);
        # Add double binding to heap
        indexstart = 0;
        for iEv in range(len(TimesSecondBind)):
            queue.heappush(cle(TimesSecondBind[iEv],indexstart),tstep)
            indexstart+=1;
        
        # Rate and time for free link binding
        RateFreeBind = self.getOneBindingRate();
        TimeFreeBind = self.logrand()/RateFreeBind;
        # Add binding to heap
        indexFreeBinding = indexstart;
        queue.heappush(cle(TimeFreeBind,indexFreeBinding),tstep)
        indexstart+=1;
        
        # For free link unbinding
        nSites = self._TotNumSites;
        RatesFreeUnbind = self.getOneUnbindingRates();
        TimesFreeUnbind = self.logrand(nSites)/RatesFreeUnbind;
        # Add to heap
        indexFreeUnbinding = indexstart;
        for iEv in range(len(TimesFreeUnbind)):
            queue.heappush(cle(TimesFreeUnbind[iEv],indexstart),tstep)
            indexstart+=1;
        
        # For CL unbinding (numLinks)
        RatesSecondUnbind = self._koff*np.ones(self._nDoubleBoundLinks)
        TimesSecondUnbind = self.logrand(self._nDoubleBoundLinks)/RatesSecondUnbind
        # Add to heap
        indexSecondUnbind = indexstart;
        for iEv in range(self._nDoubleBoundLinks):
            queue.heappush(cle(TimesSecondUnbind[iEv],indexstart),tstep)
            indexstart+=1;
        
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
            if (event==indexFreeBinding): # end binding, choose random site and bind an end
                #print('Index %d and binding' %event)
                BoundEnd = int(np.random.rand()*self._TotNumSites);
                PlusOrMinusSingleBound=1;
                # Update binding time in queue
                TimeFreeBind = self.logrand()/RateFreeBind+systime;
                queue.heappush(cle(TimeFreeBind,indexFreeBinding),tstep)
            elif (event >= indexFreeUnbinding and event < indexSecondUnbind): 
                #print('Index %d and unbinding end' %event)
                BoundEnd = event-indexFreeUnbinding;
                PlusOrMinusSingleBound=-1;
            elif (event >=indexSecondUnbind): # one end of CL unbinding
                #print('Index %d and unbinding CL end' %event)
                linkNum = event - indexSecondUnbind;
                BoundEnd = self._HeadsOfLinks[linkNum];
                UnbindingEnd = self._TailsOfLinks[linkNum];
                self._PairConnections[BoundEnd,UnbindingEnd]-=1;
                # Unbind it (remove from lists)
                self._HeadsOfLinks[linkNum] = self._HeadsOfLinks[self._nDoubleBoundLinks-1];
                self._TailsOfLinks[linkNum] = self._TailsOfLinks[self._nDoubleBoundLinks-1];
                # Update index of the previous last entry in the heap
                #print('Changing old index %d to %d' %(indexSecondUnbind+self._nDoubleBoundLinks-1, event))
                if (event <  indexSecondUnbind+self._nDoubleBoundLinks-1):
                    queue.updateEventIndex(indexSecondUnbind+self._nDoubleBoundLinks-1,event);
                self._nDoubleBoundLinks-=1;
                # Add a free end at the other site. Always assume the remaining bound end is at the left
                PlusOrMinusSingleBound=1;   
            else: # second end of CL binding
                # The index now determines which pair of sites the CL is binding to
                # In addition, we always assume the bound end is the one at the left 
                #print('Index %d and binding CL end' %event)
                BoundEnd = BindingPairs[event,0];
                UnboundEnd = BindingPairs[event,1];
                self._PairConnections[BoundEnd,UnboundEnd]+=1;
                self._HeadsOfLinks[self._nDoubleBoundLinks] = BoundEnd;
                self._TailsOfLinks[self._nDoubleBoundLinks] = UnboundEnd;
                UnbindTime = self.logrand()/self._koff+systime; # this is the time for UnboundEnd to unbind
                # Add link unbinding event
                queue.heappush(cle(UnbindTime,indexSecondUnbind+self._nDoubleBoundLinks),tstep)
                self._nDoubleBoundLinks+=1;
                PlusOrMinusSingleBound=-1;
            
            # Recompute subset of rates and times
            self._nFreeEnds+= PlusOrMinusSingleBound;
            self._FreeLinkBound[BoundEnd]+= PlusOrMinusSingleBound
            
            # Rates of binding change based on number of bound ends
            RatesSecondBind, TimesSecondBind, updatedLinks = self.updateSecondBindingRate(BoundEnd,BaseCLRates,\
                RatesSecondBind,TimesSecondBind,FirstLinkPerSite,NextLinkByLink,systime);
            
            misses = np.setdiff1d(PairIndices[BindingPairs[:,0]==BoundEnd],updatedLinks);
            if (np.any(BaseCLRates[misses])):
                raise ValueError('Link missed!')

            for iInd in (PairIndices[BindingPairs[:,0]==BoundEnd]):
                queue.heapupdate(iInd,TimesSecondBind[iInd],tstep);
                
            # Update unbinding event at BoundEnd (the end whose state has changed)
            RatesFreeUnbind[BoundEnd] = self._koff*self._FreeLinkBound[BoundEnd];
            TimesFreeUnbind[BoundEnd] = self.logrand()/RatesFreeUnbind[BoundEnd]+systime;
            queue.heapupdate(BoundEnd+indexFreeUnbinding,TimesFreeUnbind[BoundEnd],tstep)         
            #queue.checkIndices(indexSecondUnbind+self._nDoubleBoundLinks-1);
            
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
    
    def logrand(self,N=1):
        return -np.log(1-np.random.rand(N));
        
    def getOneUnbindingRates(self):
        """
        Compute rate for unbinding, which is the number of freely bound links x koff
        """
        return self._koff*self._FreeLinkBound;
      
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
        BothEndRates = np.zeros(2*nPairs);
        BothEndBindingTimes = np.zeros(2*nPairs)
        for iPair in range(2*nPairs):
            BoundEnd = BothEndBindings[iPair,0];  
            BothEndRates[iPair] = BaseRates[iPair]*self._FreeLinkBound[BoundEnd];
            BothEndBindingTimes[iPair] =self.logrand()/BothEndRates[iPair];
        return BothEndRates, BothEndBindingTimes;    
    
    def updateSecondBindingRate(self,BoundEnd,BaseRates,BothEndRates,BothEndBindingTimes,FirstLink,NextLink,systime=0):
        ThisLink = FirstLink[BoundEnd];
        updatedLinks=[];
        while (ThisLink > -1):
            BothEndRates[ThisLink] = BaseRates[ThisLink]*self._FreeLinkBound[BoundEnd];
            eltime = self.logrand()/BothEndRates[ThisLink];
            BothEndBindingTimes[ThisLink] = eltime + systime;
            updatedLinks.append(ThisLink);
            ThisLink = NextLink[ThisLink];
        return BothEndRates, BothEndBindingTimes, updatedLinks;  
    
    
    
        

