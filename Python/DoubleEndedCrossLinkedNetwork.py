from CrossLinkedNetwork import CrossLinkedNetwork
import numpy as np
import random

class DoubleEndedCrossLinkedNetwork(CrossLinkedNetwork):

    """
    This class is a child of CrossLinkedNetwork which implements a network 
    with cross links where each end matters seaprately
    There are 4 rates:
    Rate of one end to attach (self._kon) 
    Rate of the other end to attach (self._konboth)
    Rate of one end to dettach (self._koff)
    Rate of both ends to dettach and the link to float off into oblivion (self._koffBoth) 
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,koffBoth,CLseed,Dom,fibDisc,nThreads=1):
        """
        Constructor
        # In addition to iPts and jPts (lists of completed links), there is now a list of potential sites
        # that have a link attached that could bind to an available other site
        This list is self._FreeLinkBound. All other information is samea as super 
        """
        super().__init__(nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads);
        self._koffBoth = 1;
        self._FreeLinkBound = np.zeros(self._NsitesPerf*self._nFib,dtype=np.int64); # says whether or not a link is bound to a site with a free end
        self._konBoth = 1;
        random.seed(CLseed)
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
    
    def getnLinksBetween2Sites(self,Pair):
        # Find the index
        iIndices = np.argwhere(np.array(self._iPts)==Pair[0]);
        jIndices = np.argwhere(np.array(self._jPts)==Pair[1]);
        iL = np.intersect1d(iIndices,jIndices);
        return len(iL);
    
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
        
        # Form array for possible events
        # 1) Binding of one end  (code +1)
        OneEndBindings, OneEndBindingRates = self.getOneBindingEvents();
        nBindings = len(OneEndBindingRates);
        EventOnes = np.ones(nBindings,dtype=np.int64);
        linkIDOnes = -np.ones(nBindings,dtype=np.int64);
        
        # 2) Unbinding of links with only one end bound (code -1)
        OneEndUnbindings, OneEndUnbindingRates = self.getOneUnbindingEvents();
        nBoundEnds = len(OneEndUnbindingRates);
        EventTwos = -np.ones(nBoundEnds,dtype=np.int64);
        linkIDTwos = -np.ones(nBoundEnds,dtype=np.int64); 
        
        # 3) Unbinding of both ends of a bound link (code -2)
        BothEndUnbindings, BothEndUnbindingRates = self.getBothUnbindingEvents();
        EventThrees = -2*np.ones(self._numLinks,dtype=np.int64);
        linkIDThrees = np.arange(self._numLinks,dtype=np.int64);
        
        # 4) Binding of links (hardest part - requires neighbor search, code +2)
        BothEndBindings, BothEndBindingRates = self.getBothBindingEvents(SpatialDatabase,uniPts,Dom);
        nPairs = len(BothEndBindingRates);
        EventFours = 2*np.ones(nPairs);
        linkIDFours = -np.ones(nPairs,dtype=np.int64);

        # 5) Unbinding of one end of a doubly-bound CL but not both (code -3)
        OneEndofCLUnbinding, OneEndofCLUnbindingRates = self.getOneCLEndUnbindingEvents();
        EventFives = -3*np.ones(self._numLinks,dtype=np.int64);
        linkIDFives = np.arange(self._numLinks,dtype=np.int64);
        CurrentID = self._numLinks;
        
        # Make master lists of pairs of points and events, sample times as well
        allPairs = np.concatenate((OneEndBindings,OneEndUnbindings,BothEndUnbindings,BothEndBindings,OneEndofCLUnbinding));
        allRates = np.concatenate((OneEndBindingRates,OneEndUnbindingRates,BothEndUnbindingRates,BothEndBindingRates,OneEndofCLUnbindingRates));
        allTypes = np.concatenate((EventOnes,EventTwos,EventThrees,EventFours,EventFives));
        allIDs = np.concatenate((linkIDOnes,linkIDTwos,linkIDThrees,linkIDFours,linkIDFives));
        allTimes = -np.log(1-np.random.rand(len(allRates)))/allRates;
        
        # Event-driven algorithm
        event = np.argmin(allTimes);
        systime = allTimes[event];
        while (systime < tstep): 
            eventType = allTypes[event];
            if (eventType == 1): # Binding of one end
                site = allPairs[event,0];   
                allTimes[event] = -np.log(1-np.random.rand())/allRates[event]+systime; # redraw time for next binding
                # Add unbinding event to the list, increment number of bound ends
                allPairs, allRates, allTypes, allTimes, allIDs = self.addOneEnd(site,systime,allPairs, allRates, allTypes, allTimes, allIDs); 
            elif (eventType == -1): # unbinding of one end
                # Remove unbinding event from the list
                allPairs, allRates, allTypes, allTimes, allIDs = self.unBindOneEnd(event,allPairs, allRates, allTypes, allTimes, allIDs);  
            elif (eventType < -1): # unbinding of a link
                # First assume unbinding of both ends
                Pair = allPairs[event,:];
                # Remove unbinding events from list (both one end and both ends)
                allPairs, allRates, allTypes, allTimes, allIDs = self.unBindBothEnds(event,allPairs, allRates, allTypes, allTimes, allIDs);
                if (eventType == -3): # one end still bound
                    # Choose which end is still bound
                    r = int(np.random.rand() < 0.5);
                    stillbound = Pair[0]*r+Pair[1]*(1-r);
                    # Add unbinding event to the list, increment number of bound ends
                    allPairs, allRates, allTypes, allTimes, allIDs = self.addOneEnd(stillbound,systime,allPairs, allRates, allTypes, allTimes, allIDs);
            elif (eventType == 2):
                # Bind a link. Remove unbinding event associated with singly-bound link, and add unbinding events (both single
                # and double) for the link.
                allPairs,allRates,allTypes,allTimes,allIDs, CurrentID = \
                    self.addLink(event,systime,uniPts,Dom,allPairs, allRates, allTypes, allTimes, allIDs,CurrentID);
            event = np.argmin(allTimes);
            systime = allTimes[event];
            
        if (of is not None): 
            self.writeLinks(of,uniPts,Dom);
            self.writePossibles(newLinks);
    
    ## ======================================== ##
    ##    PRIVATE METHODS (INVOLVED IN UPDATE)
    ## ======================================== ##
    def addOneEnd(self,site,systime,allPairs,allRates,allTypes,allTimes,allIDs):
        """
        Add one end at a site. 
        Other inputs are lists of all events, rates, event types, and times to be modified
        (by adding an unbinding event associated with the new bound link) and returned
        """
        self._FreeLinkBound[site]+=1;
        # Add unbinding event and draw new time for another binding
        allPairs = np.concatenate((allPairs,[[site,-1]]));
        allRates = np.concatenate((allRates,[self._koff]));
        allTypes = np.concatenate((allTypes,[-1]));
        allTimes = np.concatenate((allTimes, [-np.log(1-np.random.rand())/self._koff+systime]));
        allIDs = np.concatenate((allIDs,[-1]));
        return allPairs, allRates, allTypes, allTimes, allIDs;
 
    def unBindOneEnd(self,event,allPairs,allRates,allTypes,allTimes,allIDs):
        """
        Unbind one end from a site (a free-ended link) 
        Other inputs are lists of all events, rates, event types, and times to be modified 
        (by deleting the unbinding event) and returned
        """
        site = allPairs[event,0];
        self._FreeLinkBound[site]-=1;
        # Remove unbinding event from list
        allPairs = np.delete(allPairs,event,0);
        allRates = np.delete(allRates,event);
        allTypes = np.delete(allTypes,event);
        allTimes = np.delete(allTimes,event);
        allIDs = np.delete(allIDs,event);
        return allPairs, allRates, allTypes, allTimes, allIDs;
          
    def unBindBothEnds(self,event,allPairs, allRates, allTypes, allTimes, allIDs):
        """
        Unbind both ends of a link from a site. 
        Other inputs are lists of all events, rates, event types, and times to be modified 
        (by deleting the unbinding events) and returned. The unbinding events are the
        unbinding of exactly one end of the link and both ends of the link. They have an ID 
        in that they are associated with a particular copy of the link between iPt and jPt. 
        """
        Pair = allPairs[event,:];
        # Find the index of the link to remove
        iIndices = np.argwhere(np.array(self._iPts,dtype=np.int64)==Pair[0]);
        jIndices = np.argwhere(np.array(self._jPts,dtype=np.int64)==Pair[1]);
        iL = np.intersect1d(iIndices,jIndices);
        super().removeLink(iL[0]);
        # Remove exactly 2 unbinding events from list (have to be the right ones)
        allInds = np.arange(len(allPairs[:,0])); 
        delInds = allInds[allIDs==allIDs[event]]; # match the ID
        #print('Deleting indices')
        #print(delInds)
        #print(allPairs[delInds,:])
        allPairs = np.delete(allPairs,delInds,0);
        allRates = np.delete(allRates,delInds);
        allTypes = np.delete(allTypes,delInds);
        allTimes = np.delete(allTimes,delInds);
        allIDs = np.delete(allIDs,delInds);
        return allPairs, allRates, allTypes, allTimes, allIDs;
    
    def addLink(self,event,systime,uniPts,Dom,allPairs, allRates, allTypes, allTimes, allIDs,CurrentID):
        """
        Add a new link that connects two sites. 
        Other inputs are lists of all events, rates, event types, and times to be modified.
        We pick at random (or by necessity) whether site1 or site2 is the site where one end is 
        already bound. We remove the unbinding event associated with that end and then bind the link,
        adding an unbinding event for one or two ends of it to become unbound. We then associate those
        unbinding events with a new link ID, which we increment. 
        Of course, if there are no availabel free links, we just draw a new time and return. 
        """    
        site1 = allPairs[event,0];        
        site2 = allPairs[event,1];
        # Check that there is a link at one of the sites available for binding
        freeLinkLoc = -1;
        #print('Sites %d and %d' %(site1,site2))
        #print('Free links bound %d and %d' %(self._FreeLinkBound[site1], self._FreeLinkBound[site2]))
        if (self._FreeLinkBound[site1] > 0):
            freeLinkLoc = site1;
            if (self._FreeLinkBound[site2] > 0): # if both have free link, choose one at random
                pickem = int(np.random.rand() < 0.5);
                freeLinkLoc = pickem*site1+(1-pickem)*site2;
        elif (self._FreeLinkBound[site2] > 0):
            freeLinkLoc = site2;
        # Draw new time for another binding 
        allTimes[event] = -np.log(1-np.random.rand())/allRates[event]+systime;
        if (freeLinkLoc > -1): # binding can occur
            super().addLink(site1,site2,uniPts,Dom);
            # Remove unbinding event associated with freeLinkLoc
            allInds = np.arange(len(allPairs[:,0]));
            delInds = allInds[np.logical_and(allPairs[:,0]==freeLinkLoc,allTypes==-1)];
            delInd = random.choice(delInds);
            allPairs, allRates, allTypes, allTimes, allIDs = self.unBindOneEnd(delInd,allPairs,allRates,allTypes,allTimes,allIDs)
            # Add 2 unbinding events associated with new link
            allPairs = np.concatenate((allPairs,np.array([[site1,site2],[site1,site2]])));
            offRates = np.array([self._koffBoth,self._koff])
            allRates = np.concatenate((allRates,offRates));
            allTypes = np.concatenate((allTypes,np.array([-2,-3])));
            allTimes = np.concatenate((allTimes, -np.log(1-np.random.rand(2))/offRates+systime));
            allIDs = np.concatenate((allIDs,[CurrentID, CurrentID]))
            CurrentID+=1;
        return allPairs,allRates,allTypes,allTimes,allIDs, CurrentID;              
               
   
    def calcKonOne(self,iPt,jPt,uniPts,Dom):
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
            return self._konBoth; # binds with probability 1 at every time step
        return 0;  
     
    def getOneBindingEvents(self):
        """
        Generate list of events for binding of a free link to a site
        """
        OneEndBindings = np.zeros((self._NsitesPerf*self._nFib,2),dtype=np.int64)-1;
        OneEndBindings[:,0] = np.arange(self._NsitesPerf*self._nFib,dtype=np.int64);
        OneEndBindingRates = self._kon*np.ones(self._NsitesPerf*self._nFib);
        return OneEndBindings,OneEndBindingRates;
    
    def getOneUnbindingEvents(self):
        """
        Generate list of events for unbinding of a singly-bound link from a site
        """
        nBoundEnds = np.sum(self._FreeLinkBound)
        OneEndUnbindings = np.zeros((nBoundEnds,2),dtype=np.int64)-1;
        OneEndUnbindings[:,0] = np.repeat(np.arange(self._NsitesPerf*self._nFib,dtype=np.int64),self._FreeLinkBound);
        OneEndUnbindingRates = self._koff*np.ones(nBoundEnds);
        return OneEndUnbindings, OneEndUnbindingRates;
    
    def getBothUnbindingEvents(self):
        """
        Generate list of events for unbinding of a doubly-bound link from both sites
        """
        BothEndUnbindings = np.zeros((self._numLinks,2),dtype=np.int64);
        BothEndUnbindings[:,0] = np.array(self._iPts,dtype=np.int64);
        BothEndUnbindings[:,1] = np.array(self._jPts,dtype=np.int64);
        BothEndUnbindingRates = self._koffBoth*np.ones(self._numLinks);
        return  BothEndUnbindings, BothEndUnbindingRates;     
   
    def getBothBindingEvents(self,SpatialDatabase,uniPts,Dom):
        """
        Generate list of events for binding of a doubly-bound link to both sites
        """
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = newLinks.shape;
        BothEndBindings = newLinks.astype(np.int64);
        BothEndBindingRates = np.zeros(nPairs);
        for iPair in range(nPairs):            
            BothEndBindingRates[iPair] = self.calcKonOne(BothEndBindings[iPair,0],BothEndBindings[iPair,1],uniPts,Dom);
        return BothEndBindings, BothEndBindingRates;
    
    def getOneCLEndUnbindingEvents(self): 
        """
        Generate list of events for unbinding of one end of a doubly-bound link
        """  
        OneEndofCLUnbinding = np.zeros((self._numLinks,2),dtype=np.int64);
        OneEndofCLUnbinding[:,0] = np.array(self._iPts,dtype=np.int64);
        OneEndofCLUnbinding[:,1] = np.array(self._jPts,dtype=np.int64);
        OneEndofCLUnbindingRates = self._koff*np.ones(self._numLinks);
        return OneEndofCLUnbinding, OneEndofCLUnbindingRates;
    
    
        

