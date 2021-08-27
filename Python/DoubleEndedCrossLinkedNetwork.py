from CrossLinkedNetwork import CrossLinkedNetwork, CrossLinkedSpeciesNetwork
import numpy as np
from EndedCrossLinkedNetwork import EndedCrossLinkedNetwork
import time
from warnings import warn

verbose = -1;

# Documentation last updated: 03/09/2021

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
    
    This is combined with the C++ file
    cppmodules/EndedCrossLinkers.cpp which uses the fortran code
    ../Fortran/MinHeapModule.f90
    
    There is a brief child class below for fiber species.
    
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,kon,koff,konsecond,koffsecond,CLseed,Dom,fibDisc,kT=0,nThreads=1,bindingSiteWidth=0):
        """
        Constructor
        """
        super().__init__(nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,nThreads);
        self._FreeLinkBound = np.zeros(self._TotNumSites,dtype=np.int64); # number of free-ended links bound to each site
        self._kon = kon*self._ds;
        self._konSecond = konsecond*self._ds;
        self._koff = koff;
        self._koffSecond = koffsecond;
        self._kDoubleOn = 0; # half the real value because we schedule link binding as separate events
        self._kDoubleOff = 0;
        MaxLinks = max(2*int(konsecond/koffsecond*kon/koff*self._TotNumSites),100)
        self._HeadsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._PrimedShifts = np.zeros((MaxLinks,3));
        allRates = [self._kon,self._konSecond,self._koff,self._koffSecond,self._kDoubleOn,self._kDoubleOff];
        if (kT > 0):
            self._deltaL = min(2*np.sqrt(kT/self._kCL),self._rl); # Strain-dependent rate, modify self._deltaL to be 2 or 1/2 rest length
        CLBounds = [self._rl-self._deltaL,self._rl+self._deltaL];
        print(CLBounds)
        if (CLseed is None):
            CLseed = int(time.time());
        if (bindingSiteWidth==0):
            maxPerSite = 2**10;
        else:
            maxPerSite = int(np.ceil(self._ds/bindingSiteWidth)); # 20 nm binding site
        print('Max per site %d' %maxPerSite)
        self._cppNet = EndedCrossLinkedNetwork(self._TotNumSites, maxPerSite,allRates, self._DLens,CLBounds,kT,self._rl, self._kCL,CLseed);
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
    
    def sitesPerFib(self,iFib):
        """
        Return the indices of the uniform binding sites that correspond to the 
        fiber iFib
        """
        return np.arange(iFib*self._NsitesPerf,(iFib+1)*self._NsitesPerf,dtype=np.int64);
       
    def updateNetwork(self,fiberCol,Dom,tstep):
        """
        Update the network using Kinetic MC.
        Inputs: fiberCol = fiberCollection object of the fibers (can also be species collection),
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        On the python side, we get the neighbors and filter them so that fibers cannot link to 
        themselves. Then we pass that to C++ which updates its own chain. 
        """
        # Obtain Chebyshev points and uniform points, and get
        # the neighbors of the uniform points
        if (verbose > 0):
            thist = time.time();
        # Compute the list of neighbors (constant for this step)
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        # This will return a different order every time -> lose reproducibility
        try:
            uniNeighbs = SpatialDatabase.selfNeighborList(self._rl+self._deltaL,self._NsitesPerf);
        except: # If N sites per f is not defined, it will just do 1 site per fiber and then the code below will process
            uniNeighbs = SpatialDatabase.selfNeighborList(self._rl+self._deltaL,1);
        uniNeighbs = uniNeighbs.astype(np.int64);
        # Filter the list of neighbors to exclude those on the same fiber
        Fibs = self.mapSiteToFiber(uniNeighbs);
        delInds = np.arange(len(Fibs[:,0]));
        newLinks = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
        if (verbose > 0):
            print('Neighbor search and organize time %f ' %(time.time()-thist))  
            thist = time.time();
        self._cppNet.updateNetwork(tstep,newLinks,uniPts,Dom.getg());
        if (verbose > 0):
            print('Update network time %f ' %(time.time()-thist))
        
        # Keep C++ and Python up to date (for force calculations)
        self.syncPythonAndCpp()
    
    def setLinks(self,iPts,jPts,Shifts,FreelyBound=None):
        warn('Set links in EndedCLNet not tested')
        super().setLinks(iPts,jPts,Shifts);
        # Update C++ class
        if (FreelyBound is None):
            FreelyBound = int(self._kon/self._koff)*np.ones(self._TotNumSites);
            warn('You did not set the number of free bound links - will all be set to %d' %int(self._kon/self._koff))
        self._FreeLinkBound = FreelyBound;
        self._cppNet.setLinks(self._HeadsOfLinks, self._TailsOfLinks, self._PrimedShifts, self._FreeLinkBound) 
        self.syncPythonAndCpp();    
    
    def setLinksFromFile(self,FileName,FreelyBound=None):
        super().setLinksFromFile(FileName);
        # Update C++ class
        if (FreelyBound is None):
            FreelyBound = int(self._kon/self._koff)*np.ones(self._TotNumSites);
            warn('You did not set the number of free bound links - will all be set to %d' %int(self._kon/self._koff))
        else:
            FreelyBound = np.loadtxt(FreelyBound);
        self._FreeLinkBound = FreelyBound;
        self._cppNet.setLinks(self._HeadsOfLinks, self._TailsOfLinks, self._PrimedShifts, self._FreeLinkBound) 
        self.syncPythonAndCpp();
   
    def deleteLinksFromFibers(self,fibNums):
        if (len(fibNums)==0):
            return;
        pyHeads = self._HeadsOfLinks.copy();
        pyTails = self._TailsOfLinks.copy();
        pyShifts = self._PrimedShifts.copy();
        pyFB = self._FreeLinkBound.copy();
        for iFib in fibNums:
            print('Deleting all links from fiber %d' %iFib)
            sites = self.sitesPerFib(iFib)
            pyFB[sites]=0;
            for site in sites:
                inds = np.where(np.logical_or(pyHeads==site,pyTails==site))[0];
                pyHeads = np.delete(pyHeads,inds);
                pyTails = np.delete(pyTails,inds);
                pyShifts = np.delete(pyShifts,inds,axis=0);
            self._cppNet.deleteLinksFromSites(sites);
        self.syncPythonAndCpp();
        #print('In C++ / python check for number of links %d' %self._nDoubleBoundLinks)
        #print(np.amax(np.abs(pyFB-self._FreeLinkBound)))
        #print(np.amax(np.abs(np.sort(self._HeadsOfLinks)-np.sort(pyHeads))))
        #print(np.amax(np.abs(np.sort(self._TailsOfLinks)-np.sort(pyTails))))
        #print(np.amax(np.abs(np.sort(self._PrimedShifts,axis=None)-np.sort(pyShifts,axis=None))))

    ## ======================================== ##
    ##    PRIVATE METHODS (INVOLVED IN UPDATE)
    ## ======================================== ##         
    def syncPythonAndCpp(self):
        self._HeadsOfLinks = self._cppNet.getLinkHeadsOrTails(True);
        self._TailsOfLinks = self._cppNet.getLinkHeadsOrTails(False); 
        self._nDoubleBoundLinks = len(self._HeadsOfLinks);     
        self._FreeLinkBound = self._cppNet.getNBoundEnds();
        self._PrimedShifts = self._cppNet.getLinkShifts();

class DoubleEndedCrossLinkedSpeciesNetwork(CrossLinkedSpeciesNetwork,DoubleEndedCrossLinkedNetwork):
    
    """
    CL Network in the case of fiber species. The only thing that changes is the instantiation of the super
    object (it inherits from CrossLinkedSpecies and not CrossLinkedNetwork), and the mapping that gives you 
    the index of sites on each fiber
    """

    def __init__(self,FiberSpeciesCollection,Kspring,rl,kon,koff,konsecond,koffsecond,CLseed,Dom,kT=0,nThreads=1):
        super().__init__(FiberSpeciesCollection,Kspring,rl,Dom,nThreads)
                
        self._kon = kon*self._ds;
        self._konSecond = konsecond*self._ds;
        self._koff = koff;
        self._koffSecond = koffsecond;
        self._kDoubleOn = 0; # half the real value because we schedule link binding as separate events
        self._kDoubleOff = 0;
        MaxLinks = max(2*int(konsecond/koffsecond*kon/koff*self._TotNumSites),100)
        self._HeadsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._PrimedShifts = np.zeros((MaxLinks,3));
        allRates = [self._kon,self._konSecond,self._koff,self._koffSecond,self._kDoubleOn,self._kDoubleOff];
        if (kT > 0): # modify self._deltaL to be 2 or 1/2 rest length
            self._deltaL = min(2*np.sqrt(kT/self._kCL),self._rl); # Strain-dependent rate
        CLBounds = [self._rl-self._deltaL,self._rl+self._deltaL];
        
        print(allRates)
        if (CLseed is None):
            CLseed = int(time.time());
        self._cppNet = EndedCrossLinkedNetwork(self._TotNumSites, allRates, self._DLens,CLBounds,kT,self._rl, self._kCL, CLseed);
        
    def sitesPerFib(self,iFib):
        return np.where(self._SiteToFiberMap==iFib)[0];
   
