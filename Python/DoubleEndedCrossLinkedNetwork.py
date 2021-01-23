from CrossLinkedNetwork import CrossLinkedNetwork
import scipy.sparse as sp
import numpy as np
from CrossLinkingEvent import EventQueue as heap
from EndedCrossLinkedNetwork import EndedCrossLinkedNetwork
import time

verbose = -1;

class DoubleEndedCrossLinkedNetwork(CrossLinkedNetwork):

    """
    This class is a child of CrossLinkedNetwork which implements a network 
    with cross links where each end matters separately. 
    
    This python code implements one specific model,
    but the one actually used is in DoubleEndedCrossLinkedNetworkCPP and that model 
    may be updated in the future and supersedes this one. 
    
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
    
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,kon,koff,konsecond,koffsecond,CLseed,Dom,fibDisc,nThreads=1):
        """
        Constructor
        """
        super().__init__(nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,nThreads);
        self._FreeLinkBound = np.zeros(self._TotNumSites,dtype=np.int64); # number of free-ended links bound to each site
        self._kon = kon*self._ds;
        self._konSecond = konsecond;
        self._koff = koff;
        self._koffSecond = koffsecond;
        self._kDoubleOn = 0; # half the real value because we schedule link binding as separate events
        self._kDoubleOff = 0;
        MaxLinks = max(2*int(konsecond/koffsecond*kon/koff*self._TotNumSites),100)
        self._HeadsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._PrimedShifts = np.zeros((MaxLinks,3));
        allRates = [self._kon,self._konSecond,self._koff,self._koffSecond,self._kDoubleOn,self._kDoubleOff];
        CLBounds = [(1-self._lowercldelta)*self._rl, (1+self._uppercldelta)*self._rl ];
        if (CLseed is None):
            CLseed = int(time.time());
        self._cppNet = EndedCrossLinkedNetwork(self._TotNumSites, allRates, self._DLens,CLBounds,CLseed);
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
        
    def nLinksAllSites(self,fiberCol,Dom,PossiblePairs):
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
               
        nLinksPerPair=[];
        for pair in PossiblePairs:
            if (pair[1] < pair[0]):
                raise ValueError('Your pair list has to have (head) < (tail) and no duplicates')
            nLinksPerPair.append(PairConnections[pair[0],pair[1]]);    
        return nLinksPerPair;
    
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
        if (verbose > 0):
            thist = time.time();
        # Compute the list of neighbors (constant for this step)
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        uniNeighbs = uniNeighbs.astype(np.int64);
        # Filter the list of neighbors to exclude those on the same fiber
        Fibs = uniNeighbs // self._NsitesPerf;
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
       
    ## ======================================== ##
    ##    PRIVATE METHODS (INVOLVED IN UPDATE)
    ## ======================================== ##         
    def syncPythonAndCpp(self):
        self._HeadsOfLinks = self._cppNet.getLinkHeadsOrTails(True);
        self._TailsOfLinks = self._cppNet.getLinkHeadsOrTails(False); 
        self._nDoubleBoundLinks = len(self._HeadsOfLinks);     
        self._FreeLinkBound = self._cppNet.getNBoundEnds();
        self._PrimedShifts = self._cppNet.getLinkShifts();
