from CrossLinkedNetwork import CrossLinkedNetwork, CrossLinkedSpeciesNetwork
import numpy as np
from EndedCrossLinkedNetwork import EndedCrossLinkedNetwork
import time
from warnings import warn

verbose = -1;

class DoubleEndedCrossLinkedNetwork(CrossLinkedNetwork):

    """
    This class is a child of CrossLinkedNetwork which implements a network 
    with cross links where each end matters separately. 
       
    There are 4 reactions (the even ones are the reverse of the odds):
    
    1) Binding of a floating link to one site (rate _kon)
    2) Unbinding of a link that is bound to one site to become free (reverse of 1, rate _koff)
    3) Binding of a singly-bound link to another site to make a doubly-bound CL (rate _konSecond)
    4) Unbinding of a double bound link in one site to make a single-bound CL (rate _koffSecond)
    
    For more information on the reactions and their rates, see Section 9.1.1 of Maxian's PhD
    thesis. We implement reaction 3, which is the most important because it controls the forces
    exerted on the fibers, by performing a neighbor search to find all pairs of uniform sites 
    a certain distance apart. Note also that we do not explicitly keep track of unbound cross linkers.
    We instead assume that there are always enough CLs if an event is supposed to happen.
    
    We use an event driven algorithm to simulate a time step. To efficiently organize the events, 
    we use a heap queue data structure which we borrow from the simulation package SRBD 
    (https://github.com/stochasticHydroTools/SRBD). The heap queue is implemented in the fortran
    code /Fortran/MinHeapModule.f90 in this repository.
    
    """
    
    ## =============================== ##
    ##    METHODS FOR INITIALIZATION
    ## =============================== ##
    def __init__(self,nFib,N,Nunisites,Lfib,kCL,rl,kon,koff,konsecond,koffsecond,CLseed,Dom,fibDisc,\
        UnloadedVel=0,StallForce=0,kT=0,nThreads=1,bindingSiteWidth=0,smoothForce=False):
        """
        Parameters
        ----------
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
        kon: double
            Rate (units 1/(length x time)) at which a single end of a CL binds to a site
        koff: double
            Rate (units 1/time) at which a singly-bound CL comes off from a site
        konsecond: double
            Rate (units 1/(length x time)) at which the second end of a singly-bound CL
            binds to a site (this is modified by an Arrhenuis factor later -- see the method
            updateNetwork)
        koffsecond: double
            Rate (units 1/time) at which one end of a doubly-bound CL comes off, leaving
            a singly-bound CL
        CLseed: int
            The seed for cross linking calculations
        Dom: Domain object
            Periodic domain on which the calculation is being carried out
        fibDisc: FibCollocationDiscretization object
            The discretization of each fiber's centerline
        UnloadedVel: double
            If these objects are motors, this is the unloaded gliding speed
        StallForce: double
            If these objects are motors, this is the stall force
        kT: double
            Thermal energy. If zero, the binding of a second end occurs with rate konsecond (no 
            Arrhenius factor). If nonzero, then the rate is modified (see updateNetwork method for 
            details)
        smoothForce: boolean, defaults to true
            Whether to smooth out the forcing or use an actual spring between
            the uniformly spaced points. See the method CLForce (CrossLinkedNetwork.py)
            for details on what formulas this switches between
        nThreads: int
            Number of threads for openMP calculation of CL forces and stress
        bindingSiteWidth: double, optional (defaults to 0)
            If a binding site in biology has a width $w$, then the number of links that
            can bind to one of our uniformly-spaced binding sites (spaced $\\Delta s_u$)
            is given by $N = \\lceil \\Delta s_u/w \\rceil$. This in practice becomes 
            a cap on the number of links per site.
        """
        super().__init__(nFib,N,Nunisites,Lfib,kCL,rl,Dom,fibDisc,smoothForce,nThreads);
        self._FreeLinkBound = np.zeros(self._TotNumSites,dtype=np.int64); # number of free-ended links bound to each site
        self._kon = kon*self._ds;
        self._konSecond = konsecond*self._ds;
        self._koff = koff;
        self._koffSecond = koffsecond;
        if (koffsecond > 1e-3 and koff > 1e-3):
            MaxLinks = max(2*int(konsecond/koffsecond*kon/koff*self._TotNumSites),100)
        else:
            MaxLinks = 10;
        self._HeadsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._TailsOfLinks = np.zeros(MaxLinks,dtype=np.int64);
        self._PrimedShifts = np.zeros((MaxLinks,3));
        allRates = [self._kon,self._konSecond,self._koff,self._koffSecond];
        if (kT > 0):
            self._deltaL = 2*np.sqrt(kT/self._kCL); # Strain-dependent rate, modify self._deltaL to be 2 or 1/2 rest length
        CLBounds = [self._rl-self._deltaL,self._rl+self._deltaL];
        if (self._deltaL > self._rl):
            CLBounds = [0,self._rl+self._deltaL];    
        print(CLBounds)
        if (CLseed is None):
            CLseed = int(time.time());
        if (bindingSiteWidth==0):
            maxPerSite = 2**10;
        else:
            maxPerSite = int(np.ceil(self._ds/bindingSiteWidth)); # 20 nm binding site
        print('Max per site %d' %maxPerSite)
        self._cppNet = EndedCrossLinkedNetwork(self._TotNumSites, maxPerSite,allRates, self._DLens,CLBounds,kT,self._rl, self._kCL,CLseed);
        self._isMotor = False;
        if (UnloadedVel>0): # set up motors
            self._isMotor = True;
            self._cppNet.SetMotorParams(UnloadedVel/self._ds, StallForce, self._NsitesPerf) 
        
    ## =============================== ##
    ##     PUBLIC METHODS
    ## =============================== ##
    def getnBoundEnds(self):
        return self._FreeLinkBound;
    
    def sitesPerFib(self,iFib):
        """
        Parameters
        ----------
        iFib: int
            Fiber index
        
        Returns
        -------
        int array
            The indices of the uniform binding sites that correspond to the fiber iFib
        """
        return np.arange(iFib*self._NsitesPerf,(iFib+1)*self._NsitesPerf,dtype=np.int64);
       
    def updateNetwork(self,fiberCol,Dom,tstep):
        """
        Update the network using Kinetic Monte Carlo. The full procedure we use is documented 
        in Section 9.1.1 of Maxian's PhD thsis. Briefly, we consider six possible reactions, with the 
        following rates in units 1/time:
        
        1) Binding of a floating link to one site (rate _kon $\\Delta s_u$)
        2) Unbinding of a link that is bound to one site to become free (reverse of 1, rate _koff)
        3) Binding of a singly-bound link to another site to make a doubly-bound CL. The rate of this 
           reaction, assuming $k_B T > 0$ is given by
           $$k_{on,s} = k_{on,s}^0 \\exp{\\left(\\frac{-K_c (r-\\ell)^2}{k_B T}\\right)} \\Delta s_u$$
           If $k_B T = 0$, then the Arrhenius factor is ommitted. If $k_B T > 0$, we assume possible
           connections can form if two sites are separated by $2\\sqrt{k_B T/K}$, where $K$ is the CL 
           stiffness (two standard deviations of the Gaussian above). If $k_B T =0$ and the binding
           probability is uniform, we stick to one standard deviation.
        4) Unbinding of a double bound link in one site to make a single-bound CL (rate _koffSecond)
        
        Once the rates are computed, the time at which an event occurs is taken from an exponential
        distribution via sampling a time $t=-\\ln(1-u)/k$, where $k$ is the event rate and $u$ is 
        a random draw from $U(0,1)$. These times are organized into a heap queue which efficiently
        sorts them.
        
        Parameters
        ----------
        fiberCol: fiberCollection object 
            The fibers we are simulating
        Dom: Domain object
            The periodic (sheared?) domain we are simulating
        tstep: double
            Time step we simulate over
        
        Returns
        -------
        Null
            Nothing, but updates the state of the network internally. The actual update is done in C++.
            On the python side, we get the neighbors and filter them so that fibers cannot link to
            themselves. Then we pass the list of potential neighbors to C++ which updates its own chain.
            The Python side is then synced with the C++ side.
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
        uniNeighbs = SpatialDatabase.selfNeighborList(self._rl+self._deltaL,rcutLow=self._rl-self._deltaL,numperfiber=self._NsitesPerf);
        uniNeighbs = uniNeighbs.astype(np.int64);
        # Filter the list of neighbors to exclude those on the same fiber
        Fibs = self.mapSiteToFiber(uniNeighbs);
        delInds = np.arange(len(Fibs[:,0]));
        newLinks = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
        # Compute the prime shifts between the links
        PrimedShiftsProp = Dom.ComputePrimeShifts(newLinks,uniPts);
        # Compute real proposed and old shifts
        RealShiftsProp = Dom.unprimecoords(PrimedShiftsProp);
        RealShifts = Dom.unprimecoords(self._PrimedShifts[:self._nDoubleBoundLinks,:]);
        if (verbose > 0):
            print('Neighbor search and organize time %f ' %(time.time()-thist))  
            thist = time.time();
            
        self._cppNet.updateNetwork(tstep,newLinks,uniPts,PrimedShiftsProp,RealShiftsProp,RealShifts);
        if (verbose > 0):
            print('Update network time %f ' %(time.time()-thist))
        
        # Keep C++ and Python up to date (for force calculations)
        self.syncPythonAndCpp()
        
        # Walk the motors
        if (not self._isMotor):
            return;
        tauPts = fiberCol.getXs();
        uniTau = fiberCol.getUniformTau(tauPts);
        RealShifts = Dom.unprimecoords(self._PrimedShifts[:self._nDoubleBoundLinks,:]);
        self._cppNet.WalkLinks(tstep,uniPts,uniTau,RealShifts)
        self.syncPythonAndCpp()
    
    def setLinks(self,iPts,jPts,Shifts,FreelyBound=None):
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
        FreelyBound: array, optional
            This gives the number of singly-bound links bound to each of the uniform
            sites. If not supplied, it will set the number to zero for all sites.
        """
        super().setLinks(iPts,jPts,Shifts);
        # Update C++ class
        if (FreelyBound is None):
            warn('You did not set the number of free bound links - will all be set to')
            if (self._koff>1e-3):
                FreelyBound = int(self._kon/self._koff)*np.ones(self._TotNumSites);
                print((int(self._kon/self._koff)))
            else: 
                FreelyBound = np.zeros(self._TotNumSites);
                print(' zero')
        self._FreeLinkBound = FreelyBound;
        self._cppNet.setLinks(self._HeadsOfLinks, self._TailsOfLinks, self._PrimedShifts, self._FreeLinkBound) 
        self.syncPythonAndCpp();    
    
    def setLinksFromFile(self,FileName,FreelyBound=None):
        """
        Set the links from a file name. The file has a list of iPts, 
        jPts (two ends of the links), and shift in zero strain coordinates.
        
        Parameters
        -----------
        FileName: string
            The name of the file
        FreelyBound: array, optional
            This gives the number of singly-bound links bound to each of the uniform
            sites. If not supplied, it will set the number to zero for all sites.
        """
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
        """
        Removes all links (both double and singly bound) connected to a fiber
        
        Parameters
        ----------
        fibNums: list
            List of fiber numbers to remove links from
        """
        if (len(fibNums)==0):
            return;
        pyHeads = self._HeadsOfLinks.copy();
        pyTails = self._TailsOfLinks.copy();
        pyShifts = self._PrimedShifts.copy();
        pyFB = self._FreeLinkBound.copy();
        for iFib in fibNums:
            print('Deleting all links from fiber %d' %iFib)
            sites = self.sitesPerFib(iFib)
            self._cppNet.deleteLinksFromSites(sites);
            if (False):
                pyFB[sites]=0;
                for site in sites:
                    inds = np.where(np.logical_or(pyHeads==site,pyTails==site))[0];
                    pyHeads = np.delete(pyHeads,inds);
                    pyTails = np.delete(pyTails,inds);
                    pyShifts = np.delete(pyShifts,inds,axis=0);
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
        """
        Synchronizes the python and C++ codes by copying the C++ over to python
        """
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
   
