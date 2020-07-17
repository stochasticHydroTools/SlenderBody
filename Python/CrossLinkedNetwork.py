import numpy as np
import CrossLinking as clinks
from warnings import warn 

class CrossLinkedNetwork(object):

    """
    Object with all the cross-linking information. 
    The abstraction is just a class with some cross linking binding sites 
    in it, the children implement the update methods. 
    Currently, KMCCrossLinkedNetwork implements kinetic Monte Carlo
    to update the network
    """
    
    def __init__(self,nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads=1):
        """
        Initialize the CrossLinkedNetwork. 
        Input variables: nFib = number of fibers, N = number of points per
        fiber, Nunisites = number of uniformly placed CL binding sites per fiber
        Lfib = length of each fiber, nCL = number of cross linkers, 
        kCL = cross linking spring constant, rl = rest length of the CLs, 
        kon = constant rate of binding (constant if inside a CL rest length), 
        koff = constant rate of unbinding (pre-factor in front of the rate)
        CLseed = seed for the random number generator that binds/unbinds the CLs, 
        Dom = domain object, fibDisc = fiber collocation discretization, Threads = number 
        of threads for openMP calculation of CL forces and stress
        """
        self._Npf = N; 
        self._NsitesPerf = Nunisites;
        self._Lfib = Lfib;
        self._nFib = nFib;
        self._nCL = nCL;
        self._kCL = kCL;
        self._rl = rl;
        self._kon = kon;
        self._koff = koff;
        self._sigma = 0.05*self._Lfib;  # standard deviation of Gaussian for smooth points
        self._numLinks = 0;             # number of links taken
        self._iPts = [];                # one in pairs of the uniform points (LIST)
        self._jPts = [];                # other in pairs of the uniform points (LIST)
        self._Shifts = [];              # periodic shifts for each link (in the DEFORMED DOMAIN)
        self._added = np.zeros(self._NsitesPerf*self._nFib,dtype=int); # whether there's a link at that point
        self._nThreads = nThreads;
        
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length of the fiber.
        if (self._Npf < 20):
            self._sigma = 0.10*self._Lfib;
        elif (self._Npf < 24):
            self._sigma = 0.085*self._Lfib;
        elif (self._Npf < 32):
            self._sigma = 0.07*self._Lfib;
        # Cutoff distance to form the CLs (temporarily the same as the rest length)
        self._clcut = self._rl;
        # Uniform binding locations
        self._su = np.linspace(0.0,self._Lfib,self._NsitesPerf,endpoint=True);
        self._wCheb = fibDisc.getw();
        # Seed C++ random number generator and initialize the C++ variables for cross linking 
        clinks.seedRandomCLs(CLseed);
        clinks.initDynamicCLVariables(self._kon,self._koff,self._clcut,Dom.getPeriodicLens());
        clinks.initCLForcingVariables(self._su,fibDisc.gets(),fibDisc.getw(),self._sigma,self._kCL,self._rl,self._nCL);
    
    ## ==============================================
    ## PUBLIC METHODS CALLED BY TEMPORAL INTEGRATOR
    ## ==============================================
    def updateNetwork(self,fiberCol,Dom,tstep,of=None):
        """
        Method to update the network. In the abstract parent class, 
        this method does a forward Euler update of the links. 
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt), of = output file to write links to 
        """
        if (self._nCL==0):
            return;
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        nLinks = self._numLinks;
        uniNeighbs = SpatialDatabase.selfNeighborList(self._clcut);
        
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = newLinks.shape;
        
        # Calculate (un)binding rates for each event
        rateUnbind = self.calcKoff(uniPts,Dom);
        rateBind = clinks.calcKons(newLinks,uniPts,Dom.getg()); 
            
        # Make lists of pairs of points and events
        totiPts = np.concatenate((np.array(self._iPts,dtype=int),newLinks[:,0]));
        totjPts = np.concatenate((np.array(self._jPts,dtype=int),newLinks[:,1]));
        nowBound = np.concatenate((np.ones(len(self._iPts),dtype=int),np.zeros(nPairs,dtype=int)));
        allRates = np.concatenate((rateUnbind,rateBind));
        
        # Unique rows to get unique events 
        c = np.concatenate(([totiPts],[totjPts]),axis=0);
        _, uinds = np.unique(c.T,axis=0,return_index=True);
        uinds = np.sort(uinds);
        totiPts = totiPts[uinds];
        totjPts = totjPts[uinds];
        nowBound = nowBound[uinds];
        allRates = allRates[uinds];
        removeinds = [];
        addinds = [];
        linkorder = np.random.permutation(np.arange(len(totiPts))); # go through the links in this order
        added = self._added.copy();
        for iL in linkorder: # loop over links and do forward Euler update 
            change = np.random.rand() < tstep*allRates[iL]; # change the status of the link
            iPt = totiPts[iL];
            jPt = totjPts[iL];
            if (change): 
                if (nowBound[iL]==0 and nLinks < self._nCL and added[iPt]==0 and added[jPt]==0):# wants to bind and can
                    addinds.append(iL);
                    nLinks+=1;
                    added[iPt]=1;
                    added[jPt]=1;
                elif (nowBound[iL]==1): # wants to unbind
                    removeinds.append(iL);
                    nLinks-=1;
                    added[iPt]=0;
                    added[jPt]=0;
        
        # Network update 
        breakLinks = -np.sort(-np.array(removeinds,dtype=int)); # links to break in descending order
        #print('Links to break (in descending order)');
        #print(breakLinks);
        # Break some links
        for iL in breakLinks:
            self.removeLink(iL);
        for newL in addinds: # make remaining links
            self.addLink(totiPts[newL],totjPts[newL],uniPts,Dom);
        if (of is not None): # write changes to file 
            self.writeLinks(of,uniPts,Dom);
    
    def calcLinkStrains(self,uniPoints,Dom):
        """
        Method to compute the strain in each link
        Inputs: uniPoints = uniform points on the fibers to which the 
        links are bound, Dom = domain object for periodic shifts 
        Return: an array of nLinks sisze that has the signed strain ||link length|| - rl
        for each link 
        """
        if (self._numLinks==0):
            return;
        linkStrains = np.zeros(self._numLinks);
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        for iLink in range(self._numLinks):
            ds = uniPoints[self._iPts[iLink],:]-uniPoints[self._jPts[iLink],:]-shifts[iLink,:];
            linkStrains[iLink]=np.linalg.norm(ds)-self._rl;
        return linkStrains;
        
            
    def CLForce(self,uniPoints,chebPoints,Dom):
        """
        Compute the total cross linking force.
        Inputs: uniPoints = uniform points for all fibers, chebPoints = Chebyshev points 
        on all fibers, Dom = Domain object. 
        Outputs: nPts*3 one-dimensional array of the forces at the 
        Chebyshev points due to the CLs
        """
        if (self._numLinks==0): # if no CLs don't waste time
            return np.zeros(3*self._Npf*self._nFib);
        # Call the C++ function to compute forces
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        Clforces = clinks.calcCLForces(self._iPts, self._jPts,shifts,uniPoints,chebPoints,self._nThreads);
        return Clforces;
    
    def CLStress(self,fibCollection,chebPts,Dom):
        """
        Compute the cross-linker contribution to the stress.
        Inputs: fibCollection = fiberCollection object to compute
        the stress on, fibDisc = fiber Discretization object describing
        each fiber, Dom = Domain (needed to compute the volume)
        """
        stress = np.zeros((3,3));
        if (self._numLinks==0):
            return stress;
        uniPts = fibCollection.getUniformPoints(chebPts);
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        if False: # python
            for iLink in range(self._numLinks): # loop over links
                iFib = self._iPts[iLink] // self._NsitesPerf;
                jFib = self._jPts[iLink] // self._NsitesPerf;
                iInds = fibCollection.getRowInds(iFib);
                jInds = fibCollection.getRowInds(jFib);
                # Force on each link 
                Clforces = np.reshape(clinks.calcCLForces([self._iPts[iLink]], [self._jPts[iLink]], shifts[iLink,:],\
                    uniPts,chebPts),(chebPts.shape));
                for iPt in range(self._Npf): # increment stress: use minus since force on fluid = - force on fibers
                    stress-=self._wCheb[iPt]*np.outer(chebPts[iInds[iPt],:],Clforces[iInds[iPt],:]);
                    stress-=self._wCheb[iPt]*np.outer(chebPts[jInds[iPt],:]+shifts[iLink,:],Clforces[jInds[iPt],:]);
        # C++ function call 
        stress = clinks.calcCLStress(self._iPts,self._jPts,shifts,uniPts,chebPts,self._nThreads);
        #print(stressCpp-stress)
        stress/=np.prod(Dom.getLens());
        return stress;
        

    ## ==============================================
    ##     PRIVATE METHODS ONLY ACCESSED IN CLASS
    ## ==============================================
    def addLink(self,iPt,jPt,uniPts,Dom):
        """
        Add a link to the list. 
        Update the _added arrays and append the link locations
        to self._iPts and self._jPts, and self._shifts. The shifts
        are distances in the variables (x',y',z'). 
        """
        rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
        shift = uniPts[iPt,:]-uniPts[jPt,:] - rvec;
        #print('Making link between %d and %d with distance %f' %(iPt, jPt, np.linalg.norm(rvec)));
        # For now allowing multiple links to bind to a site 
        #if (self._added[iPt]==1 or self._added[jPt]==1):
        #    raise ValueError('Bug - trying to make link in already occupied site');
        self._added[iPt]=1;
        self._added[jPt]=1;
        self._iPts.append(iPt);
        self._jPts.append(jPt);
        primeshift = Dom.primecoords(shift);
        self._Shifts.append(primeshift);
        self._numLinks+=1;
    
    def removeLink(self,iL):
        """
        Remove a link from the list 
        Update self._added array and remove the elements from
        self._iPts and self._jPts
        """
        #print('Breaking link between %d and %d' %(self._iPts[iL], self._jPts[iL]));
        self._numLinks-=1;
        self._added[self._iPts[iL]]=0;
        self._added[self._jPts[iL]]=0;
        del self._iPts[iL];
        del self._jPts[iL];
        del self._Shifts[iL];
    
    def setLinks(self,iPts,jPts,Shifts):
        """
        Set the links from input vectors of iPts, jPts, and Shifts  
        """
        self._iPts = iPts;
        self._jPts = jPts;
        self._Shifts = Shifts;
        self._numLinks = len(iPts);
    
    def setLinksFromFile(self,FileName,Dom):
        """
        Set the links from a file name. The file has a list of iPts, 
        jPts (two ends of the links), and shift in zero strain coordinates 
        """
        warn('Setting links from file - this is only functional for PERMANENT links')
        AllLinks = np.loadtxt(FileName);
        self._numLinks = len(AllLinks)-1;
        self._iPts = list(AllLinks[1:,0].astype(int));
        self._jPts = list(AllLinks[1:,1].astype(int));
        for iL in range(self._numLinks):
            shiftUnprime = -AllLinks[iL+1,2:]
            shiftPrime = Dom.primecoords(shiftUnprime);
            self._Shifts.append(shiftPrime);

    def calcKoff(self,uniPts,Dom):
        """
        Method to calculate the rate of unbinding for 
        the current links. 
        Inputs: uniPts = (Nsites*Nfib)x3 array of uniform binding 
        sites on the fibers, Dom = Domain object
        Outputs: the rates of unbinding (units 1/s) for each link
        In this parent class, we assume a constant rate of unbinding
        (in child classes will depend on the strain rate)
        """
        Koffs = self._koff*np.ones(self._numLinks);
        return Koffs;
    
    def calcKoffOne(self,iPt,jPt,uniPts,Dom):
        """
        Calculate the unbinding rate for a single link. 
        Inputs: link endpoints iPt, jPt. 
        uniPts = (Nsites*Nfib)x3 array of uniform binding 
        sites on the fibers, Dom = Domain object
        Output: the rate for a single rate.
        """
        return self._koff;
    
    def calcKon(self,newLinks,uniPts,Dom):
        """
        Method to calculate the rate of binding for 
        a set of possible newLinks.  
        Inputs: newLinks = nNew x 2 array of pairs of new uniform 
        points to link, uniPts = (Nsites*Nfib)x3 array of uniform binding 
        sites on the fibers, Dom = Domain object
        Outputs: the rates of binding (units 1/s) for each link
        In this parent class, we assume a constant rate of binding
        as long as the 2 points are actually closer than self._clcut
        """
        nNew, _ = newLinks.shape;
        Kons = np.zeros(nNew);
        for iL in range(nNew):
            iPt = newLinks[iL,0];
            jPt = newLinks[iL,1];
            Kons[iL] = self.calcKonOne(iPt,jPt,uniPts,Dom); # otherwise will stay 0
        return Kons;
    
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
        if (r < self._clcut):
            return self._kon;
        return 0;                        

    def writeLinks(self,of,uniPts,Dom):
        """
        Write the links to a file object of. 
        """
        of.write('%d \t %d \t %f \t %f \t %f \n' %(self._numLinks,-1,-1,-1,-1));
        for iL in range(self._numLinks):
            shift = Dom.unprimecoords(self._Shifts[iL]); 
            rvec = uniPts[self._iPts[iL],:]-uniPts[self._jPts[iL],:]-shift;
            of.write('%d \t %d \t %f \t %f \t %f \n' \
                %(self._iPts[iL], self._jPts[iL],rvec[0],rvec[1],rvec[2]));
    
    def writePossibles(self,newLinks):
        """
        Write the possible links to a file object of. 
        """
        nNew, _ = newLinks.shape;
        outFile = 'PossibleLinks.txt';
        of = open(outFile,'w');
        for iL in range(nNew):
            of.write('%d \t %d \n' %(newLinks[iL,0], newLinks[iL,1]));
        of.close();
        
class KMCCrossLinkedNetwork(CrossLinkedNetwork):
    
    """
    Cross-linked network that implements Kinetic MC for the making and breaking of
    links
    
    """
    def __init__(self,nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads=1):
        """
        Constructor is just the parent constructor
        """
        super().__init__(nFib,N,Nunisites,Lfib,nCL,kCL,rl,kon,koff,CLseed,Dom,fibDisc,nThreads);
    
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
        
        # Form two column array of possible pairs for linking. 
        uniNeighbs = SpatialDatabase.selfNeighborList(self._clcut);
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = newLinks.shape;
        
        # Calculate (un)binding rates for each event
        rateUnbind = self.calcKoff(uniPts,Dom);
        rateBind = clinks.calcKons(newLinks, uniPts,Dom.getg()); 
            
        # Make lists of pairs of points and events
        totiPts = np.concatenate((np.array(self._iPts,dtype=int),newLinks[:,0]));
        totjPts = np.concatenate((np.array(self._jPts,dtype=int),newLinks[:,1]));
        nowBound = np.concatenate((np.ones(len(self._iPts),dtype=int),np.zeros(nPairs,dtype=int)));
        allRates = np.concatenate((rateUnbind,rateBind));
        if (len(allRates)==0):
            return;
        added = self._added.copy();
        nLinks = self._numLinks; # local copy
        # C++ function gives a list of events in order, indexed by the number in the list
        # totiPts, totjPts, etc. 
        #print(len(allRates))
        events = clinks.newEvents(allRates, totiPts, totjPts, nowBound, added,nLinks, uniPts,Dom.getg(),tstep);
        #print(events)
        # Post process to do the events
        breakLinks = -np.sort(-events[events < len(self._iPts)]); # links to break
        #print('Links to break (in descending order)');
        #print(breakLinks);
        #print('Links to make');
        #print(np.setdiff1d(events,breakLinks))
        # Break some links
        for iL in breakLinks:
            self.removeLink(iL);
        for newL in np.setdiff1d(events,breakLinks): # make remaining links
            self.addLink(totiPts[newL],totjPts[newL],uniPts,Dom);
        if (of is not None): 
            self.writeLinks(of,uniPts,Dom);
            self.writePossibles(newLinks);
    