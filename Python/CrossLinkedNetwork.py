import numpy as np
import CrossLinking as clinks
from warnings import warn 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import time

verbose =-1;

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
        if (self._Npf <= 24):
            self._sigma = 0.10*self._Lfib;
        elif (self._Npf < 28):
            self._sigma = 0.07*self._Lfib;
        # Cutoff distance to form the CLs (temporarily the same as the rest length)
        self._lowercldelta = 0.01;
        # Add fudge factor because of discrete sites
        ds = self._Lfib/(self._NsitesPerf-1);
        self._uppercldelta = np.sqrt(((1+self._lowercldelta)*self._rl)**2 + ds**2/4)/self._rl-1;
        print('Discrete fudge factor %f ' % self._uppercldelta)
        # Allow links to form if two sites are in distance range (1-_lowercldelta)*l, (1+_uppercldelta)*l
        print('Sigma/L is %f' %(self._sigma/self._Lfib))
        # Uniform binding locations
        self._su = np.linspace(0.0,self._Lfib,self._NsitesPerf,endpoint=True);
        self._wCheb = fibDisc.getw();
        # Seed C++ random number generator and initialize the C++ variables for cross linking 
        clinks.seedRandomCLs(CLseed);
        DLens = Dom.getPeriodicLens();
        for iD in range(len(DLens)):
            if (DLens[iD] is None):
                DLens[iD] = 1e99;
        clinks.initDynamicCLVariables(self._kon,self._koff,self._lowercldelta,self._uppercldelta,DLens);
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
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        
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
 
    ## ==============================================
    ##     METHODS FOR INFORMATION ABOUT THE NETWORK
    ## ==============================================   
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
            linkStrains[iLink]=(np.linalg.norm(ds)-self._rl)/self._rl;
        #print(np.amax(np.abs(linkStrains)))
        return linkStrains;
        
    def NumSitesFull(self):
        """
        Determine how many sites on each fiber are occupied
        """
        NfullByFib = np.zeros(self._nFib);
        for iFib in range(self._nFib):
            NfullByFib[iFib]=np.sum(self._added[iFib*self._NsitesPerf:(iFib+1)*self._NsitesPerf]);
        return NfullByFib;
   
    def FindBundles(self):
        iFibs = np.array(self._iPts)//self._NsitesPerf;
        jFibs = np.array(self._jPts)//self._NsitesPerf;
        AdjacencyMatrix = csr_matrix((np.ones(self._numLinks),(iFibs,jFibs)),shape=(self._nFib,self._nFib));
        ConnectedMatrix=csr_matrix((np.ones(AdjacencyMatrix.nnz),AdjacencyMatrix.nonzero()),shape=(self._nFib,self._nFib))
        BundledFibs = AdjacencyMatrix -ConnectedMatrix; # only the ones that have 2 connections will be 1 now. 
        n_components, labels = connected_components(csgraph=BundledFibs, directed=False, return_labels=True)
        return n_components, labels;  
    
    def BundleOrderParameter(self,fibCollection,tanvecs,weights,n_components,labels):
        """
        Get the order parameter of each bundle
        """
        BundleMatrices = np.zeros((3*n_components,3));
        NPerBundle = np.zeros(n_components);
        OrderParams = np.zeros(n_components);
        for iFib in range(self._nFib):
            # Find cluster
            clusterindex = labels[iFib];
            NPerBundle[clusterindex]+=1;
            iInds = fibCollection.getRowInds(iFib);
            Xs = tanvecs[iInds,:];
            for p in range(self._Npf):
                BundleMatrices[clusterindex*3:clusterindex*3+3,:]+=np.outer(Xs[p,:],Xs[p,:])*weights[p];
        for clusterindex in range(n_components):
            B = 1/(NPerBundle[clusterindex]*self._Lfib)*BundleMatrices[clusterindex*3:clusterindex*3+3,:];
            EigValues = np.linalg.eigvalsh(B);
            OrderParams[clusterindex] = np.amax(np.abs(EigValues))
        return OrderParams, NPerBundle;
     
    
    def LocalOrientations(self,nEdges,fibCollection,tanvecs,weights):  
        """
        For every fiber within nEdges connections of me, what is the orientation
        Returns an nFib array
        """
        iFibs = np.array(self._iPts)//self._NsitesPerf;
        jFibs = np.array(self._jPts)//self._NsitesPerf;
        AdjacencyMatrix = csr_matrix((np.ones(self._numLinks),(iFibs,jFibs)),shape=(self._nFib,self._nFib));
        NeighborOrderParams = np.zeros(self._nFib);
        numCloseBy = np.zeros(self._nFib);
        Distances = shortest_path(AdjacencyMatrix, directed=False, unweighted=True, indices=None)
        for iFib in range(self._nFib):
            OtherFibDistances = Distances[iFib,:];
            Fibs = np.arange(self._nFib);
            CloseFibs = Fibs[OtherFibDistances <= nEdges];
            ninBundle = len(CloseFibs);
            numCloseBy[iFib]=ninBundle;
            B = np.zeros((3,3));
            for jFib in CloseFibs:
                jInds = fibCollection.getRowInds(jFib);
                Xs = tanvecs[jInds,:];
                for p in range(self._Npf):
                    B+=np.outer(Xs[p,:],Xs[p,:])*weights[p];
            B*= 1/(ninBundle*self._Lfib);
            EigValues = np.linalg.eigvalsh(B);
            NeighborOrderParams[iFib] = np.amax(np.abs(EigValues))
        return NeighborOrderParams, numCloseBy;
                   
    def CLForce(self,uniPoints,chebPoints,Dom,fibCollection):
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
        #import time
        #t=time.time();
        Clforces = clinks.calcCLForces(self._iPts, self._jPts,shifts,uniPoints,chebPoints,self._nThreads);
        return Clforces;
        #print('First method time %f' %(time.time()-t))
        # Alternative implementation for arbitrary binding pts
        t=time.time();
        iFibs = np.array(self._iPts)//self._NsitesPerf;
        iUnis = self._su[np.array(self._iPts) % self._NsitesPerf];
        jFibs = np.array(self._jPts)//self._NsitesPerf;
        jUnis = self._su[np.array(self._jPts) % self._NsitesPerf];
        iCenters = np.zeros((self._numLinks,3));
        jCenters = np.zeros((self._numLinks,3));
        fD = fibCollection._fiberDisc;
        # Compute coefficients of all fibers
        Coefficients = np.zeros((self._nFib*self._Npf,3));
        for iFib in range(self._nFib):
            iInds = fibCollection.getRowInds(iFib);
            Coefficients[iInds,:],_=fD.Coefficients(chebPoints[iInds,:]);
        Clforces2 = clinks.calcCLForces2(iFibs.astype(int),jFibs.astype(int),iUnis,jUnis,shifts,chebPoints,Coefficients,self._Lfib,self._nThreads);
        print('Second method time %f' %(time.time()-t))
        print('Error %f e-8' %(1e8*np.amax(np.abs(Clforces-Clforces2))))
        print(self._numLinks) 
    
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
        AllLinks = np.loadtxt(FileName);
        self._numLinks = len(AllLinks)-1;
        self._iPts = list(AllLinks[1:,0].astype(int));
        self._jPts = list(AllLinks[1:,1].astype(int));
        for iL in range(self._numLinks):
            shiftPrime = AllLinks[iL+1,2:]
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
        if (r < (1+self._uppercldelta)*self._rl and r > (1-self._lowercldelta)*self._rl):
            return self._kon;
        return 0;                        

    def writeLinks(self,of,uniPts):
        """
        Write the links to a file object of. 
        """
        of.write('%d \t %d \t %f \t %f \t %f \n' %(self._numLinks,-1,-1,-1,-1));
        for iL in range(self._numLinks):
            shift = self._Shifts[iL]; 
            of.write('%d \t %d \t %f \t %f \t %f \n' \
                %(self._iPts[iL], self._jPts[iL],shift[0],shift[1],shift[2]));
    
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
        uniNeighbs = SpatialDatabase.selfNeighborList((1+self._uppercldelta)*self._rl);
        # Filter the list of neighbors to exclude those on the same fiber
        iFibs = uniNeighbs[:,0] // self._NsitesPerf;
        jFibs = uniNeighbs[:,1] // self._NsitesPerf;
        delInds = np.arange(len(iFibs));
        newLinks = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = newLinks.shape;
        #print(nPairs)
        
        # Calculate (un)binding rates for each event
        rateUnbind = self.calcKoff(uniPts,Dom);
        #print(rateUnbind)
        rateBind = clinks.calcKons(newLinks, uniPts,Dom.getg()); 
        #print(rateBind)
            
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
        start = time.time()
        events = clinks.newEvents(allRates, totiPts, totjPts, nowBound, added,nLinks, uniPts,Dom.getg(),tstep);
        if (verbose > 0):
            print('C++ update time %f' %(time.time()-start))
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
    
