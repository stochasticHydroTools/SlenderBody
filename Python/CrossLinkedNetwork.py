import numpy as np
import CLUtils as clinks
from math import log

class CrossLinkedNetwork(object):

    """
    Object with all the cross-linking information. 
    The abstraction is just a class with some cross linking binding sites 
    in it, the children implement the update methods. 
    Currently, KMCCrossLinkedNetwork implements kinetic Monte Carlo
    to update the network
    """
    
    def __init__(self,nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff):
        """
        Initialize the CrossLinkedNetwork. 
        Input variables: nFib = number of fibers, N = number of points per
        fiber, Nunisites = number of uniformly placed CL binding sites per fiber
        Lfib = length of each fiber, grav = strength of gravity (total
        strength as a force, not force density), nCL = number of cross linkers, 
        kCL = cross linking spring constant, rl = rest length of the CLs, 
        kon = rate of binding (constant if inside a CL rest length), 
        koff = rate of unbinding (also constant)
        """
        self._grav = grav; 
        self._Npf = N; 
        self._NsitesPerf = Nunisites;
        self._Lfib = Lfib;
        self._nFib = nFib;
        self._nCL = nCL;
        self._kCL = kCL;
        self._rl = rl;
        self._kon = kon;
        self._koff = koff;
        self._sigma = 0.05*self._Lfib; # standard deviation of Gaussian for smooth points
        self._numLinks = 0; # number of links taken
        self._iPts = []; # one in pairs of the uniform points (LIST)
        self._jPts = []; # one in pairs of the uniform points (LIST)
        self._Shifts = [];    # periodic shifts for each link (in the DEFORMED DOMAIN)
        self._added = np.zeros(self._NsitesPerf*self._nFib,dtype=int); # whether there's a link at that point
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length of the fiber.
        if (self._Npf < 24):
            self._sigma = 0.10*self._Lfib;
        elif (self._Npf < 32):
            self._sigma = 0.07*self._Lfib;
        # Cutoff distance to form the CLs.
        self._clcut = self._rl;
        # Uniform binding locations
        self._su = np.linspace(0.0,2.0,self._NsitesPerf,endpoint=True);
    
    ## ==============================================
    ## PUBLIC METHODS CALLED BY TEMPORAL INTEGRATOR
    ## ==============================================
    def updateNetwork(self,fiberCol,Dom,tstep):
        """
        Method to update the network. In the abstract parent class, 
        this method does nothing rigorous mathematically, just 
        breaks all links that are too far (farther than self._clcut)
        and makes links that are close enough to be formed (within self._clcut)
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        """
        chebPts = fiberCol.getX();
        uniPts = fiberCol.getUniformPoints(chebPts);
        SpatialDatabase = fiberCol.getUniformSpatialData();
        # Break links
        self.breakLinks(uniPts,Dom);
        # Make new links
        self.makeLinks(uniPts,SpatialDatabase,Dom);

    def CLForce(self,fibDisc,chebPts,Dom):
        """
        Compute the total cross linking force.
        Inputs: fibDisc = fiberDiscretization object that 
        has the discretization info for each fiber. chebPts = 
        chebyshev points we are computing force at for all fibers, 
        Dom = Domain object. 
        Outputs: nPts*3 one-dimensional array of the forces at the 
        Chebyshev points due to the CLs
        """
        if (self._numLinks==0): # if no CLs don't waste time
            return np.zeros(self._Npf*self._nFib*3);
        ptsxyz=np.zeros((self._Npf*self._nFib,3));     # points
        # Call the C++ function to compute forces
        sCheb = fibDisc.gets();
        wCheb = fibDisc.getw();
        CLForces = np.zeros((self._Npf*self._nFib,3));
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        Clforces = clinks.CLForces(self._numLinks, self._iPts, self._jPts, self._su,shifts[:,0], \
                shifts[:,1], shifts[:,2],chebPts[:,0],chebPts[:,1],chebPts[:,2],self._nFib,\
                self._Npf,sCheb,wCheb,self._kCL,self._rl, self._sigma);
        return np.array(Clforces);
    
    def CLStress(self,fibCollection,fibDisc,Dom):
        """
        Compute the cross-linker contribution to the stress.
        Inputs: fibCollection = fiberCollection object to compute
        the stress on, fibDisc = fiber Discretization object describing
        each fiber, Dom = Domain (needed to compute the volume)
        """
        stress = np.zeros((3,3));
        sCheb = fibDisc.gets();
        wCheb = fibDisc.getw();
        chebPts = fibCollection.getX();
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        for iLink in range(self._numLinks):
            iFib = self._iPts[iLink] // self._NsitesPerf;
            jFib = self._jPts[iLink] // self._NsitesPerf;
            iInds = fibCollection.getRowInds(iFib);
            jInds = fibCollection.getRowInds(jFib);
            Clforces = np.reshape(clinks.CLForces(1, [self._iPts[iLink]], [self._jPts[iLink]], self._su,[shifts[iLink,0]], \
                [shifts[iLink,1]], [shifts[iLink,2]],chebPts[:,0],chebPts[:,1],chebPts[:,2],self._nFib,\
                self._Npf,sCheb,wCheb,self._kCL,self._rl, self._sigma),(chebPts.shape));
            for iPt in range(self._Npf):
                stress-=wCheb[iPt]*np.outer(chebPts[iInds[iPt],:],Clforces[iInds[iPt],:]);
                stress-=wCheb[iPt]*np.outer(chebPts[jInds[iPt],:]+shifts[iLink,:],Clforces[jInds[iPt],:]);
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
        are domain lengths in the deformed domain. 
        """
        rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
        shift = uniPts[iPt,:]-uniPts[jPt,:] - rvec;
        print('Making link between %d and %d with distance %f' %(iPt, jPt, np.linalg.norm(rvec)));
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
        print('Breaking link between %d and %d' %(self._iPts[iL], self._jPts[iL]));
        self._numLinks-=1;
        self._added[self._iPts[iL]]=0;
        self._added[self._jPts[iL]]=0;
        del self._iPts[iL];
        del self._jPts[iL];
        del self._Shifts[iL];
        
    def breakLinks(self,uniPts,Dom):
        """
        Break any existing cross-linkers if they are too strained.
        In the abstract parent class, we break links if they have a length
        greater than self._clcut
        Inputs: uniPts = (Nsites*Nfib) x 3 array of uniform possible binding 
        locations on each fiber. Dom = domain object
        """
        # Loop over the links in reverse
        for iL in range(self._numLinks-1,-1,-1): # loop through backwards
            shift = Dom.unprimecoords(self._Shifts[iL]); 
            rvec = uniPts[self._iPts[iL],:]-uniPts[self._jPts[iL],:]-shift;
            # Break link if too strained
            if (np.linalg.norm(rvec) > self._clcut):
                self.removeLink(iPt,jPt,iL);

    def makeLinks(self,uniPts,SpatialDatabase,Dom):
        """
        Add new links to the lists.
        Inputs: uniPts = (Nuni*Nfib) x 3 array of uniform possible binding 
        locations on each fiber. SpatialDatabase = database object for 
        finding neighbors, Dom = domain object
        In the abstract parent class, we make all links that are within 
        self._clcut of each other, until we run out of cross-linkers. 
        """
        if (self._numLinks==self._nCL): # all the links are occupied
            return;
        # Obtain possible pairs of uniform points to consider
        SpatialDatabase.updateSpatialStructures(uniPts,Dom);
        uniNeighbs = SpatialDatabase.selfNeighborList(self._clcut);
        nPairs, _ = uniNeighbs.shape;
        # Randomly orient the pairs to check for CLs
        pairOrder = np.random.permutation(nPairs);
        for iPair in pairOrder: # looping through points
            iPt = uniNeighbs[iPair,0];
            jPt = uniNeighbs[iPair,1];
            iFib = iPt // self._NsitesPerf; # integer division to get fiber number
            jFib = jPt // self._NsitesPerf;
            alreadyLinked = self._added[jPt]==1 or self._added[iPt]==1;
            if (not iFib==jFib and not alreadyLinked): # cannot link to the same fiber or if site occupied
                # Find nearest periodic image
                rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
                shift = uniPts[iPt,:]-uniPts[jPt,:] - rvec;
                # Only actually do the computation when necessary
                if (np.linalg.norm(rvec) < self._clcut):                    
                    self.addLink(iPt,jPt,uniPts,Dom);
                    if (self._numLinks==self._nCL): # stop when all CLs added
                        return;  

class KMCCrossLinkedNetwork(CrossLinkedNetwork):
    
    """
    Cross-linked network that implements Kinetic MC for the making and breaking of
    links
    """
    def __init__(self,nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff):
        super().__init__(nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff);
    
    def updateNetwork(self,fiberCol,Dom,tstep):
        """
        Update the network using Kinetic MC.
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
        This is an event-driven algorithm. We sample a time for binding/unbinding
        and then update the network to that state
        """
        if (self._nCL==0):
            return; # do nothing if there are no CLs
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
        uniNeighbs = np.delete(uniNeighbs,delInds[iFibs==jFibs],axis=0);
        nPairs, _ = uniNeighbs.shape;
        # Event-driven algorithm
        systime = 0;
        while (systime < tstep):
            # Remove rows of uniNeighbs that have points already bound -
            # those have to unbind first before they can be linked 
            BadPairs = [];
            for iPair in range(nPairs):
                if (self._added[uniNeighbs[iPair,0]] or self._added[uniNeighbs[iPair,1]]):
                    BadPairs.append(iPair);
            print('Pairs that are already taken')
            print(BadPairs)
            # Find the pairs that can link to each other
            Goodpairs = np.setdiff1d(np.arange(nPairs),np.array(BadPairs));
            newLinks = uniNeighbs[Goodpairs,:];
            print('Possible new links')
            print(newLinks)
            # Compute the rates of binding and unbinding
            rateUnbind = self.calcKoff(uniPts,Dom);
            rateBind = self.calcKon(newLinks,uniPts,Dom);
            print('Rates of unbinding')
            print(rateUnbind)
            print('Rates of binding')
            print(rateBind)
            # Compute random times for binding and unbinding
            tUnbind = -np.log(np.random.rand(len(self._iPts)))/rateUnbind; # unbinding times
            tfirstUnbind = float("inf");
            if (len(tUnbind) > 0):
                tfirstUnbind = min(tUnbind);
            tBind = -np.log(np.random.rand(len(Goodpairs)))/rateBind; # binding times
            tfirstBind = float("inf");
            if (len(tBind) > 0):
                tfirstBind = min(tBind);
            print('Binding times')
            print(tBind)
            print('Unbinding times')
            print(tUnbind)
            # Unbind first if that's what happens or if all the CLs are occupied
            if (tfirstUnbind < tfirstBind or self._numLinks==self._nCL):
                print('Unbinding first, moving to %f' %min(tUnbind));
                systime+= min(tUnbind);
                if (systime < tstep): # actually remove link if not over timestep
                    iL = np.argmin(tUnbind);
                    self.removeLink(iL);
            else: # Bind first
                print('Binding first, moving to %f' %min(tBind))
                systime+= min(tBind)
                if (systime < tstep): # actually add link if not over timestep
                    pair = newLinks[np.argmin(tBind)];
                    self.addLink(pair[0],pair[1],uniPts,Dom);
            print('New time %f' %systime);
        
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
        Koffs = np.zeros(self._numLinks);
        for iL in range(self._numLinks):
            #shift = Dom.unprimecoords(self._Shifts[iL]); 
            #rvec = uniPts[self._iPts[iL],:]-uniPts[self._jPts[iL],:]-shift;
            #r = np.linalg.norm(rvec);
            # Just a uniform probability
            Koffs[iL] = self._koff;
        return Koffs;
    
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
            # Find nearest periodic image
            rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
            # For now link with uniform probability if less than 
            # the CL cutoff distance
            r = np.linalg.norm(rvec);
            if (r < self._clcut):
                Kons[iL] = self._kon; # otherwise will stay 0
        return Kons;
            