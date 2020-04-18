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
    
    def __init__(self,nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff,CLseed):
        """
        Initialize the CrossLinkedNetwork. 
        Input variables: nFib = number of fibers, N = number of points per
        fiber, Nunisites = number of uniformly placed CL binding sites per fiber
        Lfib = length of each fiber, grav = strength of gravity (total
        strength as a force, not force density), nCL = number of cross linkers, 
        kCL = cross linking spring constant, rl = rest length of the CLs, 
        kon = constant rate of binding (constant if inside a CL rest length), 
        koff = constant rate of unbinding (pre-factor in front of the rate)
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
        # Seed C++ random number generator
        clinks.seedRandomCLs(CLseed);
    
    ## ==============================================
    ## PUBLIC METHODS CALLED BY TEMPORAL INTEGRATOR
    ## ==============================================
    def updateNetwork(self,fiberCol,Dom,tstep,of=None):
        """
        Method to update the network. In the abstract parent class, 
        this method does nothing rigorous mathematically, just 
        breaks all links that are too far (farther than self._clcut)
        and makes links that are close enough to be formed (within self._clcut)
        Inputs: fiberCol = fiberCollection object of the fibers,
        Dom = Domain object, tstep = the timestep we are updating the network
        by (a different name than dt)
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
        rateBind = clinks.calcKon(nPairs, newLinks[:,0], newLinks[:,1], np.reshape(uniPts,\
            3*self._nFib*self._NsitesPerf),self._clcut,self._kon,Dom.getg(),Dom.getLens()); 
        # Make lists of pairs of points and events
        totiPts = np.concatenate((np.array(self._iPts,dtype=int),newLinks[:,0]));
        totjPts = np.concatenate((np.array(self._jPts,dtype=int),newLinks[:,1]));
        nowBound = np.concatenate((np.ones(len(self._iPts),dtype=int),np.zeros(nPairs,dtype=int)));
        allRates = np.concatenate((rateUnbind,rateBind));
        # Unique rows
        c = np.concatenate(([totiPts],[totjPts]),axis=0);
        _, uinds = np.unique(c.T,axis=0,return_index=True);
        uinds = np.sort(uinds);
        totiPts = totiPts[uinds];
        totjPts = totjPts[uinds];
        nowBound = nowBound[uinds];
        allRates = allRates[uinds];
        removeinds = [];
        addinds = [];
        linkorder = np.random.permutation(np.arange(len(totiPts)));
        #print(linkorder);
        added = self._added.copy();
        for iL in linkorder:
            change = np.random.rand() < tstep*allRates[iL];
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
        breakLinks = -np.sort(-np.array(removeinds,dtype=int)); # links to break
        #print('Links to break (in descending order)');
        #print(breakLinks);
        # Break some links
        for iL in breakLinks:
            self.removeLink(iL);
        for newL in addinds: # make remaining links
            self.addLink(totiPts[newL],totjPts[newL],uniPts,Dom);
        if (of is not None): 
            self.writeLinks(of,uniPts,Dom);
        
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
        GravForces = np.tile([0,0,self._grav/self._Lfib],self._Npf*self._nFib);
        if (self._numLinks==0): # if no CLs don't waste time
            return GravForces;
        print('Computing CL forces with %d links' %self._numLinks);
        ptsxyz=np.zeros((self._Npf*self._nFib,3));     # points
        # Call the C++ function to compute forces
        sCheb = fibDisc.gets();
        wCheb = fibDisc.getw();
        CLForces = np.zeros((self._Npf*self._nFib,3));
        shifts = Dom.unprimecoords(np.array(self._Shifts));
        Clforces = clinks.CLForces(self._numLinks, self._iPts, self._jPts, self._su,shifts[:,0], \
                shifts[:,1], shifts[:,2],chebPts[:,0],chebPts[:,1],chebPts[:,2],self._nFib,\
                self._Npf,sCheb,wCheb,self._kCL,self._rl, self._sigma);
        return np.array(Clforces)+GravForces;
    
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
        are distances in the variables (x',y',z'). 
        """
        rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
        shift = uniPts[iPt,:]-uniPts[jPt,:] - rvec;
        #print('Making link between %d and %d with distance %f' %(iPt, jPt, np.linalg.norm(rvec)));
        if (self._added[iPt]==1 or self._added[jPt]==1):
            raise ValueError('Bug - trying to make link in already occupied site');
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
        Output: the rate for a single rate.
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
        Write the links to a file object of. 
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
    def __init__(self,nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff,CLseed):
        """
        Constructor is just the parent constructor
        """
        super().__init__(nFib,N,Nunisites,Lfib,grav,nCL,kCL,rl,kon,koff,CLseed);
    
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
        rateBind = clinks.calcKon(nPairs, newLinks[:,0], newLinks[:,1], np.reshape(uniPts,\
            3*self._nFib*self._NsitesPerf),self._clcut,self._kon,Dom.getg(),Dom.getLens()); 
        # Make lists of pairs of points and events
        totiPts = np.concatenate((np.array(self._iPts,dtype=int),newLinks[:,0]));
        totjPts = np.concatenate((np.array(self._jPts,dtype=int),newLinks[:,1]));
        nowBound = np.concatenate((np.ones(len(self._iPts),dtype=int),np.zeros(nPairs,dtype=int)));
        allRates = np.concatenate((rateUnbind,rateBind));
        #print('Rates')
        #print(allRates)
        added = self._added.copy();
        nLinks = self._numLinks; # local copy
        # C++ function gives a list of events in order, indexed by the number in the list
        # totiPts, totjPts, etc. 
        #events = clinks.newEventsCL(allRates, totiPts, totjPts, nowBound, added,nLinks, self._nCL,\
        #    np.reshape(uniPts,3*self._nFib*self._NsitesPerf),Dom.getg(), Dom.getLens(),tstep, \
        #    self._kon, self._koff, self._clcut);
        if (True): # Python version
            # Compute random times for each event to happen
            randNums = -np.log(1-np.random.rand(len(allRates)));
            times = randNums/allRates;
            events = [];
            systime = 0;
            print('Possibles')
            print(np.concatenate(([totiPts],[totjPts])))
            print('Already linked')
            print([self._iPts, self._jPts])
            print(randNums)
            while (systime < tstep):
                print('===============================')
                print('Number of links %d' %nLinks);
                nowtimes = self.updateTimes(times.copy(),totiPts,totjPts,nowBound,added,nLinks);
                print('Rates')
                print(allRates)
                print('Current times')
                print(nowtimes);
                print('Bound states')
                print(nowBound)
                # Find the minimum time and corresponding event 
                nextEvent = np.argmin(nowtimes);
                print('Next event %d' %nextEvent);
                iPt = totiPts[nextEvent];
                jPt = totjPts[nextEvent];
                systime+= nowtimes[nextEvent];
                print('The next time %f' %systime);
                # Add the event to the list of events
                if (systime < tstep):
                    if (nextEvent in events): # already done, returning to OG state, remove from list
                        print('Removing event already in the list')
                        events.remove(nextEvent);
                    else: 
                        events.append(nextEvent);
                    nowBound[nextEvent] = 1 - nowBound[nextEvent]; # change bound state of that event
                    added[iPt]=1-added[iPt]; # change local copy of added (not class member)
                    added[jPt]=1-added[jPt];
                    nLinks+=2*nowBound[nextEvent]-1; # -1 if now unbound, 1 if now bound
                    print('New number of links %d' %nLinks)
                    # Compute the rate and time for the next event
                    if (nowBound[nextEvent]): # just became bound, calc rate to unbind
                        allRates[nextEvent] = self.calcKoffOne(iPt,jPt,uniPts,Dom);
                    else: # just became unbound, calculate rate to bind back
                        # To prevent double counting of links, those that are originally 
                        # bound are only allowed to unbind once. The binding then occurs 
                        # in indices with nextEvent >= len(self._iPts)
                        if (nextEvent < len(self._iPts)): # those originally bound
                            allRates[nextEvent] = 0; # ones that unbind cannot rebind (avoid double count)
                        else: 
                            allRates[nextEvent] = self.calcKonOne(iPt,jPt,uniPts,Dom);
                    # Redraw random time for that event
                    r=1-np.random.rand();
                    print('Python rand %f' %r)
                    times[nextEvent] = -np.log(r)/allRates[nextEvent];
                    print('New time %f' %times[nextEvent]);
                    #times[nextEvent] = 0.1/allRates[nextEvent]; # to compare with C++
            #if (len(events) > 0): # compare Python w C++
            #    if (np.amax(np.abs(np.array(events)-np.array(eventscpp))) > 0):
            #        raise ValueError('Cpp and python dont match!')
        # Post process to do the events
        events = np.array(events,dtype=int);
        breakLinks = -np.sort(-events[events < len(self._iPts)]); # links to break
        print('Links to break (in descending order)');
        print(breakLinks);
        print('Links to make');
        print(np.setdiff1d(events,breakLinks))
        # Break some links
        for iL in breakLinks:
            self.removeLink(iL);
        for newL in np.setdiff1d(events,breakLinks): # make remaining links
            self.addLink(totiPts[newL],totjPts[newL],uniPts,Dom);
        if (of is not None): 
            self.writeLinks(of,uniPts,Dom);
            self.writePossibles(newLinks);
    
    def updateTimes(self,times,totiPts,totjPts,nowBound,added,nLinks):
        """
        Update the times of binding for changed states of the network. 
        Inputs: times = array or original times for each event, 
        (totiPts, totjPts) = the endpoints of each event, nowBound = 
        bound state for each event (1 if bound, 0 if unbound), added = 
        Nsites*Nfib array indicating whether there is a link at each site
        nLinks = current number of links in the system
        Output: the new updated array of times
        """
        for iL in range(len(nowBound)):
            if (nowBound[iL]==0): # currently not bound, want to bind
                if (nLinks == self._nCL or\
                    added[totiPts[iL]] or added[totjPts[iL]]):
                    # Can't bind if no links available or if site blocked
                    times[iL] = float('inf'); # will never happen
            else: # want to unbind
                if (not added[totiPts[iL]] or not added[totjPts[iL]]):
                    # Cannot unbind if there is nothing there to unbind
                    times[iL] = float('inf'); # will never happen
        return times;
            