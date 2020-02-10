import numpy as np
import CLUtils as clinks

class CrossLinkedNetwork(object):
    
    def __init__(self,nFib,N,Lfib,grav,nCL,kCL,rl):
        """
        Initialize the external forces class. 
        Input variables: nFib = number of fibers, N = number of points per
        fiber, Lfib = length of each fiber, grav = strength of gravity (total
        strength as a force (not force density), nCL = number of cross linkers, 
        kCL = cross linking spring constant, rl = rest length of the CLs, 
        Lx, Ly, Lz = periodic domain lengths in x, y, z direction 
        """
        self._grav = grav; # Gravitational force
        self._Npf = N; # number of points per fiber
        self._Lfib = Lfib;
        self._nFib = nFib;
        self._nCL = nCL;
        self._kCL = kCL;
        self._rl = rl;
        self._sigma = 0.05*self._Lfib;
        self._numLinks = 0; # number of links taken
        self._iPts = []; # one in pairs of the uniform points (LIST)
        self._jPts = []; # one in pairs of the uniform points (LIST)
        self._Shifts = [];    # periodic shifts for each link 
        self._added = np.zeros(self._Npf*self._nFib,dtype=int); # whether there's a link at that point
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length pf the fiber.
        if (self._Npf < 24):
            self._sigma = 0.10*self._Lfib;
        elif (self._Npf < 32):
            self._sigma = 0.07*self._Lfib;
        # Cutoff distance to form the CLs.
        self._clcut = self._rl;
        # Uniform binding locations
        self._su = np.linspace(0.0,2.0,self._Npf,endpoint=True);

    def updateNetwork(self,fibCol,chebPts,Dom,dt):
        uniPts = fibCol.getUniformPoints(chebPts);
        # Break links
        self._breakLinks(uniPts,Dom);
        # Make new links
        self._makeLinks(uniPts,Dom);

    def CLForce(self,fibDisc,chebPts,Dom):
        """
        Compute the total cross linking force.
        Inputs: ptsunif = (N x Nfib) x 3 array of uniform possible binding 
        locations on each fiber. g = strain in coordinate system. 
        Outputs: the force do to cross linking. 
        This method calls the method to break the links, then the method to 
        form new links, then computes the force associated with those links.
        """
        if (self._nCL==0): # if no CLs don't waste time
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

    def _breakLinks(self,uniPts,Dom):
        """
        Break any cross-linkers if they are too strained. 
        Inputs: ptsunif = (N x Nfib) x 3 array of uniform possible binding 
        locations on each fiber. g = strain in coordinate system.
        Outputs: nothing. It just deletes items in the lists of links
        (see docstring for addnewCLs for explanation of those lists)
        """
        if self._numLinks==0:
            return; # Can't break any links if there aren't any
        # Loop over the links in reverse
        for iL in xrange(self._numLinks-1,-1,-1): # loop through backwards
            iPt = self._iPts[iL];
            jPt = self._jPts[iL];
            shift = Dom.unprimecoords(self._Shifts[iL]); 
            ds = uniPts[iPt,:]-uniPts[jPt,:]+shift;
            nds = np.sqrt(ds[0]*ds[0]+ds[1]*ds[1]+ds[2]*ds[2]);
            # Break the link if it's too strained (later will depend on e^(...))
            if  (nds > 0.6*self._clcut):
                del self._iPts[iL];
                del self._jPts[iL];
                del self._Shifts[iL];
                self._numLinks-=1;
                self._added[iPt]=0;
                self._added[jPt]=0;

    def _makeLinks(self,uniPts,Dom):
        """
        Add new links to the lists.
        Inputs: ptsunif = (N x Nfib) x 3 array of uniform possible binding 
        locations on each fiber. g = strain in coordinate system. 
        This method will bin the uniform points, then go through the points
        randomly, checking the neighbors for possible links.
        If a link forms (it does with some probability if the two points are
        within a cutoff distance), update the lists iPts, jPts (uniform points 
        where the links are), and xpshifts, ypshifts, zpshifts (periodic shifts
        associated with each link). Increment number of links and set iPt and
        jPt sites as occupied.
        """
        if (self._numLinks==self._nCL): # all the links are occupied
            return;
        # Obtain possible pairs of uniform points to consider
        uniTree = Dom.makekDTree(uniPts);
        uniNeighbs = Dom.kDTreeNeighbors(uniTree,self._clcut);
        iPts=[87, 88, 121,  69,  27,  25,   4,  23,  64,  89,  26,   7,   5,  86, 120,  85,  80,  67,\
        123,  97,  79,  29,  24,   6,  66];
        jPts=[134, 133, 150, 112,  62, 116,  58, 115, 118, 132,  63,  55,  57, 127, 149, 128,  83, 111, \
            148, 158,  84, 153, 114,  56, 110];
        # nPairs, _ = uniNeighbs.shape;
        nPairs = len(iPts);
        # Randomly orient the points to check for CLs
        pairOrder = np.random.permutation(nPairs);
        for iPair in pairOrder: # looping through points
            iPt = iPts[iPair]-1;#uniNeighbs[iPair,0];
            jPt = jPts[iPair]-1;#uniNeighbs[iPair,1];
            iFib = int(iPt/self._Npf);
            jFib = int(jPt/self._Npf);
            alreadyLinked = self._added[jPt]==1 or self._added[iPt]==1;
            if (not iFib==jFib and not alreadyLinked): # cannot link to the same fiber or if site occupied
                # Find nearest periodic image
                rvec = Dom.calcShifted(uniPts[iPt,:]-uniPts[jPt,:]);
                shift = uniPts[iPt,:]-uniPts[jPt,:] - rvec;
                # Only actually do the computation when necessary
                if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]\
                     < self._clcut*self._clcut):
                    # Update lists
                    self._added[iPt]=1;
                    self._added[jPt]=1;
                    self._iPts.append(iPt);
                    self._jPts.append(jPt);
                    primeshift = Dom.primecoords(shift);
                    self._Shifts.append(primeshift);
                    self._numLinks+=1;
                    if (self._numLinks==self._nCL):
                        return;
                
