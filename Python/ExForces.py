import numpy as np
import chebfcns as cf
from EwaldStuff import EwaldSplitter
import CLUtils as clinks

class ExForces(object):
    
    def __init__(self,nFib,N,Lfib,grav,nCL,kCL,rl,Lx,Ly,Lz):
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
        self._xpShifts = [];    # periodic shifts for each link in x'
        self._ypShifts = [];    # periodic shifts for each link in y'
        self._zpShifts = [];    # periodic shifts for each link in y'
        self._added = np.zeros(self._Npf*self._nFib,dtype=int); # whether there's a link at that point
        self._Lx = Lx;
        self._Ly = Ly;
        self._Lz = Lz;
        # The standard deviation of the cross linking Gaussian depends on the
        # number of points per fiber and the length pf the fiber.
        if (self._Npf < 24):
            self._sigma = 0.10*self._Lfib;
        elif (self._Npf < 32):
            self._sigma = 0.07*self._Lfib;
        self._Ew = EwaldSplitter(Lx,Ly,Lz,10,0.1,0.1); # need access to the periodic stuff
        # Cutoff distance to form the CLs.
        self._clcut = self._rl;
        self._nxBin, self._nyBin, self._nzBin = self._Ew.calcnBins(self._clcut);
        #print 'Cutoff distance for CLs: %f' %self._clcut
        #print 'Number of bins: %d, %d, %d' %(self._nxBin, self._nyBin, self._nzBin);
        # Build the resampling matrix for the CLs
        self._srs_start = -1.0+4.0*self._sigma/(self._Lfib); # starting coord, rescaled to [-1,1]
        self._su = np.linspace(2*self._sigma,self._Lfib-2*self._sigma,self._Npf); # uniform arclength pts
        self._RS = cf.RSMat(self._Npf,self._Npf,'cl',1,self._srs_start,-self._srs_start); # resampling matrix
    
    
    def totalExForce(self,fibList,g,tint=2):
        """
        Compute the total external force from gravity and cross linking. 
        Inputs: fibList (list of fiber objects), double g = strain in the coordinate
        system, tint = temporal integrator (2 for second order CN, 1 
        for first order BE)
        """
        return self.addGravity() + self.CLForce(fibList,g,tint);
    
    def addGravity(self):
        """
        Gravitational force.
        """
        fgrav = np.tile([0,0,float(self._grav)/self._Lfib],self._Npf*self._nFib);
        return fgrav;

    def breakCLs(self,ptsunif,g):
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
            xsh = self._xpShifts[iL]*self._Lx+self._ypShifts[iL]*self._Ly*g;
            ysh = self._ypShifts[iL]*self._Ly;
            zsh = self._zpShifts[iL]*self._Lz;
            ds = ptsunif[iPt,:]-ptsunif[jPt,:]+[xsh, ysh, zsh];
            nds = np.sqrt(ds[0]*ds[0]+ds[1]*ds[1]+ds[2]*ds[2]);
            # Break the link if it's too strained (later will depend on e^(...))
            if  (nds > self._clcut*0.6):
                del self._iPts[iL];
                del self._jPts[iL];
                del self._xpShifts[iL];
                del self._ypShifts[iL];
                del self._zpShifts[iL];
                self._numLinks-=1;
                self._added[iPt]=0;
                self._added[jPt]=0;

    def addnewCLs(self,ptsunif,g):
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
        # Bin the uniform points
        uptBins = self._Ew.binsbyP(ptsunif,self._nxBin,self._nyBin,self._nzBin,g);
        ufirst, unext = self._Ew.binPoints(self._Npf*self._nFib,ptsunif,self._nxBin,self._nyBin,self._nzBin,g);
        # Randomly orient the points to check for CLs
        ptOrder = np.random.permutation(self._Npf*self._nFib);
        ptOrder = range(self._Npf*self._nFib); # temporary to compare w Matlab
        print 'Cross linkers looping through sequentially to compare w Matlab'
        for iPt in ptOrder: # looping through points
            iFib = int(iPt/self._Npf+1e-12);
            tbin = [uptBins[iPt,:]];
            sN = EwaldSplitter.neighborBins(tbin,self._nxBin,self._nyBin,self._nzBin);
            jPt = ufirst[sN[0]];
            for iSn in xrange(len(sN)): # loop through neighboring bins
                jPt = ufirst[sN[iSn]];
                while (jPt !=-1): # neighboring points
                    jFib = int(jPt/self._Npf+1e-12);
                    alreadyLinked = self._added[jPt]==1 or self._added[iPt]==1;
                    if (not iFib==jFib and not alreadyLinked): # cannot link to the same fiber or if site occupied
                        # Find nearest periodic image
                        rvec = self._Ew.calcShifted(ptsunif[iPt,:]-ptsunif[jPt,:],g);
                        shift = ptsunif[iPt,:]-ptsunif[jPt,:] - rvec;
                        # Only actually do the computation when necessary
                        if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]\
                             < self._clcut*self._clcut):
                            # Update lists
                            self._added[iPt]=1;
                            self._added[jPt]=1;
                            self._iPts.append(iPt);
                            self._jPts.append(jPt);
                            self._xpShifts.append(int(round(-shift[0]/self._Lx+g*shift[1]/self._Ly)));
                            self._ypShifts.append(int(round(-shift[1]/self._Ly)));
                            self._zpShifts.append(int(round(-shift[2]/self._Lz)));
                            self._numLinks+=1;
                            if (self._numLinks==self._nCL):
                                return;
                    jPt = unext[jPt];
                if (self._added[iPt]==1): # prevent multiple links from forming at same point per timestep
                    break;
                
    def CLForce(self,fibList,g,tint=2):
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
        ptsunif=np.zeros((self._Npf*self._nFib,3));     # uniform points
        # Fill up lists
        for iFib in xrange(len(fibList)):
            fib = fibList[iFib];
            # Get the position arguments from the fiber class.
            X, _,_,_ = fib.getNLargs(tint);
            rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
            ptsxyz[rowinds,:] = X;
            ptsunif[rowinds,:] = np.dot(self._RS,X);
        # Break any links
        self.breakCLs(ptsunif,g);
        # Add new links
        self.addnewCLs(ptsunif,g);
        # Call the C++ function to compute forces
        s, w = cf.chebpts(self._Npf,[0,self._Lfib],1);
        Clforces = clinks.CLForces(self._numLinks, self._iPts, self._jPts, self._su,self._xpShifts, \
                self._ypShifts, self._zpShifts,ptsxyz[:,0],ptsxyz[:,1],ptsxyz[:,2],self._nFib,\
                self._Npf,s,w,self._kCL,self._rl, g, self._Lx,self._Ly, self._Lz, self._sigma);
        #print np.reshape(Clforces,(self._Npf*self._nFib,3));
        return Clforces;
