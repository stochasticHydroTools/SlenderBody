import finufftpy as fi
import numpy as np
from scipy import special
from scipy.spatial import cKDTree
import EwaldUtils as ewc

class EwaldSplitter(object):
    """
    This class handles the Ewald splitting for the calculation of
    the non-local velocity.
    """
    def __init__(self,Lx,Ly,Lz,xi,a,mu):
        """
        Constructor. Initialize the Ewald splitter. 
        Input variables: Lx, Ly, Lz = periodic length in the x, y, and z directions
        xi = Ewald splitting parameter, a = hydrodynamic radius of the RPY blobs 
        (a = sqrt(3/2)*epsilon*L), mu = fluid viscosity
        """
        self._Lx = Lx;
        self._Ly = Ly;
        self._Lz = Lz;
        self._xi = xi;
        self._a = a;
        self._mu = mu; 
        self.calcrcut();

    def calcrcut(self):
        """
        Calculate the truncation distance for the Ewald near field. 
        We truncate the near field at 3 digits.
        """
        Lmin = min(self._Lx,self._Ly,self._Lz);
        ML2 = np.amax(abs(self.RPYNear([Lmin*0.5,0,0])));
        if (ML2 > 1e-3): # throw an error if the Ewald is too large (interactions with more than 1 image)
            raise ValueError('Need to increase xi or L, there are near interactions w/ more than 1 image');
        rcut=0;
        Mcut = abs(self.RPYNear([rcut,0,0]));
        M0 = min(np.amax(Mcut),1); # probably shouldn't have the 1 here, but self does get subtracted
        while (np.amax(Mcut)/M0 > 1e-3):
            rcut=rcut+1e-2;
            Mcut = abs(self.RPYNear([rcut,0,0]));   
        self._rcut =  rcut;
    
    def calcnBins(self,rcut):
        nx = int(np.floor(float(self._Lx)/rcut+1e-10));
        ny = int(np.floor(float(self._Ly)/rcut+1e-10));
        nz = int(np.floor(float(self._Lz)/rcut+1e-10));
        return nx, ny, nz;
    
    def EwaldFarVel(self,Npts,ptsxyz,forces,g):
        """
        This function computes the far field Ewald velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points,
        forces = forces at those points, g = the strain in the coordinate system. 
        This function relies entirely on calls to FINUFFT. See the documentation
        there for more information
        """
        # Compute the coordinates in the transformed basis
        pts = EwaldSplitter.primecoords(ptsxyz,g);
        # Rescale to [-pi,pi] (for FINUFFT)
        Lens = np.array([self._Lx,self._Ly,self._Lz]);
        pts = 2*np.pi*np.mod(pts,Lens)/Lens-np.pi;
        # Estimate the required grid
        gw=1.0/(2*self._xi);
        h = gw/1.6;
        nx, ny, nz = 2**(np.ceil(np.log2(Lens/h)));
        nx = int(nx+1e-10); ny=int(ny+1e-10); nz=int(nz+1e-10);
        # Wave numbers
        kvx = np.concatenate((np.arange(nx/2),np.arange(-nx/2,0)))*2*np.pi/self._Lx;
        kvy = np.concatenate((np.arange(ny/2),np.arange(-ny/2,0)))*2*np.pi/self._Ly;
        kvz = np.concatenate((np.arange(nz/2),np.arange(-nz/2,0)))*2*np.pi/self._Lz;
        ky, kx, kz = np.meshgrid(kvy,kvx,kvz);
        fxhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        fyhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        fzhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        tol=1e-10;
        # Forcing on the grid (FINUFFT type 1)
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,0],-1,tol,\
                    nx,ny,nz,fxhat,modeord=1);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,1],-1,tol,\
                    nx,ny,nz,fyhat,modeord=1);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,2],-1,tol,\
                    nx,ny,nz,fzhat,modeord=1);
        k = np.sqrt(kx*kx+(ky-g*kx)*(ky-g*kx)+kz*kz);
        # Multiplication factor for the RPY tensor
        factor = 1.0/(self._mu*k*k)*np.sinc(k*self._a/np.pi)**2;
        factor *= (1+k*k/(4*self._xi*self._xi))*np.exp(-k*k/(4*self._xi*self._xi));
        factor[0,0,0] = 0;
        uxhat = factor * fxhat;
        uyhat = factor * fyhat;
        uzhat = factor * fzhat;
        uprojx = uxhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*kx/(k*k);
        uprojx[0,0,0]=0;
        uprojy = uyhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*(ky-g*kx)/(k*k);
        uprojy[0,0,0]=0;
        uprojz = uzhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*kz/(k*k);
        uprojz[0,0,0]=0;
        ut = np.zeros([Npts],dtype=np.complex128);
        vt = np.zeros([Npts],dtype=np.complex128);
        wt = np.zeros([Npts],dtype=np.complex128);
        # Velocities at the points (FINUFFT type 2)
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],ut,1,tol,uprojx,modeord=1);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],vt,1,tol,uprojy,modeord=1);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],wt,1,tol,uprojz,modeord=1);
        vol = 1.0*self._Lx*self._Ly*self._Lz;
        ut = np.real(ut)/vol;
        vt = np.real(vt)/vol;
        wt = np.real(wt)/vol;
        return np.concatenate(([ut],[vt],[wt])).T;
    
    def EwaldNearVel(self,Npts,pts,forces,g):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points,
        forces = forces at those points, g = the strain in the coordinate system.
        This version uses binning and is SLOW!
        """
        velNear = np.zeros((pts.T).shape);
        # Sort points into bins
        rwsafety = self._rcut*(1+0.5*g*g+0.5*np.sqrt(g*g*(g*g+4.0)));
        self._nxBin, self._nyBin, self._nzBin = self.calcnBins(rwsafety);
        ptbins = self.binsbyP(pts,self._nxBin,self._nyBin,self._nzBin,g);
        print 'Number of bins (%d, %d, %d)' %(self._nxBin, self._nyBin, self._nzBin);
        bfirst, pnext = self.binPoints(Npts,pts,self._nxBin,self._nyBin,self._nzBin,g);
        # Go point by point to compute the velocity. Can parallelize this
        # outer loop as necessary.
        for iPt in xrange(Npts): # loop over points
            tbin=[ptbins[iPt,:]];
            sN = EwaldSplitter.neighborBins(tbin,self._nxBin,self._nyBin,self._nzBin);
            for iSn in xrange(len(sN)): # loop over neighboring bins
                jPt = bfirst[sN[iSn]];
                while (jPt !=-1):
                    # Find nearest periodic image (might need to speed this up)
                    rvec = self.calcShifted(pts[iPt,:]-pts[jPt,:],g);
                    # Only actually do the computation when necessary
                    if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+\
                        rvec[2]*rvec[2] < self._rcut*self._rcut):
                        velNear[:,iPt]+=ewc.RPYNKer(rvec,forces[jPt,:],self._mu,self._xi,self._a);
                    jPt = pnext[jPt];
        return velNear.T;
        
        
    def EwaldNearVelkD(self,Npts,ptsxyz,forces,g):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points,
        forces = forces at those points, g = the strain in the coordinate system.
        This method uses the kD tree to compute a list of the neighbors.
        """
        # Compute the coordinates in the transformed basis
        ptsprime = EwaldSplitter.primecoords(ptsxyz,g);
        # Sort points into kD tree using safety factor
        rwsafety = self._rcut*(1+0.5*g*g+0.5*np.sqrt(g*g*(g*g+4.0)));
        # Mod the points so they are on the right bounds [0,Lx] x [0,Ly] x [0,Lz]
        Lens = np.array([self._Lx,self._Ly,self._Lz]);
        ptsprime = np.mod(ptsprime,Lens);
        self._nearTree = cKDTree(ptsprime,boxsize=Lens);
        # Find all pairs (returns an array of the pairs)
        pairpts = self._nearTree.query_pairs(rwsafety,output_type='ndarray');
        Npairs = len(pairpts);
        # Call the C+ function which takes as input the pairs of points, number of points and gives
        # you the near field
        velNear = ewc.RPYNKerPairs(Npts,Npairs,pairpts[:,0],pairpts[:,1],ptsxyz[:,0],ptsxyz[:,1],ptsxyz[:,2],\
                forces[:,0],forces[:,1],forces[:,2],self._mu,self._xi,self._a,self._Lx, self._Ly,\
                self._Lz, g, self._rcut);
        return np.reshape(velNear,(Npts,3));
        

    
    def EwaldNearVelQuad(self,Npts,pts,forces,g):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points,
        forces = forces at those points, g = the strain in the coordinate system.
        This is the dumb quadratic method.
        """
        velNear = np.zeros((pts.T).shape);
        # Go point by point to compute the velocity. Can parallelize this
        # outer loop as necessary.
        for iPt in xrange(Npts): # loop over points
            for jPt in xrange(Npts):
                # Find nearest periodic image (might need to speed this up)
                rvec = self.calcShifted(pts[iPt,:]-pts[jPt,:],g);
                # Only actually do the computation when necessary
                if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+\
                    rvec[2]*rvec[2] < self._rcut*self._rcut):
                    velNear[:,iPt]+=ewc.RPYNKer(rvec,forces[jPt,:],self._mu,self._xi,self._a);
        return velNear.T;

    
    def binPoints(self,Npts,pts,nxBin,nyBin,nzBin,g):
        """ 
        Bin the points based on the truncation distance for the near field. 
        Inputs = number of pts, locations of pts (in (x,y,z) space), 
        g = strain in the coordinate system. 
        Returns the 2 linked lists first (for the first point in each bin) and
        next (for the next point in the bin).
        """
        bins = self.binsbyP(pts,nxBin,nyBin,nzBin,g);
        sbins = bins[:,0]+nxBin*bins[:,1]+nxBin*nyBin*bins[:,2];
        # Form the linked lists
        bfirst = -np.ones(nxBin*nyBin*nzBin,dtype=np.int);
        pnext = -np.ones(Npts,dtype=np.int);
        for iPt in xrange(Npts):
            if (bfirst[sbins[iPt]] == -1):
                bfirst[sbins[iPt]] = iPt;
            else:
                jPt = bfirst[sbins[iPt]];
                while (jPt !=-1):
                    jPtprev = jPt;
                    jPt = pnext[jPt];
                pnext[jPtprev] = iPt;
        return bfirst, pnext;
    
    def binsbyP(self,pts,nxBin,nyBin,nzBin,g):
        coords = EwaldSplitter.primecoords(pts,g);
        # Shift all coordinates so they are on [0,L]^3
        coords = coords - np.floor(coords/[self._Lx,self._Ly,self._Lz])\
                *[self._Lx,self._Ly,self._Lz];
        # Get the bin for each pt
        dperbin = np.array([[float(nxBin)/self._Lx, float(nyBin)/self._Ly,\
                                float(nzBin)/self._Lz]]);
        bins = np.int_(np.floor(coords*dperbin));
        bins = np.mod(bins,[nxBin, nyBin, nzBin]); # takes care of any rounding issues
        return bins;
    
    @staticmethod
    def neighborBins(tbin,nxBin,nyBin,nzBin):
        neighbors = tbin+np.int_([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],\
                        [1,1,0],[-1,1,0],[0,-1,0],[1,-1,0],[-1,-1,0],[0,0,1],\
                        [1,0,1],[-1,0,1],[0,1,1],[1,1,1],[-1,1,1],[0,-1,1],\
                        [1,-1,1],[-1,-1,1],[0,0,-1],[1,0,-1],[-1,0,-1],\
                        [0,1,-1,],[1,1,-1],[-1,1,-1],[0,-1,-1],[1,-1,-1],\
                        [-1,-1,-1]]);
        neighbors = np.mod(neighbors,[nxBin, nyBin, nzBin]);
        _, idx = np.unique(neighbors,axis=0,return_index=True);
        neighbors = neighbors[np.sort(idx)];
        sN = neighbors[:,0]+nxBin*neighbors[:,1]+nxBin*nyBin*neighbors[:,2];
        return sN;
    
    def calcShifted(self,dvec,g):
        """
        Method to compute the nearest periodic image
        of a point. The periodicity is in (x',y',z') 
        where the strain in the coordinate system is given
        by g. 
        Inputs: vector (not shifted), and g
        Output: vector shifted so it's on [-L/2,L/2]^3 (this is 
        necessary to estimate the minimum norm)
        """
        # Just call the C++ function (the python was unbelievably slow!)
        newvec = ewc.calcShifted(dvec, g, self._Lx, self._Ly, self._Lz);
        #shift = 0*dvec;
        ## Shift in oblique y and z
        #shift[1] = round(dvec[1]/(1.0*self._Ly));
        #shift[2] = round(dvec[2]/(1.0*self._Lz));
        # Shift the vector
        #dvec[0]-=g*self._Ly*shift[1];
        #dvec[1]-=self._Ly*shift[1];
        #dvec[2]-=self._Lz*shift[2];
        # Shift in x
        #shift[0] = round(dvec[0]/self._Lx);
        #dvec[0]-=shift[0]*self._Lx;
        #print dvec - cppres
        return np.array(newvec);

    # Python versions of functions - not being used as these functions are being
    # done in C++ (except RPYNear to determine rcut)
    def RPYNear(self,rvec):
        """
        Evaluate the near field RPY kernel.
        Input: the vector rvec to evaluate the kernel at
        Output: value of kernel.
        This function relies on the C++ functions Fnear
        and Gnear, which evaluate the complicated near field \
        kernels faster than Python.
        """
        rvec = np.reshape(rvec,(1,3));
        r=np.sqrt(rvec[:,0]*rvec[:,0]+rvec[:,1]*rvec[:,1]+rvec[:,2]*rvec[:,2]);
        rhat=rvec/r;
        rhat[np.isnan(rhat)]=0;
        RR = np.dot(rhat.T,rhat);
        return 1.0/(6*np.pi*self._mu*self._a)*(ewc.Fnear(r,self._xi,self._a)*\
                (np.identity(3)-RR)+ewc.Gnear(r,self._xi,self._a)*RR);
     
    """
    def F(self,r,xi,a):
        if (r < 1e-10): # Taylor series
            val = 1/(4*np.sqrt(np.pi)*xi*a)*(1-np.exp(-4*a**2*xi**2)+\
                4*np.sqrt(np.pi)*a*xi*special.erfc(2*a*xi));
            return val;
        if (r>2*a):
            f0=0;
            f1=(18*r**2*xi**2+3)/(64*np.sqrt(np.pi)*a*r**2*xi**3);
            f2=(2*xi**2*(2*a-r)*(4*a**2+4*a*r+9*r**2)-2*a-3*r)/\
                (128*np.sqrt(np.pi)*a*r**3*xi**3);
            f3=(-2*xi**2*(2*a+r)*(4*a**2-4*a*r+9*r**2)+2*a-3*r)/\
                (128*np.sqrt(np.pi)*a*r**3*xi**3);
            f4=(3-36*r**4*xi**4)/(128*a*r**3*xi**4);
            f5=(4*xi**4*(r-2*a)**2*(4*a**2+4*a*r+9*r**2)-3)/(256*a*r**3*xi**4);
            f6=(4*xi**4*(r+2*a)**2*(4*a**2-4*a*r+9*r**2)-3)/(256*a*r**3*xi**4);
        else:
            f0=-(r-2*a)**2*(4*a**2+4*a*r+9*r**2)/(32*a*r**3);
            f1=(18*r**2*xi**2+3)/(64*np.sqrt(np.pi)*a*r**2*xi**3);
            f2=(2*xi**2*(2*a-r)*(4*a**2+4*a*r+9*r**2)-2*a-3*r)/\
                (128*np.sqrt(np.pi)*a*r**3*xi**3);
            f3=(-2*xi**2*(2*a+r)*(4*a**2-4*a*r+9*r**2)+2*a-3*r)/\
                (128*np.sqrt(np.pi)*a*r**3*xi**3);
            f4=(3-36*r**4*xi**4)/(128*a*r**3*xi**4);
            f5=(4*xi**4*(r-2*a)**2*(4*a**2+4*a*r+9*r**2)-3)/\
                (256*a*r**3*xi**4);
            f6=(4*xi**4*(r+2*a)**2*(4*a**2-4*a*r+9*r**2)-3)/\
                (256*a*r**3*xi**4);
        val = f0+f1*np.exp(-r**2*xi**2)+f2*np.exp(-(r-2*a)**2*xi**2)+\
            f3*np.exp(-(r+2*a)**2*xi**2)+f4*special.erfc(r*xi)+f5*special.erfc((r-2*a)*xi)+\
            f6*special.erfc((r+2*a)*xi);
        return val;
    
    def G(self,r,xi,a):
        if (r < 1e-10):
            return 0;
        if (r>2*a):
            g0=0;
            g1=(6*r**2*xi**2-3)/(32*np.sqrt(np.pi)*a*r**2*xi**3);
            g2=(-2*xi**2*(r-2*a)**2*(2*a+3*r)+2*a+3*r)/\
                (64*np.sqrt(np.pi)*a*r**3*xi**3);
            g3=(2*xi**2*(r+2*a)**2*(2*a-3*r)-2*a+3*r)/\
                (64*np.sqrt(np.pi)*a*r**3*xi**3);
            g4=-3*(4*r**4*xi**4+1)/(64*a*r**3*xi**4);
            g5=(3-4*xi**4*(2*a-r)**3*(2*a+3*r))/(128*a*r**3*xi**4);
            g6=(3-4*xi**4*(2*a-3*r)*(2*a+r)**3)/(128*a*r**3*xi**4);
        else:
            g0=(2*a-r)**3*(2*a+3*r)/(16*a*r**3);
            g1=(6*r**2*xi**2-3)/(32*np.sqrt(np.pi)*a*r**2*xi**3);
            g2=(-2*xi**2*(r-2*a)**2*(2*a+3*r)+2*a+3*r)/\
                (64*np.sqrt(np.pi)*a*r**3*xi**3);
            g3=(2*xi**2*(r+2*a)**2*(2*a-3*r)-2*a+3*r)/\
                (64*np.sqrt(np.pi)*a*r**3*xi**3);
            g4=-3*(4*r**4*xi**4+1)/(64*a*r**3*xi**4);
            g5=(3-4*xi**4*(2*a-r)**3*(2*a+3*r))/(128*a*r**3*xi**4);
            g6=(3-4*xi**4*(2*a-3*r)*(2*a+r)**3)/(128*a*r**3*xi**4);
        val = g0+g1*np.exp(-r**2*xi**2)+g2*np.exp(-(r-2*a)**2*xi**2)+\
            g3*np.exp(-(r+2*a)**2*xi**2)+g4*special.erfc(r*xi)+g5*special.erfc((r-2*a)*xi)+\
            g6*special.erfc((r+2*a)*xi);
        return val;
    """
    
    @staticmethod
    def primecoords(ptsxyz,g):
        """
        Method to calculate the coordinates of a vector 
        in the shifted coordinate system. Inputs = ptsxyz (pts in
        the x,y,z coordinate system), and the strain g. 
        Output is the vector pts', i.e. in the x',y',z' coordinate system
        """
        L = np.array([[1,g,0],[0,1,0],[0,0,1]]);
        pts = np.linalg.solve(L,ptsxyz.T).T;
        return pts;
    
    @staticmethod
    def unprimecoords(ptsprime,g):
        """
        Method to compute the coordinates of a point in the (x,y,z)
        coordinate system from the coordinates in the (x',y',z') coordinate
        system. Inputs: the prime coordinates and the strain g. 
        Outpts: the coordinates in the (x,y,z) space
        """
        L = np.array([[1,g,0],[0,1,0],[0,0,1]]);
        pts = np.dot(L,ptsprime.T).T;
        return pts;

        


