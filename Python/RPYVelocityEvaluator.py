import finufftpy as fi
import numpy as np
import numba as nb 
from RPYKernelEvaluator import RPYKernelEvaluator as RPYcpp
import time
from math import pi
from warnings import warn

verbose = -1;

# Documentation last updated: 03/12/2021

class RPYVelocityEvaluator(object):
    """
    This is a class that evaluates the velocity on N blobs due to N forces 
    The default is to do the "dumb" quadratic thing in free space - just 
    evaluating the total RPY kernel
    """
    ## ========================================
    ##          METHODS FOR INITIALIZATION
    ## ========================================
    def __init__(self,a,mu,Npts):
        """
        Input variables: a = hydrodynamic radius of the RPY blobs, mu = fluid viscosity,
        Npts = total number of blobs
        """
        self._a = float(a);
        print('Ewald radius %f' %self._a)
        self._mu = float(mu);
        self._Npts = Npts;

    ## =========================================
    ##    PUBLIC METHODS CALLED OUTSIDE CLASS
    ## =========================================
    def calcBlobTotalVel(self,ptsxyz,forces,Dom,SpatialData,nThr):
        """
        Compute the total velocity of Npts due to forces at those pts. 
        Inputs: ptsxyz = Npts x 3 array of locations, forces = Npts x 3 array of forces,
        Dom = Domain object where the computation is done, SpatialData = spatialDatabase
        that has the array of points,nThr = number of OpenMP threads
        Ouputs: Npts x 3 array of the velocities at the Npts by calling the Numba function
        to do the free space computation
        """
        return RPYVelocityEvaluator.RPYKernel(self._Npts,ptsxyz,self._Npts,ptsxyz,forces,self._mu,self._a);
    
    def NeedsGPU(self):
        return False;
    
    @staticmethod
    @nb.njit(nb.float64[:,:](nb.int64,nb.float64[:,:],nb.int64,nb.float64[:,:],\
        nb.float64[:,:],nb.float64,nb.float64))
    def RPYKernel(Ntarg,Xtarg,Nsrc,Xsrc,forces,mu,a):
        """
        The dumb quadratic method to sum the RPY kernel. 
        Inputs: Ntarg = # of targets, Xtarg = Ntarg x 3 array of locations of targets, 
        Nsrc = # of sources, Xsrc = Nsrc x 3 array of locations of sources, forces = 
        Nsrc x 3 array of forces, mu = viscosity, a = RPY radius. 
        Output: the velocities at the targets
        """
        utot=np.zeros((Ntarg,3));
        oneOvermu = 1.0/mu;
        for iTarg in range(Ntarg):
            for iSrc in range(Nsrc):
                rvec = Xtarg[iTarg,:]-Xsrc[iSrc,:];
                r = np.linalg.norm(rvec);
                rhat = rvec/r;
                rhat[np.isnan(rhat)]=0;
                rdotf = np.sum(rhat*forces[iSrc,:]);
                if (r>2*a):
                    fval = (2*a**2 + 3*r**2)/(24*pi*r**3);
                    gval = (-2*a**2 + 3*r**2)/(12*pi*r**3);
                else:
                    fval = (32*a - 9*r)/(192*a**2*pi);
                    gval = (16*a - 3*r)/(96*a**2*pi);
                utot[iTarg,:]+= oneOvermu*(fval*forces[iSrc,:]+rdotf*(gval-fval)*rhat);
        return utot;


## Some parameters specific to Ewald
nearcut = 1e-10; # cutoff for near field interactions
fartol = 1e-10; # far field tolerance for FINUFFT
rcuttol = 1e-2; # accuracy of truncation distance for Ewald
trouble_xi_step = 0.1; # if we have to increase Ewald parameter xi mid-run, how much should we increase by?
class EwaldSplitter(RPYVelocityEvaluator):

    """
    This class implements Ewald splitting for the calculation of
    the non-local velocity on a TRIPLY PERIODIC DOMAIN
    
    It uses FINUFFT for the far field and my own near field codes
    """
    ## ========================================
    ##          METHODS FOR INITIALIZATION
    ## ========================================
    def __init__(self,a,mu,xi,PerDom,Npts):
        """
        Constructor. Initialize the Ewald splitter. 
        Extra input variables: xi = Ewald splitting parameter,
        PerDom = PeriodicDomain object, Npts = number of blobs
        """
        super().__init__(a,mu,Npts);
        self._xi = float(xi);
        self._currentDomain = PerDom;
        
        # Initialize C++ code
        self._RPYcpp = RPYcpp(a,mu,Npts,PerDom.getPeriodicLens());
        
        # Calculate the truncation distance for Ewald
        self.calcrcut();
        self.updateFarFieldArrays();
        self._ufarx = np.zeros([self._Npts],dtype=np.complex128);
        self._ufary = np.zeros([self._Npts],dtype=np.complex128);
        self._ufarz = np.zeros([self._Npts],dtype=np.complex128);
        print('%.2E far field tolerance' %fartol)
        #warn('Tolerances for Ewald are low to compare to GPU code')
        
    ## =========================================
    ##    PUBLIC METHODS CALLED OUTSIDE CLASS
    ## =========================================
    def calcBlobTotalVel(self,ptsxyz,forces,Dom,SpatialData,nThr=1):
        """
        Total velocity due to Ewald (far field + near field). 
        Inputs: ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points,
        SpatialData = SpatialDatabase object for fast neighbor computation.
        nThr = number of threads for parallel processing
        Output: the total velocity as an Npts x 3 array.
        """
        # Update domain object
        self._currentDomain = Dom;
        # First check if Ewald parameter is ok
        self.checkrcut();
        # Compute far field and near field
        t = time.time();
        Ewaldfar = self.EwaldFarVel(ptsxyz,forces,nThr); # far field Ewald
        if (verbose>=0):
            print ('Far field Ewald time %f' %(time.time()-t));
        t=time.time()
        Ewaldnear = self.EwaldNearVel(ptsxyz,forces,SpatialData,nThr); # near field Ewald
        if (verbose>=0):
            print ('Near field Ewald time %f' %(time.time()-t));
        return Ewaldfar+Ewaldnear; 
    
    def NeedsGPU(self):
        return False;

    ## =========================================
    ##  PRIVATE METHODS ONLY CALLED WITHIN CLASS
    ## =========================================
    def EwaldFarVel(self,ptsxyz,forces,nThr):
        """
        This function computes the far field Ewald velocity. 
        Inputs: ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points.
        This function relies entirely on calls to FINUFFT. See the documentation
        there for more information.
        """
        # Compute the coordinates in the transformed basis
        pts = self._currentDomain.primecoords(ptsxyz);
        # Rescale to [-pi,pi] (for FINUFFT)
        Lens = self._currentDomain.getPeriodicLens();
        pts = 2*pi*np.mod(pts,Lens)/Lens-pi;
        # Forcing on the grid (FINUFFT type 1)
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,0],-1,fartol,\
                    self._nx,self._ny,self._nz,self._fxhat,modeord=1,nThreads=nThr);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,1],-1,fartol,\
                    self._nx,self._ny,self._nz,self._fyhat,modeord=1,nThreads=nThr);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,2],-1,fartol,\
                    self._nx,self._ny,self._nz,self._fzhat,modeord=1,nThreads=nThr);
        # Manipulation in Fourier space
        kxP, kyP, kzP = self._currentDomain.primeWaveNumbersFromUnprimed(self._kx, self._ky, self._kz);
        k = np.sqrt(kxP*kxP+kyP*kyP+kzP*kzP);
        # Multiplication factor for the RPY tensor
        factor = 1.0/(self._mu*k*k)*np.sinc(k*self._a/pi)**2;
        factor *= (1+k*k/(4*self._xi*self._xi))*np.exp(-k*k/(4*self._xi*self._xi)); # splitting function
        factor[0,0,0] = 0; # zero out 0 mode
        uxhat = factor * self._fxhat;
        uyhat = factor * self._fyhat;
        uzhat = factor * self._fzhat;
        # Project off so we get divergence free
        uprojx = uxhat-(kxP*uxhat+kyP*uyhat+kzP*uzhat)*kxP/(k*k);
        uprojx[0,0,0]=0;
        uprojy = uyhat-(kxP*uxhat+kyP*uyhat+kzP*uzhat)*kyP/(k*k);
        uprojy[0,0,0]=0;
        uprojz = uzhat-(kxP*uxhat+kyP*uyhat+kzP*uzhat)*kzP/(k*k);
        uprojz[0,0,0]=0;
        # Velocities at the points (FINUFFT type 2)
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],self._ufarx,1,fartol,uprojx,modeord=1,nThreads=nThr);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],self._ufary,1,fartol,uprojy,modeord=1,nThreads=nThr);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],self._ufarz,1,fartol,uprojz,modeord=1,nThreads=nThr);
        vol = self._currentDomain.getVol();
        return np.concatenate(([np.real(self._ufarx)/vol],[np.real(self._ufary)/vol],[np.real(self._ufarz)/vol])).T;
    
    def EwaldNearVel(self,ptsxyz,forces,SpatialData,nThreads=1):
        """
        Near field velocity. 
        Inputs: ptsxyz = the list of points in undeformed, Cartesian coordinates, 
        forces = forces at those points,SpatialData = SpatialDatabase object 
        for fast neighbor computation. nThreads = number of threads 
        Output: the near field velocity
        """
        # Find all pairs (returns an array of the pairs) within rcut
        neighborList = SpatialData.selfNeighborList(self._rcut);
        g = self._currentDomain.getg();
        # Call the C+ function which takes as input the pairs of points, number of points and gives
        # you the near field
        t=time.time();
        velNear = self._RPYcpp.EvaluateRPYNearPairs(neighborList,ptsxyz,forces,self._xi,g,self._rcut,nThreads);
        if (verbose>=0):
            print('Pairwise sum %f' %(time.time()-t))
        return velNear;

    def calcrcut(self):
        """
        Calculate the truncation distance for the Ewald near field. 
        We truncate the near field at the value rcut.
        """
        rcut=0;          # determine rcut
        Vatcut = np.abs(self._RPYcpp.RPYNearKernel([rcut,0,0],[1,0,0],self._xi));
        V0 = min(np.amax(Vatcut),1); # M0 is relative to the value at 0 or 1, whichever is smaller
        while (np.amax(Vatcut)/V0 > nearcut):
            rcut=rcut+rcuttol;
            Vatcut = np.abs(self._RPYcpp.RPYNearKernel([rcut,0,0],[1,0,0],self._xi));
        self._rcut =  rcut;
        print ('Ewald cut %f' %self._rcut)
    
    def checkrcut(self):
        """
        Check if rcut is less than one half period dynamically (the absolute
        length of a half period varies as the strain g changes). 
        If rcut is less than a half period, increase xi until rcut is less than a half
        period.
        """
        Lper = self._currentDomain.getPeriodicLens();
        try:
            Lmin = np.amin(Lper)/self._currentDomain.safetyfactor();
        except:
            raise NotImplementedError('Periodic velocity solver only implemented for triply periodic');
        vLover2 = np.amax(self._RPYcpp.RPYNearKernel([Lmin*0.5,0,0],[1,0,0],self._xi));
        if (vLover2 <= nearcut): # no interactions with more than 1 image
            return;
        print ('Need to increase xi or L, there are near interactions w/ more than 1 image');
        while (vLover2 > nearcut):
            # Modify xi
            self._xi+=trouble_xi_step;
            self.calcrcut();
            vLover2 = np.amax(self._RPYcpp.RPYNearKernel([Lmin*0.5,0,0],[1,0,0],self._xi));
            print('The new value of xi is %f' %self._xi)
            print('The new rcut %f' %self._rcut)
        # Update the far field arrays for the new xi
        self.updateFarFieldArrays();

    def updateFarFieldArrays(self):
        """
        Update/initialize the far field arrays when self._xi changes. 
        Method updates the self._waveNumbers on a standard 3-periodic grid and
        initialized the arrays for FINUFFT to put the forces.
        """
        Lens = self._currentDomain.getPeriodicLens();
        # Estimate the required grid
        gw=1.0/(2*self._xi); # width of the Gaussian
        h = gw/1.6;          # approximate grid spacing needed to resolve Gaussian
        nx, ny, nz = 2**(np.ceil(np.log2(Lens/h)));
        self._nx = int(nx); self._ny=int(ny); self._nz=int(nz); # number of grid points
        # Wave numbers (FFTW ordering) on UNDEFORMED COORDINATE SYSTEM
        kvx = np.concatenate((np.arange(self._nx/2),np.arange(-self._nx/2,0)))*2*pi/Lens[0];
        kvy = np.concatenate((np.arange(self._ny/2),np.arange(-self._ny/2,0)))*2*pi/Lens[1];
        kvz = np.concatenate((np.arange(self._nz/2),np.arange(-self._nz/2,0)))*2*pi/Lens[2];
        self._ky, self._kx, self._kz = np.meshgrid(kvy,kvx,kvz);
        # Prepare the arrays for FINUFFT
        self._fxhat = np.zeros([self._nx,self._ny,self._nz],dtype=np.complex128,order='F');
        self._fyhat = np.zeros([self._nx,self._ny,self._nz],dtype=np.complex128,order='F');
        self._fzhat = np.zeros([self._nx,self._ny,self._nz],dtype=np.complex128,order='F');
        self._fhat = np.zeros([self._nx,self._ny,self._nz,3],dtype=np.complex128,order='F');


# Parameters for Raul's code 
GPUtol = 1e-3 # single precision, so 1e-6 is smallest tolerance
class GPUEwaldSplitter(EwaldSplitter):
    
    """
    This class implements Ewald splitting using Raul's UAMMD code on the GPU. 
    It is much faster than my code 
    """
    ## ========================================
    ##          METHODS FOR INITIALIZATION
    ## ========================================
    def __init__(self,a,mu,xi,PerDom,Npts):
        """
        Constructor. Initialize the Ewald splitter. 
        Extra input variables: xi = Ewald splitting parameter,
        PerDom = PeriodicDomain object, Npts = number of blobs
        """
        self._a = float(a);
        self._mu = float(mu);
        self._Npts = Npts;

        self._xi = float(xi);
        self._PerLengths = PerDom.getPeriodicLens()
        self._currentDomain = PerDom;
        
        # Initialize C++ code (for checking rcut dynamically)
        self._RPYcpp = RPYcpp(a,mu,Npts,PerDom.getPeriodicLens());
        
        # Raul's code
        import uammd
        self._GPUParams = uammd.PSEParameters(psi=self._xi, viscosity=self._mu, hydrodynamicRadius=self._a, tolerance=GPUtol, \
            Lx=self._PerLengths[0],Ly=self._PerLengths[1],Lz=self._PerLengths[2],shearStrain=0.0);
        self._GPUEwald = uammd.UAMMD(self._GPUParams,self._Npts);
        
        # Calculate the truncation distance for Ewald
        print('%.2E GPU tolerance' %GPUtol)
    
    def NeedsGPU(self):
        return True;
        
    ## =========================================
    ##    PUBLIC METHODS CALLED OUTSIDE CLASS
    ## =========================================
    def calcBlobTotalVel(self,ptsxyz,forces,Dom,SpatialData,nThr=1):
        """
        Total velocity due to Ewald (far field + near field). 
        Inputs: ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points,
        SpatialData = SpatialDatabase object for fast neighbor computation.
        nThr = number of threads for parallel processing
        Output: the total velocity as an Npts x 3 array.
        """
        # First check if Ewald parameter is ok
        if (verbose > 0):
            thist=time.time();
        self._currentDomain = Dom;
        pts = self._currentDomain.primecoords(ptsxyz);
        self.checkrcut();
        # Reshape for GPU
        positions = np.array(np.reshape(pts,3*self._Npts), np.float64);
        forcesR = np.array(np.reshape(forces,3*self._Npts), np.float64);
        # It is really important that the result array has the same floating precision as the compiled uammd, otherwise
        # python will just silently pass by copy and the results will be lost
        MF=np.zeros(3*self._Npts, np.float64);
        self._GPUEwald.setShearStrain(self._currentDomain.getg())
        self._GPUEwald.computeHydrodynamicDisplacements(positions, forcesR,MF)
        if (np.amax(np.abs(MF))==0 and np.amax(np.abs(forcesR)) > 0):
        	raise ValueError('You are getting zero velocity with finite force, your UAMMD precision is wrong!')
        	# You need to change the precision in MF to np.float32 or float64, depending on how you compiled UAMMD
        if (verbose > 0):
            print('Method Raul time: %f' %(time.time()-thist));
        return np.array(np.reshape(MF,(self._Npts,3)),np.float64);
    
    def calcMOneHalfW(self,ptsxyz,Dom):
        """
        Total velocity due to Ewald (far field + near field). 
        Inputs: ptsxyz = the list of points 
        Output: the total velocity as an Npts x 3 array.
        """
        # First check if Ewald parameter is ok
        if (verbose > 0):
            thist=time.time();
        self._currentDomain = Dom;
        pts = self._currentDomain.primecoords(ptsxyz);
        self.checkrcut();
        # Reshape for GPU
        positions = np.array(np.reshape(pts,3*self._Npts), np.float64);
        forces = 0*positions;
        # It is really important that the result array has the same floating precision as the compiled uammd, otherwise
        # python will just silently pass by copy and the results will be lost
        MHalfW=np.zeros(3*self._Npts, np.float64);
        self._GPUEwald.setShearStrain(self._currentDomain.getg())
        self._GPUEwald.computeHydrodynamicDisplacements(positions,forces,MHalfW,temperature=0.5,prefactor = 1.0)
        return MHalfW;
    
    def updateFarFieldArrays(self):
        """
        Update GPU if the Ewald parameter has to be enlarged
        """
        print('Updating GPU far field arrays')
        import uammd
        self._GPUParams = uammd.PSEParameters(psi=self._xi, viscosity=self._mu, hydrodynamicRadius=self._a, tolerance=GPUtol, \
            Lx=self._PerLengths[0],Ly=self._PerLengths[1],Lz=self._PerLengths[2]);
        self._GPUEwald = uammd.UAMMD(self._GPUParams,self._Npts);
        
