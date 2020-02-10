import finufftpy as fi
import numpy as np
import EwaldUtils as ewc
from math import pi

nearcut = 1e-3; # cutoff for near field interactions
fartol = 1e-10; # far field tolerance for FINUFFT (artificially low rn to compare w Matlab)

class EwaldSplitter(object):
    """
    This class handles the Ewald splitting for the calculation of
    the non-local velocity.
    """
    
    ## METHODS FOR INITIALIZATION
    def __init__(self,xi,a,mu):
        """
        Constructor. Initialize the Ewald splitter. 
        Input variables: Dom = domain object on which the Ewald is being done (this is the
        same object that is inside fiberCollection.py)
        xi = Ewald splitting parameter, a = hydrodynamic radius of the RPY blobs 
        (a = sqrt(3/2)*epsilon*L), mu = fluid viscosity
        """
        self._xi = xi;
        self._a = a;
        self._mu = mu;
        # Calculate the truncation distance for Ewald
        self._calcrcut();

    ## "PUBLIC" METHODS (NEEDED EXTERNALLY)
    def EwaldTotalVel(self,Npts,ptsxyz,forces,Dom,treeofPoints):
        """
        Total velocity due to Ewald (far field + near field). 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points,
        treeofPoints = ckD tree object for the near field. 
        Output: the total velocity as an Npts x 3 array.
        """
        # First check if Ewald parameter is ok
        self._checkrcut(Dom);
        Ewaldfar = self._EwaldFarVel(Npts,ptsxyz,forces,Dom); # far field Ewald
        Ewaldnear = self._EwaldNearVelkD(Npts,ptsxyz,Dom,treeofPoints,forces); # near field Ewald
        #Ewaldnear2 = self._EwaldNearVelQuad(Npts,ptsxyz,forces,Dom); # near field Ewald quadratic
        #print 'Near field error'
        #print np.amax(Ewaldnear-Ewaldnear2)
        return Ewaldfar+Ewaldnear; 

    ## "PRIVATE" METHODS (ONLY CALLED WITHIN THE CLASS)
    def _EwaldFarVel(self,Npts,ptsxyz,forces,Dom):
        """
        This function computes the far field Ewald velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points.
        This function relies entirely on calls to FINUFFT. See the documentation
        there for more information.
        """
        # Compute the coordinates in the transformed basis
        pts = Dom.primecoords(ptsxyz);
        g = Dom.getg();
        # Rescale to [-pi,pi] (for FINUFFT)
        Lens = Dom.getLens();
        pts = 2*pi*np.mod(pts,Lens)/Lens-pi;
        # Estimate the required grid
        gw=1.0/(2*self._xi);
        h = gw/1.6;
        nx, ny, nz = 2**(np.ceil(np.log2(Lens/h)));
        nx = int(nx); ny=int(ny); nz=int(nz);
        # Wave numbers (FFTW ordering)
        kvx = np.concatenate((np.arange(nx/2),np.arange(-nx/2,0)))*2*pi/Lens[0];
        kvy = np.concatenate((np.arange(ny/2),np.arange(-ny/2,0)))*2*pi/Lens[1];
        kvz = np.concatenate((np.arange(nz/2),np.arange(-nz/2,0)))*2*pi/Lens[2];
        ky, kx, kz = np.meshgrid(kvy,kvx,kvz);
        # Prepare the arrays for FINUFFT - this should be done during initialization I think
        fxhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        fyhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        fzhat = np.zeros([nx,ny,nz],dtype=np.complex128,order='F');
        # Forcing on the grid (FINUFFT type 1)
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,0],-1,fartol,\
                    nx,ny,nz,fxhat,modeord=1);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,1],-1,fartol,\
                    nx,ny,nz,fyhat,modeord=1);
        fi.nufft3d1(pts[:,0],pts[:,1],pts[:,2],forces[:,2],-1,fartol,\
                    nx,ny,nz,fzhat,modeord=1);
        # Manipulation in Fourier space
        k = np.sqrt(kx*kx+(ky-g*kx)*(ky-g*kx)+kz*kz);
        # Multiplication factor for the RPY tensor
        factor = 1.0/(self._mu*k*k)*np.sinc(k*self._a/pi)**2;
        factor *= (1+k*k/(4*self._xi*self._xi))*np.exp(-k*k/(4*self._xi*self._xi)); # splitting function
        factor[0,0,0] = 0; # zero out 0 mode
        uxhat = factor * fxhat;
        uyhat = factor * fyhat;
        uzhat = factor * fzhat;
        # Project off so we get divergence free
        uprojx = uxhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*kx/(k*k);
        uprojx[0,0,0]=0;
        uprojy = uyhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*(ky-g*kx)/(k*k);
        uprojy[0,0,0]=0;
        uprojz = uzhat-(kx*uxhat+(ky-g*kx)*uyhat+kz*uzhat)*kz/(k*k);
        uprojz[0,0,0]=0;
        ux = np.zeros([Npts],dtype=np.complex128);
        uy = np.zeros([Npts],dtype=np.complex128);
        uz = np.zeros([Npts],dtype=np.complex128);
        # Velocities at the points (FINUFFT type 2)
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],ux,1,fartol,uprojx,modeord=1);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],uy,1,fartol,uprojy,modeord=1);
        fi.nufft3d2(pts[:,0],pts[:,1],pts[:,2],uz,1,fartol,uprojz,modeord=1);
        vol = Dom.getVol();
        ux = np.real(ux)/vol;
        uy = np.real(uy)/vol;
        uz = np.real(uz)/vol;
        return np.concatenate(([ux],[uy],[uz])).T;
    
    def _EwaldNearVelkD(self,Npts,ptsxyz,Dom,treeOfPoints,forces):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points,
        treeOfPoints = ckD tree of the points.
        This method uses the kD tree to compute a list of the neighbors.
        Output: the near field velocity
        """
        # Find all pairs (returns an array of the pairs) within rcut
        #import time
        #t=time.time();
        # print 'Doing Ewald with g=%f' %Dom.getg();
        pairpts = Dom.kDTreeNeighbors(treeOfPoints, self._rcut);
        Npairs = len(pairpts);
        #print 'Time to find pairs %f' %(time.time()-t);
        # Call the C+ function which takes as input the pairs of points, number of points and gives
        # you the near field
        Lens = Dom.getLens();
        g = Dom.getg();
        velNear = ewc.RPYNKerPairs(Npts,Npairs,pairpts[:,0],pairpts[:,1],ptsxyz[:,0],ptsxyz[:,1],ptsxyz[:,2],\
                forces[:,0],forces[:,1],forces[:,2],self._mu,self._xi,self._a,Lens[0],Lens[1],Lens[2],\
                g, self._rcut);
        return np.reshape(velNear,(Npts,3));   

    def _calcrcut(self):
        """
        Calculate the truncation distance for the Ewald near field. 
        We truncate the near field at the value nearcut.
        """
        rcut=0;          # determine rcut
        rcutstep = 1e-2; # accuracy of rcut
        Vatcut = abs(np.array(ewc.RPYNKer([rcut,0,0],[1,0,0],self._mu,self._xi,self._a)));
        V0 = min(np.amax(Vatcut),1); # M0 is relative to the value at 0 or 1, whichever is smaller
        while (np.amax(Vatcut)/V0 > nearcut):
            rcut=rcut+rcutstep;
            Vatcut = abs(np.array(ewc.RPYNKer([rcut,0,0],[1,0,0],self._mu,self._xi,self._a)));
        self._rcut =  rcut;
        print 'Ewald cut %f' %self._rcut
    
    def _checkrcut(self, Dom):
        """
        Check if rcut is less than one half period dynamically (the absolute
        length of a half period varies as the strain g changes). 
        If rcut is less than a half period, increase xi until rcut is less than a half
        period.
        """
        Lmin = Dom.getminLen()/Dom.safetyfactor();
        #print 'Safety distance is %f' %(Lmin*0.5)
        vLover2 = np.amax(ewc.RPYNKer([Lmin*0.5,0,0],[1,0,0],self._mu,self._xi,self._a));
        if (vLover2 > nearcut):
            print 'Need to increase xi or L, there are near interactions w/ more than 1 image';
        while (vLover2 > nearcut):
            # Modify xi
            self._xi+=0.1;
            self._calcrcut();
            vLover2 = np.amax(ewc.RPYNKer([Lmin*0.5,0,0],[1,0,0],self._mu,self._xi,self._a));
            print 'The new value of xi is %f' %self._xi
            print 'The new rcut %f' %self._rcut

    ## SLOW METHODS FOR CHECKING FAST METHODS
    def _EwaldNearVelQuad(self,Npts,pts,forces,Dom):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points.
        This is the dumb quadratic method.
        """
        velNear = np.zeros((pts.T).shape);
        # Go point by point to compute the velocity. Can parallelize this
        # outer loop as necessary.
        for iPt in xrange(Npts): # loop over points
            for jPt in xrange(Npts):
                # Find nearest periodic image (might need to speed this up)
                rvec = Dom.calcShifted(pts[iPt,:]-pts[jPt,:]);
                # Only actually do the computation when necessary
                if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+\
                    rvec[2]*rvec[2] < self._rcut*self._rcut):
                    velNear[:,iPt]+=ewc.RPYNKer(rvec,forces[jPt,:],self._mu,self._xi,self._a);
        return velNear.T;
    
    def _EwaldNearVelBins(self,Npts,pts,forces,Dom):
        """
        Near field velocity. 
        Inputs: Npts = the number of blobs, ptsxyz = the list of points 
        in undeformed, Cartesian coordinates, forces = forces at those points.
        This version uses binning and is SLOW!
        """
        print 'Warning - this near field function is based on bins and is SLOW!'
        velNear = np.zeros((pts.T).shape);
        # Sort points into bins
        nxBin, nyBin, nzBin = Dom.calcnBins(self._rcut);
        ptbins = Dom.binsbyP(pts,nxBin,nyBin,nzBin);
        print 'Number of bins (%d, %d, %d)' %(nxBin, nyBin, nzBin);
        bfirst, pnext = Dom.binPoints(Npts,pts,nxBin,nyBin,nzBin);
        # Go point by point to compute the velocity. Can parallelize this
        # outer loop as necessary.
        for iPt in xrange(Npts): # loop over points
            tbin=[ptbins[iPt,:]];
            neighBins = Dom.neighborBins(tbin,nxBin,nyBin,nzBin);
            for iSn in xrange(len(neighBins)): # loop over neighboring bins
                jPt = bfirst[neighBins[iSn]];
                while (jPt !=-1):
                    # Find nearest periodic image (might need to speed this up)
                    rvec = Dom.calcShifted(pts[iPt,:]-pts[jPt,:]);
                    # Only actually do the computation when necessary
                    if (rvec[0]*rvec[0]+rvec[1]*rvec[1]+\
                        rvec[2]*rvec[2] < self._rcut*self._rcut):
                        velNear[:,iPt]+=ewc.RPYNKer(rvec,forces[jPt,:],self._mu,self._xi,self._a);
                    jPt = pnext[jPt];
        return velNear.T;

    # Python versions of functions - not being used as these functions are being
    # done in C++
    def RPYNear(self,rvec):
        """
        #Evaluate the near field RPY kernel.
        #Input: the vector rvec to evaluate the kernel at
        #Output: value of kernel.
        #This function relies on the C++ functions Fnear
        #and Gnear, which evaluate the complicated near field \
        #kernels faster than Python.
        """
        rvec = np.reshape(rvec,(1,3));
        r=np.sqrt(rvec[:,0]*rvec[:,0]+rvec[:,1]*rvec[:,1]+rvec[:,2]*rvec[:,2]);
        rhat=rvec/r;
        rhat[np.isnan(rhat)]=0;
        RR = np.dot(rhat.T,rhat);
        return 1.0/(6*pi*self._mu*self._a)*(ewc.Fnear(r,self._xi,self._a)*\
                (np.identity(3)-RR)+ewc.Gnear(r,self._xi,self._a)*RR);
     
    
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
        


