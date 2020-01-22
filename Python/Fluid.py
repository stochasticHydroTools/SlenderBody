import numpy as np
from EwaldStuff import EwaldSplitter
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
import SpecQuadUtils as sq
import chebfcns as cf

rho_crit = 1.114; # critical Bernstein radius for special quad

class Fluid(object):
    """ 
    This is the class for the 'fluid,' in the sense of the background
    flow and nonlocal velocities
    """
    def __init__(self,Nfibs,Nperfib,nonLocal,Lx,Ly,Lz,xi,mu,omega,gam0,Lf,epsilon):
        """
        Constructor for the fluid class. 
        Inputs: Nfibs = number of fibers, Nperfib = number of points per fiber, nonLocal = 
        are we doing nonlocal hydro?, Lx, Ly, Lz = periodic lengths in each direction, 
        xi = Ewald splitting parameter, mu = fluid viscosity, omega = frequency of background
        flow oscillations, gam0 = base strain rate of the background flow, Lf = fiber length, 
        epsilon = aspect ratio of the fiber (r/L).
        """
        self._Nfib = Nfibs;
        self._Npf = Nperfib;
        self._nonLocal = nonLocal;
        self._mu = mu;
        self._omega = omega;
        self._gam0 = gam0;
        self._g = 0;
        self._Lf = Lf;
        self._epsilon = epsilon;
        self._Ew = EwaldSplitter(Lx,Ly,Lz,xi,np.sqrt(1.5)*Lf*epsilon,mu);
        # Precompute the LU factorization of the Vandermonde matrix for 32 points (for nonlocal quadrature)
        t32, _ = cf.chebpts(32,[-1,1],1);
        self._v32LUpiv = lu_factor(np.vander(t32,increasing=True).T);
    
    def setg(self,ing):
        self._g = ing;
    
    def getg(self,t):
        g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        g-=round(g);
        return g;
    
    def evalU0(self,Xin,t):
        """
        Compute the background flow on input array Xin at time t.
        Xin is assumed to be an N x 3 array with each column being
        the x, y, and z locations. 
        This method returns the background flow in the same format 
        (N x 3). It also sets the total strain in the system.
        """
        U0 = 0*Xin;
        U0[:,0]=self._gam0*np.cos(self._omega*t)*Xin[:,1];
        self._g = 1.0*self._gam0/self._omega*np.sin(self._omega*t);
        self._g-= round(self._g); # Now self._g is in [-1/2,1/2]
        return U0;

    def nLvel(self,fiblist,tint,LMMlam,fext,t):
        """
        Compute the non-local velocity due to the fibers. 
        Inputs: fiblist = list of nFib fiber objects, tint = type of temporal 
        integrator (2 for Crank-Nicolson, 1 for backward Euler, -1 for forward Euler), 
        LMMlam = whether to use linear multistep combinations for lambda as an argument
        for the non-local fluid velocity, fext = any external forcing on the fibers. 
        Outputs: non-local velocity as an (nFib x 3*N per fib) one-dimensional array.
        """
        totnum = self._Nfib*self._Npf;
        # If we're not doing non-local hydro, return just background flow and stop.
        if (not self._nonLocal):
            U0fibers = np.zeros((totnum,3));            # Background flows
            for iFib in xrange(len(fiblist)):
                fib = fiblist[iFib];
                X, Xs, fE, lam = fib.getNLargs(tint,LMMlam);
                rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
                # Background flow values
                U0fibers[rowinds,:] = self.evalU0(X,t);
            return np.reshape(U0fibers,totnum*3);
        # Establish large lists that will be used for the non-local Ewald
        ptsxyz=np.zeros((totnum,3));            # points
        ptsuni=np.zeros((16*self._Nfib,3));     # 16 uniformly distributed points
        forceDs=np.zeros((totnum,3));           # force densities
        forces = np.zeros((totnum,3));          # forces
        selfVel = np.zeros((totnum,3));         # RPY self velocity (to subtract)
        FpVel = np.zeros(totnum*3);             # Finite part integral velocities
        LocVel = np.zeros(totnum*3);            # Purely local velocities
        U0fibers = np.zeros((totnum,3));            # Background flows
        # Fill up the lists by looping through the fibers
        for iFib in xrange(len(fiblist)):
            fib = fiblist[iFib];
            # Get the arguments from the fiber class.
            X, Xs, fE, lam = fib.getNLargs(tint,LMMlam);
            rowinds = range(iFib*self._Npf,(iFib+1)*self._Npf);
            stackinds = range(iFib*3*self._Npf,(iFib+1)*3*self._Npf);
            exFib = fext[rowinds,:];
            w = np.reshape(fib.getw(),(self._Npf,1));
            ptsxyz[rowinds,:] = X;
            ptsuni[iFib*16:(iFib+1)*16,:] = fib.getUniLocs(X,16);
            forceDs[rowinds,:] = fE+lam+exFib
            #forceDs[rowinds,:] = exFib; #use to test for given external forcing
            forces[rowinds,:] = (fE+lam+exFib)*w;
            #forces[rowinds,:] = exFib*w; #use to test for given external forcing
            fdThis = forceDs[rowinds,:];
            # Compute self velocity from RPY (which will be subtracted), finite part integral
            # velocity, and purely local velocity (this is needed in the non-local corrections
            # for very close fibers.
            selfVel[rowinds,:] = fib.subtractRPY(X,(fdThis)*w);
            FpVel[stackinds] = fib.calcFPVelocity(X,Xs,fdThis);
            LocVel[stackinds] = fib.calcLocalVelocity(np.reshape(Xs,(3*self._Npf)),\
                                np.reshape(fdThis,(3*self._Npf)));
            # Background flow values
            U0fibers[rowinds,:] = self.evalU0(X,t);
        # Ewald splitting to compute the far field and near field
        ufar = self._Ew.EwaldFarVel(totnum,ptsxyz,forces,self._g); # far field Ewald
        unear = self._Ew.EwaldNearVel(totnum,ptsxyz,forces,self._g); # near field Ewald
        perEwald = ufar+unear-selfVel; # The periodized Ewald sum, with self term subtracted
        # Corrections for fibers that are close together
        # First determine which sets of points and targets need corrections
        targs, fibers, methods, shifts = self.determineQuad(ptsxyz,ptsuni);
        # Loop over the points that need corrections and do the corrections
        for iT in xrange(len(targs)):
            iPt = targs[iT];
            jFib = fibers[iT];
            rowinds = range(jFib*self._Npf,(jFib+1)*self._Npf)
            fibpts = ptsxyz[rowinds,:]+shifts[iT,:];
            ffib = forces[rowinds,:];
            fdfib = forceDs[rowinds,:];
            stackinds = range(jFib*3*self._Npf,(jFib+1)*3*self._Npf);
            centerVels = np.reshape(FpVel[stackinds]+LocVel[stackinds],(self._Npf,3));
            cvel = self.correctVel(ptsxyz[iPt,:],fiblist[jFib],fibpts,ffib,fdfib,centerVels,methods[iT]);
            perEwald[iPt,:]+=cvel; # add the correction to the value at the target
        # Return the velocity due to the other fibers + the finite part integral velocity
        return np.reshape(perEwald+U0fibers,totnum*3)+FpVel;

    def determineQuad(self,ptsxyz,ptsuni):
        """
        Determine which (target, fiber) pairs have wrong integrals when computed 
        by Ewald splitting, and decide what kind of corrective scheme to use. 
        Inputs: ptsxyz = self_Npf*self._nFib x 3 of ALL targets (this is the Chebyshev
        points), and ptsuni = (16*self._nFib) x 3 array of uniform points. This method
        uses the uniform points to determine if a fiber is close to a target. 
        Outputs: 4 arrays. Targs[i] = target number that needs correction, 
        fibers[i] = fiber number that is used for the correction at target targs[i], 
        methods[i] = method that needs to be used to do that correction (1 for direct
        with 32 points, 2 for special quadrature), shifts[i] = periodic shift in the 
        fiber so that it is actually numerically close to the target.
        """
        totnum = self._Nfib*self._Npf;
        # Sort the uniform points.
        # Figure out how many bins to put them in.
        rcutu = 0.15*1.05*self._Lf;
        nxBin, nyBin, nzBin = self._Ew.calcnBins(rcutu);
        # Bin the target points for each fiber
        cptBins = self._Ew.binsbyP(ptsxyz,nxBin,nyBin,nzBin,self._g);
        # Sort the uniform points into bins
        uFirst, uNext = self._Ew.binPoints(16*self._Nfib,ptsuni,nxBin,nyBin,nzBin,self._g);
        qtypes = sp.lil_matrix((totnum,self._Nfib)); # List of quadrature types by fiber
        xShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the x direction
        yShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the y direction
        zShifts = sp.lil_matrix((totnum,self._Nfib)); # periodic shifts in the z direction
        for iPt in xrange(totnum): # looping over the target points
            iFib = int(iPt/self._Npf+1e-12); # the i fiber number
            tbin=[cptBins[iPt,:]];
            sN = EwaldSplitter.neighborBins(tbin,nxBin,nyBin,nzBin);
            for iSn in xrange(len(sN)): # looking in the neighboring bins for uniform pts
                jPt = uFirst[sN[iSn]];
                while (jPt !=-1):
                    jFib = int(jPt/16+1e-12); # the j fiber number (FOR UNIFORM POINTS!!)
                    if (not iFib == jFib): # we already corrected for iFib==jFib.
                        # Find nearest periodic image (might need to speed this up)
                        rvec = self._Ew.calcShifted(ptsxyz[iPt,:]-ptsuni[jPt,:],self._g);
                        nr = np.sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
                        # Use the distance to determine what type of quadrature is needed
                        if (nr < self._Lf*0.06*1.20 and qtypes[iPt,jFib] < 2):
                            qtypes[iPt,jFib] = 2; # special quad needed
                            xShifts[iPt,jFib] =  ptsxyz[iPt,0]-ptsuni[jPt,0]-rvec[0]+1e-99;
                            yShifts[iPt,jFib] =  ptsxyz[iPt,1]-ptsuni[jPt,1]-rvec[1]+1e-99;
                            zShifts[iPt,jFib] =  ptsxyz[iPt,2]-ptsuni[jPt,2]-rvec[2]+1e-99;
                        elif (nr < rcutu and qtypes[iPt,jFib] < 1):
                            qtypes[iPt,jFib]=1; # direct quad with N=32
                            xShifts[iPt,jFib] =  ptsxyz[iPt,0]-ptsuni[jPt,0]-rvec[0]+1e-99;
                            yShifts[iPt,jFib] =  ptsxyz[iPt,1]-ptsuni[jPt,1]-rvec[1]+1e-99;
                            zShifts[iPt,jFib] =  ptsxyz[iPt,2]-ptsuni[jPt,2]-rvec[2]+1e-99;
                    jPt = uNext[jPt];
        # Shift data types so we can loop through the points that need correction
        qtypes = sp.coo_matrix(qtypes);
        xS = (sp.coo_matrix(xShifts)).data;
        yS = (sp.coo_matrix(yShifts)).data;
        zS = (sp.coo_matrix(zShifts)).data;
        shifts = np.concatenate(([xS],[yS],[zS])).T;
        targs = qtypes.row;
        fibers = qtypes.col;
        methods = qtypes.data;
        return targs, fibers, methods, shifts
    
    def correctVel(self,tpt,fiber,fibpts,forces,forceDs,centerVels,method):
        """
        Method to correct the velocity when fibers are close together. 
        Inputs: tpt = target point (3 array), fiber = the fiber object that
        is close to the target, fibpts = self._Npts x 3 array of the (shifted)
        fiber locations that are close to the target, forces = the forces on the 
        fiber (not force densities; these are the force densities*weights), 
        forceDs = self._Npts x 3 array of force densities on the fiber, 
        centerVels = self._nPts x 3 array of the velocity on the fiber centerline
        (for use when points get really close to the fiber), and 
        the correction method (1 for 32 point quadrature), otherwise this will 
        do special quadrature 
        Outputs: the correction to the velocity at the target as a 3 array
        """
        # Subtract the Ewald with self._Npf points on the fiber
        cvel = -fiber.RPYSBTKernel(tpt,fibpts,forces);
        # If doing N=32, upsample to N=32, do the free space quad sum, and stop
        X32 = fiber.resample(fibpts,32);
        f32 = fiber.resample(forceDs,32);
        cvel32 = fiber.resample(centerVels,32);
        w32 = fiber.newWeights(32);
        forces32 = f32*np.reshape(w32,(32,1));
        if (method==1): # free space sum for N = 32
            cvel+= fiber.RPYSBTKernel(tpt,X32,forces32,sbt=1);
            return cvel;
        # If doing special quad, find the complex root using the domain for centerline
        # t in [-1,1]
        t32, _ = cf.chebpts(32,[-1,1],1);
        troot, converged, cdist, clvel = Fluid.calcRoot(32,t32,X32,tpt,cvel32,rho_crit);
        # If not converged, point must be far, just do direct with N = 32 and stop
        if (not converged):
            cvel+= fiber.RPYSBTKernel(tpt,X32,forces32,sbt=1);
            return cvel;
        # Now we know we have converged to the root
        # How far are we from the fiber in a non-dimensional sense?
        dstar = cdist/(self._epsilon*self._Lf);
        # If the point is inside the fiber "cross section," which we define as dstar < 2.2,
        # compute the centerline velocity at the approximate closest point and return (in fact we need
        # it when cdist/(epsilon*L) < 4.4; in that case we are going to interpolate
        # How this is implemented is to set wtCL > 0 if we need to interpolate
        wtCL = 0.0;
        if (dstar < 4.4):
            wtCL = (4.4-dstar)/2.2;
        wtSBT = 1.0-wtCL;
        if (dstar < 2.2): # return the centerline velocity and stop
            print 'Target close to fiber - setting velocity = centerline velocity'
            cvel+= clvel;
            return cvel;
        # Now we are dealing with the case where we need the free space slender integral.
        # 3 possible options:
        # 1. Don't need special quad (determined by Bernstein radius)
        # 2. Special quad needed, but 1 panel of 32 is ok
        # 3. Special quad needed, but need 2 panels of 32
        # Bernstein radius
        bradius = Fluid.bernstein_radius(troot);
        sqneeded = (bradius < rho_crit);
        if (not sqneeded):
            cvel+= wtSBT*fiber.RPYSBTKernel(tpt,X32,forces32,sbt=1)+wtCL*clvel;
            return cvel;
        if (dstar > 8.8): # Ok to proceed with 1 panel of 32
            # Special quad weights
            wts = self.specialWts32(troot,t32);
            cvel+= fiber.SBTKernelSplit(tpt,X32,f32,wts[:,0],wts[:,1],wts[:,2])
            return cvel;
        # dstar < 8.8, need to redo special quad with 2 panels of 32
        # Resample at 2 panels of 32
        X2p32 = fiber.resample(fibpts,32,232);
        f2p32 = fiber.resample(forceDs,32,232);
        SBTvel = np.zeros(3);
        for iPan in xrange(2): # looping over the panels
            indpan = np.arange(32)+iPan*32;
            # Points and force densities for the panel
            Xpan = X2p32[indpan,:];
            fdpan = f2p32[indpan,:];
            # Calculate the root as before. The method will waste time
            # computing the closest point, etc, but those are trivial computations
            # and it's easier for the sake of convenience to put it all in one method.
            troot, converged, _, _ = Fluid.calcRoot(32,t32,Xpan,tpt,cvel32,rho_crit);
            bradius = Fluid.bernstein_radius(troot);
            # Do we need special quad for this panel? (Will only need for 1 panel,
            # whatever one has the points closest to the fiber).
            sqneeded = converged and (bradius < rho_crit);
            if (not sqneeded):
                # Directly do the integral for 1 panel (weights have to halved because
                # we cut the fiber in 2)
                SBTvel+=fiber.RPYSBTKernel(tpt,Xpan,fdpan*np.reshape(w32,(32,1))/2.0,sbt=1);
            else:
                # Compute special quad weights (divide weights by 2 for 2 panels)
                wts = self.specialWts32(troot,t32)/2.0;
                SBTvel+= fiber.SBTKernelSplit(tpt,Xpan,fdpan,wts[:,0],wts[:,1],wts[:,2]);
        cvel+= wtSBT*SBTvel+wtCL*clvel;
        return cvel;
    
    @staticmethod
    def calcRoot(N,s,X,tpt,centerVels,rho_crit):
        """
        Compute the root troot using special quadrature. 
        Inputs: N = number of points on the fiber N, s = N array of quadrature
        nodes on [-1,1], X = N x 3 array of fiber points, tpt = 3 array target 
        point where we seek the velocity due to the fiber, centerVels = N x 3 
        array of velocities on the fiber centerline (for close points), 
        rho_crit = critical Bernstein radius. 
        Outputs: troot = complex root for special quad, converged = 1 if we actually
        converged to that root, 0 if we didn't (or if the initial guess was so far that
        we didn't need to), cdist = approximate distance from the fiber centerline 
        to the point, clvel_close = 3 array of velocity at the closest centerline point
        """
        tinit = sq.rf_iguess(s,X[:,0],X[:,1],X[:,2],N,tpt[0],tpt[1],tpt[2]);
        # If initial guess is too far, return so we can do direct with N = 32
        if (Fluid.bernstein_radius(tinit) > 1.5*rho_crit):
            converged=0;
            troot=0+0j;
            cdist = 1000; # a big number
            return troot, converged, cdist, np.zeros(3);
        # Compute the Chebyshev coefficients and the coefficients of the derivative
        tcos = cf.coeffs(N,cf.th1(N),X);
        dcos = cf.coeffDiff(tcos,N);
        # From Ludvig: cap the expansion at 16 coefficients for root finding (this seems
        # to be more stable near the boundary). Use this expansion to compute the root
        # (C++ code). Returns a tuple with elements (troot, converged).
        trconv = sq.rootfinder(tcos[:16,0],tcos[:16,1],tcos[:16,2],dcos[:16,0],dcos[:16,1],dcos[:16,2],\
                                  16, tpt[0], tpt[1], tpt[2],tinit);
        troot = trconv[0];
        converged = trconv[1];
        # Use troot to estimate the distance from the fiber
        tapprox = troot;
        if (np.real(troot) < -1): # root is off the fiber centerline in tangental dir
            tapprox = -1.0+1j*np.imag(tapprox);
        elif (np.real(troot) > 1): # root is off the fiber centerline in tangental dir
            tapprox = 1.0+1j*np.imag(tapprox);
        # Evaluate the Chebyshev series at tapprox
        cdist = np.linalg.norm(np.real(cf.evalSeries(tcos,np.arccos(tapprox),N))-tpt);
        # Evaluate centerline velocity at tapprox
        clv_cos = cf.coeffs(N,cf.th1(N),centerVels);
        clvel_close = np.reshape(cf.evalSeries(clv_cos,np.arccos(np.real(tapprox)),N),3);
        return troot, converged, cdist, clvel_close
    
    def specialWts32(self,troot,t32):
        """
        Weights for the special quadrature scheme. Because this scheme is only called 
        for N = 32, and because the LU factorization of the Vandermonde matrix has 
        been precomputed only for N = 32, there are assumed to be 32 points. 
        Inputs: complex root troot, and t32 = 32 array of quadrature nodes. 
        Outputs: the quadrature weights as a 32 x 3 array. The first column has the weights
        for the R^-1 integral, second column R^-3, third column R^-5.
        """
        # Call the C++ code to get the integrals for the root
        wall = np.reshape(sq.spec_ints(troot,32),(3,32)).T;
        # Solve with the precomputed LU factorization
        seriescos = lu_solve(self._v32LUpiv,wall);
        dpows = np.reshape(np.concatenate((abs(t32-troot),abs(t32-troot)**3,abs(t32-troot)**5)),(3,32)).T;
        # Rescale weights (which come back on [-1,1]) by multiplying by Lf/2.0
        wts = seriescos*dpows*self._Lf/2.0;
        return wts;

    @staticmethod
    def bernstein_radius(z):
        return np.abs(z + np.sqrt(z - 1.0)*np.sqrt(z+1.0));
