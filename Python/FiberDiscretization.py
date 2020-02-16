import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import EwaldUtils as ewc
import FiberUtils as fc
import EwaldNumba as ewnb
from math import sqrt
from scipy.linalg import lu_factor, lu_solve

# Definitions
chebGridType = 1; # always use a type 1 grid for fiber discretization
D4BCgridType = 2; # always a type 2 grid to enforce the BCs
numBCs = 4; # number of boundary conditions
# Numbers of uniform/upsampled points
nptsUpsample = 32; # number of points to upsample to for special quadrature
nptsUniform = 16; # number of uniform points to estimate distance

class FiberDiscretization(object):
    """ 
    The object code for a fiber discretization. This is the only
    class that knows about Chebyshev.
    """
    def __init__(self, L, epsilon,ellipsoidal=0,Eb=1,mu=1,N=16):
        """
        Constructor. Object variables are:
        L = fiber length, epsilon = fiber aspect ratio, ellipsoidal = shape of fibers (1 for
        ellipsoidal, 0 for cylindrical), Eb = bending stiffness, mu = fluid viscosity. 
        N = number of points on discretized fiber.
        """
        self._L = L;
        self._epsilon = epsilon;
        self._ellipsoidal = ellipsoidal;
        self._Eb = Eb;
        self._mu = mu;
        self._N = N;
        self._nptsUpsample = nptsUpsample
        self._nptsUniform = nptsUniform;
        # Chebyshev grid and weights
        self._s = cf.chebPts(self._N,[0,self._L],chebGridType);
        self._w = cf.chebWts(self._N,[0,self._L],chebGridType);
        # Chebyshev initializations
        self._Lmat = cf.CoeffstoValuesMatrix(self._N,self._N,chebGridType);
        self.initDiffMatrices();
        self.initIs();
        self.initD4BC();
        self.initFPMatrix();
        self.initResamplingMatrices();
        self.initSpecialQuadMatrices();
        self.initLocalcvals();

    ## METHODS FOR INITIALIZATION AND PRECOMPUTATION
    # Donev:Is this a method of this class that clients/callers ever use
    # If it is only called in __init__ then how about indicating somehow it is a private 
    # function, maybe with underscore in the name.
    # As I explained it is important to make it clear what method is public
    # so that if you implement this with Legendre it is clear which functions need to be written
    def initIs(self):
        """
        Initialize the identity matrix and matrix that takes 
        definite integrals (these are constant throughout a simulation)
        """
        self._I = np.zeros((3*self._N,3));
        self._wIt = np.zeros((3,3*self._N));
        for j in range(self._N):
            self._I[3*j:3*j+3,:]=np.identity(3);
            self._wIt[:,3*j:3*j+3]=np.identity(3)*self._w[j];
        
    def initD4BC(self):
        """
        Compute the operator D_4^BC that is necessary when computing 
        the bending forces. The constant -E_bend is included in 
        this operator
        """
        DownsamplingMat = cf.ResamplingMatrix(self._N,self._N+numBCs,chebGridType,D4BCgridType);
        SecDMat_upgrid = cf.diffMat(2,[0,self._L],2,self._N+numBCs,D4BCgridType,D4BCgridType);
        ThirDMat_upgrid = cf.diffMat(3,[0,self._L],2,self._N+numBCs,D4BCgridType,D4BCgridType);
        TotalBCMatrix = np.concatenate((DownsamplingMat,SecDMat_upgrid,ThirDMat_upgrid));
        # This is the only place where you would modify the "free fiber" BCs
        RHS = np.concatenate((np.identity(self._N),np.zeros((4,self._N))));
        TildeConfigMatrix = np.linalg.solve(TotalBCMatrix,RHS);
        FourDMat_fromUptoDwn = cf.diffMat(4,[0,self._L],self._N,self._N+numBCs,chebGridType,D4BCgridType);
        OneD4BC = -self._Eb*np.dot(FourDMat_fromUptoDwn,TildeConfigMatrix);
        # Stack up OneD4BC three times since there are 3 coordinates
        self._D4BC = np.zeros((3*self._N,3*self._N));
        self._D4BC[0::3,0::3]=OneD4BC;
        self._D4BC[1::3,1::3]=OneD4BC;
        self._D4BC[2::3,2::3]=OneD4BC;

    def initFPMatrix(self):
        """
        Initialize the matrix for the finite part integral. 
        Uses the adjoint method of Anna Karin Tornberg. 
        """
        if (self._N > 40):
            raise ValueError('The finite part integral is not well conditioned with > 40 pts,\
                                loss of accuracy expected');
        k = np.arange(self._N);
        s_dim1scaled = -1+2*self._s/self._L
        s_scaled = np.reshape(s_dim1scaled,(self._N,1)); # Rescale everything to [-1,1];
        q=(1.0+(-1.0)**(k+1.0)-2*s_scaled**(k+1.0))/(k+1.0);
        VanderMat = np.vander(s_dim1scaled,increasing=True);
        self._FPMatrix = np.linalg.solve(VanderMat.T,q.T);
        
    def initResamplingMatrices(self):
        """
        Pre-compute resampling matrices that go from N points to 2N points and vice versa.
        """
        self._MatfromNto2N = cf.ResamplingMatrix(2*self._N,self._N,chebGridType,chebGridType);
        self._Matfrom2NtoN = cf.ResamplingMatrix(self._N,2*self._N,chebGridType,chebGridType);

    def initSpecialQuadMatrices(self):
        """
        Initialize matrices that are necessary for special quadrature / resampling.
        """
        self._MatfromNtoUpsamp = cf.ResamplingMatrix(self._nptsUpsample,self._N,chebGridType,chebGridType);
        self._MatfromNto2panUp = cf.ResamplingMatrix(self._nptsUpsample,self._N,chebGridType,chebGridType,\
                    nPantarg=2);
        self._MatfromNtoUniform = cf.ResamplingMatrix(self._nptsUniform,self._N,'u',chebGridType);
        self._UpsampledCoefficientsMatrix = cf.CoeffstoValuesMatrix(self._nptsUpsample,self._nptsUpsample,chebGridType);
        # Initialize LU factorization of vandermonde matrix for special quadrature
        self._specQuadNodes = cf.chebPts(self._nptsUpsample,[-1,1],chebGridType);
        self._upsampledWeights = cf.chebWts(self._nptsUpsample,[0, self._L], chebGridType);
        self._NupsampLUpiv = lu_factor(np.vander(self._specQuadNodes,increasing=True).T);        

    def initDiffMatrices(self):
        """
        Pre-compute differentiation matrices on the N and 2N grid, as well
        as the pseudo-inverse on the 2N grid.
        """
        self._Dmat = cf.diffMat(1,[0,self._L],self._N,self._N,chebGridType,chebGridType);
        self._Dmat2N = cf.diffMat(1,[0,self._L],2*self._N,2*self._N,chebGridType,chebGridType);
        self._Dpinv2N = np.linalg.pinv(self._Dmat2N);
    
    def initLocalcvals(self):
        """
        Initialize local leading order coefficients for the local drag matrix M. 
        For ellipsoidal fibers these coefficents are constant = -log(epsilon^2), but
        for cylindrical fibers they vary along the fibers.
        """
        nodes_rescaled = self._s*2/self._L-1;
        aeps = 2*self._epsilon;
        ccyl = np.log((2*(1-nodes_rescaled**2)+2*np.sqrt((1-nodes_rescaled**2)**2+\
            4*aeps**2))/aeps**2);
        self._leadordercs = ccyl*(1-self._ellipsoidal)-np.log(self._epsilon**2)*self._ellipsoidal;
        self._matlist = [None]*self._N; # allocate memory for sparse matrix

    ## METHODS FOR RESAMPLING AND ACCESS BY OTHER CLASSES
    def resampleUniform(self,Xarg):
        """ 
        Get the locations of uniformly distributed nodes on the fiber.
        Xarg = the fiber coordinates (self._N x 3 vector).
        The number of uniform points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNtoUniform,Xarg);

    def getNumUpsample(self):
        return self._nptsUpsample;

    def getNumUniform(self):
        return self._nptsUniform;
    
    def upsampleGlobally(self,Xarg):
        """
        Get the locations on some upsampled nodes on the fiber. 
        Xarg = the coordinates (self._N x 3 vector). 
        The number of updampled points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNtoUpsamp,Xarg);
    
    # Donev: Somehow the hard-wiring of 2 panels (why not 4?) seems wrong to me
    # Maybe the number of panels can be an argument instead?
    def upsample2Panels(self,Xarg):
        """
        Get the locations on some 2 panels of upsampled nodes on the fiber.
        Xarg = the coordinates (self._N x 3 vector). 
        The number of updampled points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNto2panUp,Xarg);

    def upsampledCoefficients(self,Xarg):
        """
        Get the coefficients of an upsampled representation of the fiber.
        Inputs: Xarg = upsampled fiber representation
        Outputs: the coefficients and derivative coefficients of the upsampled
        representation
        """
        Xcoeffs= np.linalg.solve(self._UpsampledCoefficientsMatrix,Xarg);
        Xprimecoeffs = cf.diffCoefficients(Xcoeffs,self._nptsUpsample);
        return Xcoeffs,Xprimecoeffs;

    def evaluatePosition(self,tapprox,coeffs):
        """
        Evaluate the Chebyshev series representing the fiber 
        at a value tapprox.
        Inputs: tapprox = value to evaluate at. This value must be
        rescaled so that the centerline is parameterized by t in [-1,1],
        coeffs = coefficients that represent the fiber centerline 
        (arbitrary number)
        """
        return cf.evalSeries(coeffs,np.arccos(tapprox));
    
    def getSpecialQuadNodes(self):
        return self._specQuadNodes;
    
    def getUpsampledWeights(self):
        return self._upsampledWeights;

    def specialWeightsfromIntegrals(self,integrals,troot):
        """
        Compute the special quad weights from the integrals involving the 
        root troot. 
        Input: integrals = the values of the integrals, troot = complex root,
        where the centerline in this basis is on [-1,1]
        Output: the weights for special quadrature.
        """
        # Solve with the precomputed LU factorization
        seriescos = lu_solve(self._NupsampLUpiv,integrals);
        distance_pows = np.reshape(np.concatenate((abs(self._specQuadNodes-troot),\
                abs(self._specQuadNodes-troot)**3,abs(self._specQuadNodes-troot)**5)),\
                (3,len(self._specQuadNodes))).T;
        # Rescale weights (which come back on [-1,1]) by multiplying by Lf/2.0
        special_wts = seriescos*distance_pows*self._L/2.0;
        return special_wts;
    
    def resample(self,Xarg,Nrs,typetarg=chebGridType):
        """
        Resample the fiber at Nrs Chebyshev nodes of type type
        Inputs: Xarg = a self._N x 3 vector of the fiber points,
        Nrs = number of points where the resampling is desired. typetarg = type
        of Chebyshev points (defaults to 1)
        Outputs: the resampled locations as an Nrs x 3 vector of points. 
        """
        RMatrix = cf.ResamplingMatrix(Nrs,self._N,typetarg,chebGridType);
        Xrs = np.dot(RMatrix,Xarg);
        return Xrs;
    
    def newNodes(self,Nrs,typetarg=chebGridType, numPanels=1):
        """
        New quadrature weights for a set of Chebyshev nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new weights wN
        """
        sNew = cf.chebPts(Nrs,[0,self._L],typetarg,numPanels);
        return sNew;
    
    def newWeights(self,Nrs,typetarg=chebGridType):
        """
        New quadrature weights for a set of Chebyshev nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new weights wN
        """
        wNew = cf.chebWts(Nrs,[0,self._L],typetarg);
        return wNew;
        
    def getN(self):
        return self._N;

    def gets(self):
        return self._s;
    
    def getw(self):
        return self._w;
    
    def getepsilonL(self):
        return self._epsilon, self._L;

    def calcfE(self,X):
        """
        Calculates the bending force fE
        Inputs: position X. 
        Outpts: bending forces fE
        """
        return np.dot(self._D4BC,X);

    ## METHODS FOR FIBER EVOLUTION
    def calcKs(self,Xs):
        """
        Computes the matrix K(X). The only input is X_s, which
        has to be input as a 3N one-dimensional array.
        From X_s this method computes the normals and then 
        the matrix K.   
        """
        XsUpsampled = np.dot(self._MatfromNto2N,np.reshape(Xs,(self._N,3)));
        theta, phi, r = FiberDiscretization.cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
        n1x = -np.sin(theta);
        n1y = np.cos(theta);
        n1z = np.zeros(2*self._N);
        n2x = -np.cos(theta)*np.sin(phi);
        n2y = -np.sin(theta)*np.sin(phi);
        n2z = np.cos(phi);
        K = np.zeros((3*self._N,2*self._N-2));
        K[0::3,0:self._N-1]= self.deAliasIntegral(n1x);
        K[1::3,0:self._N-1]= self.deAliasIntegral(n1y);
        K[2::3,0:self._N-1]= self.deAliasIntegral(n1z);
        K[0::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2x);
        K[1::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2y);
        K[2::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2z);
        Kt = K.copy();
        Kt[0::3,:]*=np.reshape(self._w,(self._N,1));
        Kt[1::3,:]*=np.reshape(self._w,(self._N,1));
        Kt[2::3,:]*=np.reshape(self._w,(self._N,1));
        Kt = Kt.T;
        return K, Kt;
    
    def deAliasIntegral(self,f):
        """
        Method to dealias the product of f (an N array) with each of the first
        N-1 Chebyshev polynomials.
        Upsample to a 2N grid, perform multiplication and integration, then 
        downsample to an N point grid.
        """
        # Upsample the multiplication of f with Chebyshev polys for anti-aliasing
        UpSampMulti = np.reshape(f,(2*self._N,1)) \
            *np.dot(self._MatfromNto2N,self._Lmat[:,:self._N-1]);
        # Integrals on the original grid (integrate on upsampled grid and downsample)
        OGIntegrals = np.dot(self._Matfrom2NtoN,np.dot(self._Dpinv2N,UpSampMulti));
        return OGIntegrals;

    def calcM(self,Xs):
        """
        Calculates the local drag matrix M. The only input is X_s which
        has to be input as a 3N one-dimensional array.
        From X_s this method computes the matrix M (changes for 
        ellipsoidal vs. cylindrical fibers).
        """
        for j in range(self._N):
            v = Xs[j*3:j*3+3];
            XsXs = np.outer(v,v);
            self._matlist[j]=1/(8*np.pi*self._mu)*\
             (self._leadordercs[j]*(np.identity(3)+XsXs)+ \
              np.identity(3)-3*XsXs);
        Mloc =sp.block_diag(*self._matlist);
        return Mloc;

    def alphaLambdaSolve(self,Xarg,Xsarg,dt,impco,nLvel,exF):
        """
        This method solves the linear system for 
        lambda and alpha for a given RHS. 
        Inputs: Xarg and Xsarg = X and Xs to build the matrices for
        dt = timestep, tint = type of temporal integrator (2 for
        Crank-Nicolson, 1 for backward Euler, -1 for forward Euler), 
        nLvel = non-local velocity as a 3N vector, exF = external forces as a 3N vector
        """
        M = self.calcM(Xsarg);
        K, Kt = self.calcKs(Xsarg);
        # Schur complement solve
        B = np.concatenate((K-impco*dt*np.dot(M,np.dot(self._D4BC,K)),\
         self._I-impco*dt*np.dot(M,np.dot(self._D4BC,self._I))),axis=1);
        C = np.concatenate((Kt,self._wIt));
        D1 = np.zeros((2*self._N-2,2*self._N+1));
        D2 = impco*dt*np.dot(self._wIt, np.dot(self._D4BC,K));
        D3 = impco*dt*np.dot(self._wIt, np.dot(self._D4BC,self._I));
        D = np.concatenate((D1,np.concatenate((D2,D3),axis=1)));
        fE = self.calcfE(Xarg);
        RHS = np.dot(C,fE+exF)+np.concatenate((np.zeros(2*self._N-2),\
            -np.dot(self._wIt,fE)))+np.dot(C,np.linalg.solve(M,nLvel));
        S = np.dot(C,np.linalg.solve(M,B))+D;
        alphaU,res,rank,s = np.linalg.lstsq(S,RHS,-1);
        vel = np.dot(K,alphaU[0:2*self._N-2])+\
            np.dot(self._I,alphaU[2*self._N-2:2*self._N+1]);
        lambdas = np.linalg.solve(M,vel-nLvel)-fE-exF- \
                impco*dt*np.dot(self._D4BC,vel);
        return alphaU[:2*self._N-2], vel, lambdas;
                
    def XFromXs(self, XsNow, XsOneHalf, alpha, dt):
        """
        Compute the new tangent vectors. 
        Inputs: XsNow = tangent vectors to rotate (N x 3 array), 
        XsOneHalf = tangent vectors to use for evaluation of the 
        angular velocity Omega (also an N x 3 array), alphaU = alphas
        from the fiber that are used to determine Omega (a 2N-2 array),
        dt = timestep
        Outputs: the new positions and tangent vectors X and Xsp1
        """
        N = self._N; # save writing
        # Compute Omega on the upsampled grid
        XsUpsampled = np.dot(self._MatfromNto2N,np.reshape(XsOneHalf,(self._N,3)));
        theta, phi, r = FiberDiscretization.cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
        n1 = np.concatenate(([-np.sin(theta)],[np.cos(theta)],[np.zeros(2*N)])).T;
        n2 = np.concatenate(([-np.cos(theta)*np.sin(phi)],[-np.sin(theta)*np.sin(phi)],[np.cos(phi)])).T;
        ChebPolys = self._Lmat[:,:N-1];
        g1 = np.reshape(np.dot(self._MatfromNto2N,np.dot(ChebPolys,alpha[0:N-1])),(2*N,1));
        g2 = np.reshape(np.dot(self._MatfromNto2N,np.dot(ChebPolys,alpha[N-1:2*N-2])),(2*N,1));
        Omega = g1*n2-g2*n1;
        # Downsample Omega
        Omega = np.dot(self._Matfrom2NtoN,Omega);
        nOm = np.sqrt(np.sum(Omega*Omega,axis=1));
        k= Omega / np.reshape(nOm,(N,1));
        k[nOm < 1e-6,:]=0;
        nOm = np.reshape(nOm,(N,1));
        Xsrs = np.reshape(XsNow,(N,3));
        # Rodriguez rotation
        Xsp1 = Xsrs*np.cos(nOm*dt)+np.cross(k,Xsrs)*np.sin(nOm*dt)+ \
            k*np.reshape(np.sum(k*Xsrs,axis=1),(N,1))*(1-np.cos(nOm*dt));
        Xshat = np.linalg.solve(self._Lmat,Xsp1); # coefficients
        Xhat = cf.intCoefficients(Xshat,self._N,[0,self._L]); # one integration
        X = np.dot(self._Lmat,Xhat); # back to real space
        return X, Xsp1;
    
    ## METHODS FOR EVALUATION OF NON-LOCAL VELOCITY TERMS
    def calcFPVelocity(self,Xarg,Xsarg,forceDs):
        """ 
        Calculate the velocity due to the finite part integral. 
        Inputs: Xarg = N x 3 array of the positions X, 
        Xsarg = N x 3 array of the tangent vectors, 
        forceDs = N x 3 array of the force densities. In general it
        is better to have these as inputs rather than object variables as
        it can get messy with the LMMs and different non-local velocities.
        Outpts: the velocity due to the finite part integral as a 3N one-dimensional vector
        """
        FPvel = np.zeros((3,self._N));
        Xss = np.dot(self._Dmat,Xsarg);
        fprime = np.dot(self._Dmat,forceDs);
        for iPt in range(self._N):
            # Call the C++ function to compute the correct density,
            # 1/(s[j]-s[i])*((I+Rhat*Rhat)*f[j]*abs(s[j]-s[i])/R - (I+Xs*Xs)*f[i])
            gloc = fc.FPDensity(self._N,Xarg[:,0],Xarg[:,1],Xarg[:,2],Xsarg[iPt,:],Xss[iPt,:],fprime[iPt,:],\
                    forceDs[:,0],forceDs[:,1],forceDs[:,2],self._s,iPt);
            gloc = np.reshape(gloc,(self._N,3)).T;
            FPvel[:,iPt]=0.5*self._L*np.dot(gloc,self._FPMatrix[:,iPt]);
        return 1.0/(8.0*np.pi*self._mu)*np.reshape(FPvel.T,3*self._N);
    
    def calcLocalVelocity(self,Xsarg,forceDs):
        """
        Purely local velocity = M*f. 
        Inputs: tangent vectors Xs as a 3N one-dimensional vector, and 
        forceDs as a 3N one-dimensional vector
        Outputs: 3N one-d vector of local velocities.
        """
        M = self.calcM(Xsarg);
        return np.dot(M,forceDs);
    
    def subtractRPY(self,Xarg,forces):
        """
        Method to subtract the SELF RPY kernel. 
        Inputs: Xarg (N x 3 vector of the fiber points - 
        this is a linear combination of X and Xprev depending 
        on the temporal integrator), and the forces, also as an N x 3 vector.
        Outpts: the velocity due to N blobs (self RPY kernel) in FREE SPACE
        as an N x 3 vector
        For this one the cpp code is slightly faster, but this is not a large 
        computation so leaving the numba for now.
        """
        a = np.sqrt(3.0/2.0)*self._epsilon*self._L;
        # Python version:
        #import time
        #t=time.time();
        subVel_Py = ewnb.RPYSBTK(self._N,Xarg,self._N,Xarg,forces,self._mu,a,0);
        #print('Time to do numba %f' %(time.time()-t));
        #e1=time.time();
        # Call the C++ function to add up the RPY kernel
        #RPYSBTvel = ewc.RPYSBTKernel(self._N,Xarg[:,0],Xarg[:,1],Xarg[:,2],self._N,Xarg[:,0],\
        #            Xarg[:,1],Xarg[:,2],forces[:,0],forces[:,1],forces[:,2],self._mu,a,0);
        #print('Time to do cpp %f' %(time.time()-e1));
        #print('Error from numba ', subVel_Py-np.reshape(RPYSBTvel,(self._N,3)));
        return subVel_Py;

    def RPYSBTKernel(self,Xtarg,Xarg,forces,sbt=0):
        """
        Method to compute the RPY kernel at a target Xtarg due to fiber self
        Inputs: Xtarg = target point (3 vector), Xarg = points on the fiber (as an
        N x 3 dimensional vector), forces = N x 3 vector of forces (WITH weights) on the fiber. 
        sbt = 0 for the RPY kernel (they are different when r < 2a) and sbt = 1 for the SBT kernel.
        Outpts: the velocity due to N blobs (self RPY kernel) at target point as a 3 vector.
        This is a free space kernel; all periodic shifts are done elsewhere.
        Numba appears slightly faster than cpp here.
        """
        Npts,_ = Xarg.shape;
        a = sqrt(1.5)*self._L*self._epsilon;
        #import time
        #t=time.time();
        RPYSBTVel_Py = ewnb.RPYSBTK(1,np.array([Xtarg]),Npts,Xarg,forces,self._mu,a,sbt);
        #print('RPYSBT Time to do numba %f' %(time.time()-t));
        #e1=time.time();
        #RPYSBTvel = ewc.RPYSBTKernel(1,np.array([Xtarg[0]]),np.array([Xtarg[1]]),np.array([Xtarg[2]]),\
        #                          Npts,Xarg[:,0],Xarg[:,1],Xarg[:,2],forces[:,0],forces[:,1],forces[:,2],\
        #                          self._mu,a,sbt);
        #print('Time to do cpp %f' %(time.time()-e1));
        #print('Error from numba ', RPYSBTVel_Py-RPYSBTvel)
        return np.reshape(RPYSBTVel_Py,3);


    def SBTKernelSplit(self,Xtarg,Xarg,forceDs,w1,w3,w5):
        """
        Method to compute the SBT kernel at point Xtarg using 
        modified quadrature (special quadrature from Ludvig and Barnett).
        We have to split the kernel into 1/R, 1/R^3, and 1/R^5 because the weights
        are different for the separate kernels. 
        Inputs: Xtarg = target point (as a 3 vector), Xarg = fiber points (as an r x 3 vector)
        (r gets determined in the code), forceDs = force densities (as an r x 3 vector), 
        N vectors w1, w3, w5 of quadrature weights (those come from special quad).
        In this method numba is slightly faster than cpp
        """
        Npts,_ = Xarg.shape;
        #import time
        #t=time.time();
        NewVel = ewnb.SBTKSpl(Xtarg,Npts,Xarg,forceDs,self._mu,self._epsilon,\
                            self._L,w1,w3,w5);
        #print('Split time to do numba %f' %(time.time()-t));
        #e1=time.time();
        #CppVel = ewc.SBTKernelSplit(Xtarg,Npts,Xarg[:,0],Xarg[:,1],Xarg[:,2],forceDs[:,0],forceDs[:,1],\
        #        forceDs[:,2],self._mu,self._epsilon,self._L,w1,w3,w5);
        #print('Time to do cpp %f' %(time.time()-e1));
        #print('Error from cpp special', NewVel-CppVel)
        return NewVel;
        
    @staticmethod
    def cart2sph(x,y,z):
        """
        Method to convert point(s) (x,y,z) to 
        spherical coordinates (azimuth, elevation, r)
        """
        azimuth = np.arctan2(y,x);
        elevation = np.arctan2(z,np.sqrt(x**2 + y**2));
        r = np.sqrt(x**2 + y**2 + z**2);
        azimuth[(np.abs(np.abs(elevation)-np.pi/2) < 1e-12)] = 0;
        return azimuth, elevation, r
    
