import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import time
from math import sqrt
from scipy.linalg import lu_factor, lu_solve

# Some definitions that are not specific to any particular discretization
numBCs = 4; # number of boundary conditions
# Numbers of uniform/upsampled points
nptsUpsample = 32; # number of points to upsample to for special quadrature
nptsUniform = 16; # number of uniform points to estimate distance

class FibCollocationDiscretization(object):

    """ 
    The object code for a fiber discretization. 
    The abstract class is a template for ANY collocation discretization,
    then the child class is specific to Chebyshev. The child class contains
    the relevant methods that need to be modified if we were to switch to e.g. 
    Legendre.
    """
    
    ## ===========================================
    ##           METHODS FOR INITIALIZATION
    ## ===========================================
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16):
        """
        Constructor. 
        L = fiber length, epsilon = fiber aspect ratio, Eb = bending stiffness, mu = fluid viscosity. 
        N = number of points on discretized fiber.
        """
        self._L = L;
        self._epsilon = epsilon;
        self._Eb = Eb;
        self._mu = mu;
        self._N = N;
        self._nptsUpsample = nptsUpsample;
        self._nptsUniform = nptsUniform;

    def initIs(self):
        """
        Initialize the identity matrix and matrix that takes 
        definite integrals (these are constant throughout a simulation).
        This method depends only on the quadrature weights, which we 
        assume have already been initialized
        """
        self._I = np.zeros((3*self._N,3));
        self._wIt = np.zeros((3,3*self._N));
        try:
            for j in range(self._N):
                self._I[3*j:3*j+3,:]=np.identity(3);
                self._wIt[:,3*j:3*j+3]=np.identity(3)*self._w[j];
        except:
            raise NotImplementedError('You need to instantiate a specific discretization')

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

    def initLocalcvals(self,delta=0.1):
        """
        Initialize local leading order coefficients for the local drag matrix M. 
        The distance delta is the fraction of the fiber over which the ellipsoidal endpoint
        decay occurs. 
        Between [0,delta]: set to ellipsoidal decay, c = -log(epsilon^2), 
        Between [2*delta, L/2]: set to cylindrical constant radius
        Between [delta,2*delta]: interpolate between cylindrical and ellipsoidal
        """
        radii = np.zeros(self._N);
        self._delta = delta;
        sNew = 0.5*self._L*np.ones(self._N);
        if (delta < 0.5):
            x = 2*self._s/self._L-1;
            regwt = np.tanh((x+1)/delta)-np.tanh((x-1)/delta)-1;
            sNew = self._s.copy();
            sNew[self._s < self._L/2] =  regwt[self._s < self._L/2]*self._s[self._s < self._L/2]+\
                (1-regwt[self._s < self._L/2]**2)*delta*self._L/2;
            sNew[self._s > self._L/2] = self._L-np.flip(sNew[self._s < self._L/2]);
        self._leadordercs = np.log(4.0*sNew*(self._L-sNew)/(self._epsilon*self._L)**2);
        self._matlist = [None]*self._N; # allocate memory for sparse matrix

    ## ====================================================
    ##  METHODS FOR RESAMPLING AND ACCESS BY OTHER CLASSES
    ## ====================================================
    def resampleUniform(self,Xarg):
        """ 
        Get the locations of uniformly distributed nodes on the fiber.
        Xarg = the fiber coordinates (self._N x 3 vector).
        The number of uniform points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNtoUniform,Xarg);
    
    def upsampleGlobally(self,Xarg):
        """
        Get the locations of some upsampled nodes on the fiber. 
        Xarg = the coordinates (self._N x 3 vector). 
        The number of upsampled points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNtoUpsamp,Xarg);
    
    def getUpsamplingMatrix(self):
        return self._MatfromNtoUpsamp;
    
    def upsample2Panels(self,Xarg):
        """
        Get the locations of 2 panels of upsampled nodes on the fiber.
        Xarg = the coordinates (self._N x 3 vector). 
        The number of upsampled points should have been set in the precomputation.
        Note: In theory the number of panels could be an argument and then all the resampling
        matrices precomputed, then a bunch of if statements. But we are being lazy...
        """
        return np.dot(self._MatfromNto2panUp,Xarg);

    def specialWeightsfromIntegrals(self,integrals,troot):
        """
        Compute the special quad weights from the integrals involving the 
        root troot. 
        Input: integrals = the values of the integrals, troot = complex root,
        where the centerline in this basis is on [-1,1]
        Output: the weights for special quadrature.
        """
        # Solve with the precomputed LU factorization
        seriescos = lu_solve(self._NupsampLUpiv,integrals,check_finite=False);
        distance_pows = np.reshape(np.concatenate((abs(self._specQuadNodes-troot),\
                abs(self._specQuadNodes-troot)**3,abs(self._specQuadNodes-troot)**5)),\
                (3,len(self._specQuadNodes))).T;
        # Rescale weights (which come back on [-1,1]) by multiplying by L/2.0
        special_wts = seriescos*distance_pows*self._L/2.0;
        return special_wts;
    
    def upsampledCoefficients(self,Xarg):
        """
        Get the coefficients of an upsampled representation of the fiber.
        Inputs: Xarg = upsampled fiber representation
        Outputs: the coefficients and derivative coefficients of the upsampled
        representation
        """
        raise NotImplementedError('upsampledCoefficients needs specific discretization.');

    def evaluatePosition(self,tapprox,coeffs):
        """
        Evaluate the series representing the fiber at a value tapprox.
        Inputs: tapprox = value to evaluate at. This value must be
        rescaled so that the centerline is parameterized by t in [-1,1],
        coeffs = coefficients that represent the fiber centerline 
        (arbitrary number)
        """
        raise NotImplementedError('evaluatePosition needs specific discretization.');

    def resample(self,Xarg,Nrs,typetarg):
        """
        Resample the fiber at Nrs nodes of type type
        Inputs: Xarg = a self._N x 3 vector of the fiber points,
        Nrs = number of points where the resampling is desired. typetarg = type
        of points
        Outputs: the resampled locations as an Nrs x 3 vector of points. 
        """
        raise NotImplementedError('resample needs specific discretization.');

    def newNodes(self,Nrs,typetarg,numPanels=1):
        """
        New nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new nodes
        """
        raise NotImplementedError('newNodes needs specific discretization.');

    def newWeights(self,Nrs,typetarg):
        """
        New quadrature weights for a set of Chebyshev nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new weights wN
        """
        raise NotImplementedError('newWeights needs specific discretization.');

    def integrateXs(self,Xsp1):
        """
        Method to integrate Xs and get X.
        """
        raise NotImplementedError('integrateXs needs specific discretization.');
    
    def getNumUpsample(self):
        return self._nptsUpsample;

    def getNumUniform(self):
        return self._nptsUniform;
    
    def getSpecialQuadNodes(self):
        return self._specQuadNodes;
    
    def getUpsampledWeights(self):
        return self._upsampledWeights;

    def getN(self):
        return self._N;

    def gets(self):
        return self._s;
    
    def getw(self):
        return self._w;
    
    def getepsilonL(self):
        return self._epsilon, self._L;
    
    def getValstoCoeffsMatrix(self):
        """
        Return the N x N matrix that gives the coefficients of 
        a Chebyshev series from the values
        """
        return np.linalg.inv(self._Lmat);
    
    def getDiffMat(self):
        return self._Dmat;

    def calcfE(self,X):
        """
        Calculates the bending force fE
        Inputs: position X. 
        Outpts: bending forces fE
        """
        return np.dot(self._D4BC,X);

    ## ====================================================
    ##  METHODS FOR FIBER EVOLUTION (PUBLIC)
    ## ====================================================
    def alphaLambdaSolve(self,Xarg,Xsarg,dt,impco,nLvel,exF):
        """
        This method solves the linear system for 
        lambda and alpha for a given RHS. 
        Specifically, the system we are solving is in block form
        [-M K-impco*dt*M*L*K; K^T 0]] [lambda; alpha] = ...
            [M*L*X^n + U_0 + nLvel + M*exF; 0]
        here impco = 1 for backward Euler, 1/2 for CN, 0 for explicit. 
        We solve the system using the Schur complement. 
        The Schur blocks are [M B; C 0] correspdong to the LHS matrix above, and 
        L is encoded via self._D4BC.
        Inputs: Xarg and Xsarg = X and Xs to build the matrices for
        dt = timestep, impco = implicit coefficient (coming from the temporal integrator)
        nLvel = non-local velocity as a 3N vector, exF = external forces as a 3N vector
        """
        tim = time.time();
        M = self.calcM(Xsarg);
        K, Kt = self.calcKs(Xsarg);
        # Schur complement solve
        B = np.concatenate((K-impco*dt*np.dot(M,np.dot(self._D4BC,K)),\
         self._I-impco*dt*np.dot(M,np.dot(self._D4BC,self._I))),axis=1);
        C = np.concatenate((Kt,self._wIt));
        fE = self.calcfE(Xarg);
        RHS = np.dot(C,fE+exF)+np.dot(C,np.linalg.solve(M,nLvel));
        S = np.dot(C,np.linalg.solve(M,B));
        alphaU,res,rank,s = np.linalg.lstsq(S,RHS,-1);
        vel = np.dot(K,alphaU[0:2*self._N-2])+\
            np.dot(self._I,alphaU[2*self._N-2:2*self._N+1]);
        lambdas = np.linalg.solve(M,vel-nLvel)-fE-exF- \
                impco*dt*np.dot(self._D4BC,vel);
        return alphaU, vel, lambdas;
    
    def KalphProduct(self,Xsarg,alphaU,lambdas):
        """
        The products K*alpha and K'*lambda for a given alpha and lambda. 
        Inputs: Xsarg = tangent vectors along fiber as 3*N one-d array, 
        alphaU = alphas and lambdas. 
        Outputs: products K*alpha and K^* *lambda.
        """
        K, Kt = self.calcKs(Xsarg);
        Kalph = np.dot(K,alphaU[0:2*self._N-2])+\
            np.dot(self._I,alphaU[2*self._N-2:2*self._N+1]);
        Kstlam = np.dot(np.concatenate((Kt,self._wIt)),lambdas);
        return Kalph,Kstlam;
    
    def calcH(self,Xin):
        """
        Compute discrete integral of bending force given 3*N vector of point
        locations Xin
        """
        return np.dot(self._wIt, np.dot(self._D4BC,Xin));
    
    ## ====================================================
    ##  METHODS FOR NON-LOCAL VELOCITY EVALUATION (PUBLIC)
    ## ====================================================
    def getFPMatrix(self):
        """
        Finite part matrix
        This is the matrix A such that U = A*g, where g is the 
        modified finite part density with the singularity factored out. 
        """
        return 1.0/(8.0*np.pi*self._mu)*0.5*self._L*self._FPMatrix.T;
    
    def calcLocalVelocity(self,Xsarg,forceDs):
        """
        Purely local velocity = M*f. 
        Inputs: tangent vectors Xs as a 3N one-dimensional vector, and 
        forceDs as a 3N one-dimensional vector
        Outputs: 3N one-d vector of local velocities.
        """
        M = self.calcM(Xsarg);
        return np.dot(M,forceDs);
    
    ## ====================================================
    ##  PRIVATE METHODS INVOLVED IN FIBER EVOLUTION
    ## ====================================================
    def calcKs(self,Xs):
        """
        Computes the matrix K(X). The only input is X_s, which
        has to be input as a 3N one-dimensional array.
        From X_s this method computes the normals and then 
        the matrix K.   
        """
        XsUpsampled = np.dot(self._MatfromNto2N,np.reshape(Xs,(self._N,3)));
        theta, phi, r = FibCollocationDiscretization.cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
        n1x = -np.sin(theta);
        n1y = np.cos(theta);
        n1z = np.zeros(2*self._N);
        n2x = -np.cos(theta)*np.sin(phi);
        n2y = -np.sin(theta)*np.sin(phi);
        n2z = np.cos(phi);
        J = np.zeros((6*self._N,2*self._N-2));
        J[0::3,0:self._N-1]= self.deAliasIntegral(n1x);
        J[1::3,0:self._N-1]= self.deAliasIntegral(n1y);
        J[2::3,0:self._N-1]= self.deAliasIntegral(n1z);
        J[0::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2x);
        J[1::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2y);
        J[2::3,self._N-1:2*self._N-2]= self.deAliasIntegral(n2z);
        K = np.dot(self._stackMatfrom2NtoN,J);
        UTWU = np.dot(self._stackMatfromNto2N.T,np.dot(np.diag(np.repeat(self._w2N,3)),self._stackMatfromNto2N));
        K = np.linalg.solve(UTWU,np.dot(self._stackMatfromNto2N.T,np.dot(np.diag(np.repeat(self._w2N,3)),J)));
        Kt = np.dot(J.T,np.dot(np.diag(np.repeat(self._w2N,3)),self._stackMatfromNto2N));
        return K, Kt;
    
    def deAliasIntegral(self,f):
        """
        Method to dealias the product of f (an N array) with each of the first
        N-1 polynomials (the type of polynomial depends on the child class).
        Upsample to a 2N grid, perform multiplication and integration, then 
        downsample to an N point grid.
        """
        # Upsample the multiplication of f with Chebyshev polys for anti-aliasing
        UpSampMulti = np.reshape(f,(2*self._N,1)) \
            *np.dot(self._MatfromNto2N,self._Lmat[:,:self._N-1]);
        # Integrals on the original grid (integrate on upsampled grid and downsample)
        OGIntegrals = np.dot(self._Dpinv2N,UpSampMulti);
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
        return azimuth, elevation, r;

# Definitions specific to Chebyshev
chebGridType = 1; # always use a type 1 grid for fiber discretization
D4BCgridType = 2; # always a type 2 grid to enforce the BCs
class ChebyshevDiscretization(FibCollocationDiscretization):

    """
    Child class of FibCollocationDiscretization that implements
    methods specific to CHEBYSHEV.
    Notice that every method in this class has cf. in it.
    """
	
    ## ===========================================
    ##           METHODS FOR INITIALIZATION
    ## ===========================================
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16,deltaLocal=1):
        super().__init__(L,epsilon,Eb,mu,N);
		# Chebyshev grid and weights
        self._s = cf.chebPts(self._N,[0,self._L],chebGridType);
        self._w = cf.chebWts(self._N,[0,self._L],chebGridType);
        self._w2N = cf.chebWts(2*self._N,[0,self._L],chebGridType);
        # Chebyshev initializations
        self._Lmat = cf.CoeffstoValuesMatrix(self._N,self._N,chebGridType);
        self._LUCoeffs = lu_factor(self._Lmat);
        self.initIs();
        self.initFPMatrix();
        self.initLocalcvals(deltaLocal);
        self.initDiffMatrices();
        self.initD4BC();
        self.initResamplingMatrices();
        self.initSpecialQuadMatrices();

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
        for iD in range(3):
            self._D4BC[iD::3,iD::3]=OneD4BC;

    def initResamplingMatrices(self):
        """
        Pre-compute resampling matrices that go from N points to 2N points and vice versa.
        """
        self._MatfromNto2N = cf.ResamplingMatrix(2*self._N,self._N,chebGridType,chebGridType);
        self._Matfrom2NtoN = cf.ResamplingMatrix(self._N,2*self._N,chebGridType,chebGridType);
        self._stackMatfromNto2N = np.zeros((6*self._N,3*self._N));
        self._stackMatfrom2NtoN = np.zeros((3*self._N,6*self._N));
        for iD in range(3):
            self._stackMatfromNto2N[iD::3,iD::3]=self._MatfromNto2N;
            self._stackMatfrom2NtoN[iD::3,iD::3]=self._Matfrom2NtoN;

    def initSpecialQuadMatrices(self):
        """
        Initialize matrices that are necessary for special quadrature / resampling.
        """
        self._MatfromNtoUpsamp = cf.ResamplingMatrix(self._nptsUpsample,self._N,chebGridType,chebGridType);
        self._MatfromNto2panUp = cf.ResamplingMatrix(self._nptsUpsample,self._N,chebGridType,chebGridType,\
                    nPantarg=2);
        self._MatfromNtoUniform = cf.ResamplingMatrix(self._nptsUniform,self._N,'u',chebGridType);
        self._UpsampledCoefficientsMatrix = cf.CoeffstoValuesMatrix(self._nptsUpsample,self._nptsUpsample,chebGridType);
        self._UpsampCoeffLU = lu_factor(self._UpsampledCoefficientsMatrix);
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

    ## ====================================================
    ##  METHODS FOR RESAMPLING AND ACCESS BY OTHER CLASSES
    ## ====================================================   
    def Coefficients(self,Xarg):
        """
        Get the coefficients of the fiber represenation. 
        Inputs: Xarg = fiber representation
        Outputs: the coefficients and derivative coefficients of the
        representation
        """
        Xcoeffs= lu_solve(self._LUCoeffs,Xarg,check_finite=False);
        Xprimecoeffs = cf.diffCoefficients(Xcoeffs,self._N);
        return Xcoeffs,Xprimecoeffs;

    def upsampledCoefficients(self,Xarg):
        """
        Get the coefficients of an upsampled representation of the fiber.
        Inputs: Xarg = upsampled fiber representation
        Outputs: the coefficients and derivative coefficients of the upsampled
        representation
        """
        Xcoeffs= lu_solve(self._UpsampCoeffLU,Xarg,check_finite=False);
        Xprimecoeffs = cf.diffCoefficients(Xcoeffs,self._nptsUpsample);
        return Xcoeffs,Xprimecoeffs;

    def evaluatePosition(self,tapprox,coeffs):
        """
        Evaluate the Chebyshev series representing the fiber 
        at a value tapprox.
        Inputs: tapprox = values to evaluate at. These values must be
        rescaled so that the centerline is parameterized by t in [-1,1],
        coeffs = coefficients that represent the fiber centerline 
        (arbitrary number)
        """
        return cf.evalSeries(coeffs,np.arccos(tapprox));
    
    def calcFibCurvature(self,X):
        """
        Calculate curvature for fiber X (N x 3 vector)
        """
        Xss = np.dot(self._Dmat,np.dot(self._Dmat,X));
        curvatures = np.sqrt(np.sum(Xss*Xss,axis=1));
        avgcurvature = 1.0/self._L*sum(self._w*curvatures);
        return avgcurvature;

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
    
    def get2PanelUpsamplingMatrix(self):
        return cf.ResamplingMatrix(self._N,self._N,chebGridType,chebGridType,\
                    nPantarg=2);

    def newNodes(self,Nrs,typetarg=chebGridType, numPanels=1):
        """
        New Chebyshev nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new nodes sNew
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

    def integrateXs(self,Xsp1):
        """
        Method to integrate Xs and get X.
        """
        Xshat = np.linalg.solve(self._Lmat,Xsp1); # coefficients
        Xhat = cf.intCoefficients(Xshat,self._N,[0,self._L]); # one integration
        X = np.dot(self._Lmat,Xhat); # back to real space
        return X;


    
