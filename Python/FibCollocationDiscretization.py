import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import time
from math import sqrt, exp
from scipy.linalg import lu_factor, lu_solve

# Documentation last updated: 03/12/2021

aRPYFac = exp(1.5)/4        # equivalent RPY blob radius a = aRPYFac*epsilon*L;
# Some definitions that are not specific to any particular discretization
numBCs = 4; # number of boundary conditions
# Numbers of uniform/upsampled points
nptsUpsample = 32; # number of points to upsample to for special quadrature 
                   # (distinct from upsampling for direct quadrature)

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
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16,NupsampleForDirect=64,nptsUniform=16,rigid=False,trueRPYMobility=False):
        """
        Constructor. 
        L = fiber length, epsilon = fiber aspect ratio, Eb = bending stiffness, mu = fluid viscosity. 
        N = number of points on discretized fiber.
        NupsampleForDirect = how much to upsample by for direct quad, nptsUniform = number of uniform 
        points (for cross-linking / checking distances between fibers)
        """
        self._L = L;
        self._epsilon = epsilon;
        self._a = aRPYFac*self._epsilon*self._L;
        self._Eb = Eb;
        self._mu = mu;
        self._N = N;
        self._nptsUpsample = nptsUpsample;
        self._nptsUniform = nptsUniform;
        self._nptsDirect = NupsampleForDirect;
        self._nPolys = self._N-1;
        self._truRPYMob = trueRPYMobility;
        if (rigid):
            self._nPolys = 1;

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
        #if (self._N > 40):
        #    raise ValueError('The finite part integral is not well conditioned with > 40 pts,\
        #                        loss of accuracy expected');
        k = np.arange(self._N);
        s_dim1scaled = -1+2*self._s/self._L
        s_scaled = np.reshape(s_dim1scaled,(self._N,1)); # Rescale everything to [-1,1];
        q=(1.0+(-1.0)**(k+1.0)-2*s_scaled**(k+1.0))/(k+1.0);
        VanderMat = np.vander(s_dim1scaled,increasing=True);
        self._FPMatrix = 1.0/(8.0*np.pi*self._mu)*0.5*self._L*np.linalg.solve(VanderMat.T,q.T);
        self._DoubletFPMatrix = np.zeros((self._N,self._N));
        self._Nsmall = 0;
        self._RLess2aResamplingMat = np.zeros((self._N*self._Nsmall,self._N));
        self._RLess2aWts = np.zeros(self._N*self._Nsmall);

    def initLocalcvals(self,delta=0.1):
        """
        Initialize local leading order coefficients for the local drag matrix M. 
        The distance delta is the fraction of the fiber over which the ellipsoidal endpoint
        decay occurs. 
        See pg. 9 here: https://arxiv.org/pdf/2007.11728.pdf for formulas
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
    
    def initRigidFiberMobilityMatrixConstants(self,wFinitePart=False):
        if (wFinitePart):
            if (self._epsilon==0.01):
                alpha = 0.3841;
                beta = 0.2230;
                gamma = 3.4263;
            elif (self._epsilon==0.008):
                alpha = 0.4020;
                beta = 0.2413;
                gamma = 3.6412;
            elif (self._epsilon==0.006):
                alpha = 0.4251;
                beta = 0.2646;
                gamma = 3.9179;
            elif (self._epsilon==0.005):
                alpha = 0.4396;
                beta = 0.2793;
                gamma = 4.0931;
            elif (self._epsilon==0.004):
                alpha = 0.4575;
                beta = 0.2973;
                gamma = 4.3073;
            elif (self._epsilon==0.002):
                alpha = 0.5129;
                beta = 0.3530;
                gamma = 4.9721;
            elif (self._epsilon==0.001):
                alpha = 0.5682
                beta = 0.4085;
                gamma = 5.6361;
            else:
                raise ValueError('Coefficients not tabulated for N for this epsilon')
        else:
            if (self._epsilon==0.01):
                alpha = 0.3848;
                beta = 0.2249;
                gamma = 4.3910;
            elif (self._epsilon==0.008):
                alpha = 0.4026;
                beta = 0.2428;
                gamma = 4.6047;
            elif (self._epsilon==0.006):
                alpha = 0.4255;
                beta = 0.2658;
                gamma = 4.8802;
            elif (self._epsilon==0.005):
                alpha = 0.4401;
                beta = 0.2804;
                gamma = 5.0547;
            elif (self._epsilon==0.004):
                alpha = 0.4579;
                beta = 0.2983;
                gamma = 5.2683;
            elif (self._epsilon==0.002):
                alpha = 0.5132;
                beta = 0.3537;
                gamma = 5.9316;
            elif (self._epsilon==0.001):
                alpha = 0.5684
                beta = 0.4090;
                gamma = 6.5946;
            else:
                raise ValueError('Coefficients not tabulated for N for this epsilon')
        self._alpha = alpha;
        self._beta = beta;
        self._gamma =gamma;

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
    
    def resampleForDirectQuad(self,Xarg):
        """ 
        Get the locations of nodes for direct quad (Ndirect pts) on the fiber.
        Xarg = the fiber coordinates (self._N x 3 vector).
        The number of uniform points should have been set in the precomputation.
        """
        return np.dot(self._MatfromNtoDirectN,Xarg);
        
    def downsampleFromDirectQuad(self,Xarg):
        """ 
        Get the locations of nodes for direct quad (Ndirect pts) on the fiber.
        Xarg = the fiber coordinates (self._N x 3 vector).
        The number of uniform points should have been set in the precomputation.
        """
        return np.dot(self._MatfromDirectNtoN,Xarg);
    
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
    
    def getNumUpsample(self):
        return self._nptsUpsample;

    def getNumUniform(self):
        return self._nptsUniform;
    
    def getNumDirect(self):
        return self._nptsDirect;
    
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
    
    def getwDirect(self):
        return self._wForDirect;
    
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
    
    def computeNhalf(self,Xs):
        """
        Compute N^(1/2), where N is the grand 6 x 6 mobility matrix
        Input: Xs as a 3 vector
        Return N^(1/2) as a 6 x 6 matrix
        """
        Nhalf = np.zeros((6,6));
        Nhalf[0:3,0:3]=1/np.sqrt(self._mu*self._L)*(np.sqrt(self._alpha)*np.identity(3)\
            +(-np.sqrt(self._alpha)+np.sqrt(self._alpha+self._beta))*np.outer(Xs,Xs));
        Nhalf[3:,3:]=np.sqrt(self._gamma/(self._mu*self._L**3))*(np.identity(3)-np.outer(Xs,Xs));
        return Nhalf;
    
    def getnPolys(self):
        return self._nPolys;
        
    def getFPMatrix(self):
        """
        Finite part matrix
        This is the matrix A such that U = A*g, where g is the 
        modified finite part density with the singularity factored out. 
        """
        return self._FPMatrix.T;
    
    def getDoubletFPMatrix(self):
        """
        Finite part matrix
        This is the matrix A such that U = A*g, where g is the 
        modified finite part density with the singularity factored out. 
        """
        return self._DoubletFPMatrix.T;
    
    def getRless2aResampMat(self):
        return self._RLess2aResamplingMat;


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
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16,deltaLocal=1,NupsampleForDirect=64,nptsUniform=16,\
        rigid=False,trueRPYMobility=False,kinematicOversample=2):
        super().__init__(L,epsilon,Eb,mu,N,NupsampleForDirect,nptsUniform,rigid,trueRPYMobility);
		# Chebyshev grid and weights
        self._s = cf.chebPts(self._N,[0,self._L],chebGridType);
        self._w = cf.chebWts(self._N,[0,self._L],chebGridType);
        self._w2N = cf.chebWts(2*self._N,[0,self._L],chebGridType);
        self._wForDirect = cf.chebWts(self._nptsDirect, [0,self._L],chebGridType);
        # Chebyshev initializations
        self._Lmat = cf.CoeffstoValuesMatrix(self._N,self._N,chebGridType);
        self._LUCoeffs = lu_factor(self._Lmat);
        self._NForK = kinematicOversample*self._N;
        self.initIs();
        self.initFPMatrix();
        self.initLocalcvals(deltaLocal);
        self.initDiffMatrices();
        self.initD4BC();
        self.initResamplingMatrices();
        self.initSpecialQuadMatrices();
    
    def initFPMatrix(self):
        """
        Initialize the matrix for the finite part integral. 
        Uses the adjoint method of Anna Karin Tornberg. 
        This method is distinct from the one in the parent class because it uses 
        numerical integration of Chebyshev polynomials instead of exact integration
        of monomials
        """
        sscale=-1+2*self._s/self._L;
        AllQs = np.zeros((self._N,self._N));
        AllDs = np.zeros((self._N,self._N));
        self._DoubletFPMatrix = np.zeros((self._N,self._N));
        self._Nsmall = 0;
        self._RLess2aResamplingMat = np.zeros((self._N*self._Nsmall,self._N));
        self._RLess2aWts = np.zeros(self._N*self._Nsmall);
        a = 0;
        if (self._truRPYMob):
            a = self._a;
        for iPt in range(self._N):
            s = self._s[iPt];
            eta = sscale[iPt];
            sLow = max(s-2*a, 0);
            sHi = min(s+2*a, self._L);
            etaLow = -1+2*sLow/self._L;
            etaHi = -1+2*sHi/self._L;
            # Compute integrals numerically to high accuracy
            q = np.zeros(self._N);
            qd = np.zeros(self._N);
            NoversampToCompute = 200; # enough to get 10 digits
            for kk in range(self._N):
                if (etaLow > -1):
                    n = cf.chebPts(NoversampToCompute,[-1, etaLow],1);
                    w = cf.chebWts(NoversampToCompute,[-1, etaLow],1);
                    poly = np.cos(kk*np.arccos(n));
                    q[kk]=np.dot(w,((n-eta)/np.abs(n-eta)*poly));
                    qd[kk]=np.dot(w,((n-eta)/np.abs(n-eta)**3*poly));
                if (etaHi < 1):
                    n = cf.chebPts(NoversampToCompute,[etaHi, 1],1);
                    w = cf.chebWts(NoversampToCompute,[etaHi, 1],1);
                    poly = np.cos(kk*np.arccos(n));
                    q[kk]+=np.dot(w,((n-eta)/np.abs(n-eta)*poly));
                    qd[kk]+=np.dot(w,((n-eta)/np.abs(n-eta)**3*poly));
            AllQs[iPt,:]=q;
            AllDs[iPt,:]=qd;
        self._FPMatrix =  1.0/(8.0*np.pi*self._mu)*0.5*self._L*np.linalg.solve(self._Lmat.T,AllQs.T);
        if (self._truRPYMob):
            self._DoubletFPMatrix =  1.0/(8.0*np.pi*self._mu)*0.5*self._L*np.linalg.solve(self._Lmat.T,AllDs.T);
            # Initialize the resampling matrices for R < 2a
            # The only thing that can be precomputed are the matrices on [s-a, s] and [s,s+a] for resampling the
            # fiber positions
            # Each of these matrices is Nsmall/2 x N 
            # So the total matrix is Nsmall x N for each point. What gets passed into C++ is the row stacked version
            # of the grand matrix, which is (N*Nsmall) x N
            self._Nsmall = 4;
            if (self._epsilon > 1e-3):
                self._Nsmall = 8;
            self._RLess2aResamplingMat = np.zeros((self._N*self._Nsmall,self._N));
            self._RLess2aWts = np.zeros(self._N*self._Nsmall);
            for iPt in range(self._N):
                for iD in range(2):   
                    dom = [self._s[iPt], min(self._s[iPt]+2*self._a,self._L)];
                    if (iD==0):
                        dom = [max(self._s[iPt]-2*self._a,0),self._s[iPt]];         
                    ssm = cf.chebPts(self._Nsmall//2,dom,'L');
                    wsm = cf.chebWts(self._Nsmall//2,dom,'L');
                    ChebCoeffsToVals = cf.CoeffstoValuesMatrix(self._N,self._N,chebGridType);
                    LegCoeffsToVals = np.cos(np.outer(np.arccos(2*ssm/self._L-1),np.arange(self._N)));
                    self._RLess2aResamplingMat[iPt*self._Nsmall+iD*self._Nsmall//2:iPt*self._Nsmall+(iD+1)*self._Nsmall//2,:] =\
                     np.dot(LegCoeffsToVals,np.linalg.inv(ChebCoeffsToVals));
                    self._RLess2aWts[iPt*self._Nsmall+iD*self._Nsmall//2:iPt*self._Nsmall+(iD+1)*self._Nsmall//2]=wsm;
    
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
        self._MatfromNtoDirectN = cf.ResamplingMatrix(self._nptsDirect,self._N,chebGridType,chebGridType);
        self._MatfromDirectNtoN = cf.ResamplingMatrix(self._N,self._nptsDirect,chebGridType,chebGridType);
        self._Matfrom2NtoN = cf.ResamplingMatrix(self._N,2*self._N,chebGridType,chebGridType);
        self._MatfromKintoN = cf.ResamplingMatrix(self._N,self._NForK ,chebGridType,chebGridType);
        self._MatfromNtoKin = cf.ResamplingMatrix(self._NForK ,self._N,chebGridType,chebGridType);
        self._stackMatfromNto2N = np.zeros((6*self._N,3*self._N));
        self._stackMatfrom2NtoN = np.zeros((3*self._N,6*self._N));
        for iD in range(3):
            self._stackMatfromNto2N[iD::3,iD::3]=self._MatfromNto2N;
            self._stackMatfrom2NtoN[iD::3,iD::3]=self._Matfrom2NtoN;
        W2N = np.diag(np.repeat(self._w2N,3));
        UTWU = np.dot(self._stackMatfromNto2N.T,np.dot(W2N,self._stackMatfromNto2N));
        self._LeastSquaresDownsampler = np.linalg.solve(UTWU,np.dot(self._stackMatfromNto2N.T,W2N))
        self._WeightedUpsamplingMat= np.dot(W2N,self._stackMatfromNto2N);
        self._UpsampledChebPolys = np.dot(self._MatfromNto2N,self._Lmat[:,:self._nPolys]).T;

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
        curvaturessq = np.sum(Xss*Xss,axis=1);
        avgcurvature = np.sqrt(1.0/self._L*sum(self._w*curvaturessq));
        #normalizedcurvature = avgcurvature*self._L/(2*np.pi);
        # Obtain energy from avg curvature by kappa*L*avgcurvature**2
        return avgcurvature;
        
    def averageTau(self,Xs):
        avgTau = np.zeros(3);
        for p in range(self._Ntau):
            avgTau+= 1/(self._L)*Xs[p,:]*self._w[p];
        return avgTau
    
    def averageXsXs(self,Xs):
        avgXsXs = np.zeros((3,3));
        for p in range(self._Ntau):
            avgXsXs+= 1/(self._L)*np.outer(Xs[p,:],Xs[p,:])*self._w[p];
        return avgXsXs

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
        return cf.ResamplingMatrix(self._N,self._N,chebGridType,chebGridType, nPantarg=2);

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


    
