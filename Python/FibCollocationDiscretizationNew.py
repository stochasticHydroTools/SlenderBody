import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import time
from math import sqrt, exp
from scipy.linalg import lu_factor, lu_solve, sqrtm

# Documentation last updated: 03/12/2021

aRPYFac = exp(1.5)/4        # equivalent RPY blob radius a = aRPYFac*epsilon*L;
# Some definitions that are not specific to any particular discretization
numBCs = 4; # number of boundary conditions
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
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16,NupsampleForDirect=64,nptsUniform=16,rigid=False,\
        trueRPYMobility=False,UseEnergyDisc=True):
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
        self._Ntau = N;
        self._Nx = N+1;
        self._nptsUniform = nptsUniform;
        self._nptsDirect = NupsampleForDirect;
        self._truRPYMob = trueRPYMobility;
        self._Nsmall = 0;
        self._BendMatX0 = np.zeros(3*self._Nx)
        if (self._truRPYMob):
            self._Nsmall = 4;
            if (self._epsilon > 1e-3):
                self._Nsmall = 8;

    def initIs(self):
        """
        Initialize the identity matrix and matrix that takes 
        definite integrals (these are constant throughout a simulation).
        This method depends only on the quadrature weights, which we 
        assume have already been initialized
        """
        self._I = np.zeros((3*self._Nx,3));
        try:
            for j in range(self._Nx):
                self._I[3*j:3*j+3,:]=np.identity(3);
        except:
            raise NotImplementedError('You need to instantiate a specific discretization')

    def initLocalcvals(self,delta=0.1):
        """
        Initialize local leading order coefficients for the local drag matrix M. 
        The distance delta is the fraction of the fiber over which the ellipsoidal endpoint
        decay occurs. 
        See pg. 9 here: https://arxiv.org/pdf/2007.11728.pdf for formulas
        """
        radii = np.zeros(self._Nx);
        self._delta = delta;
        sNew = 0.5*self._L*np.ones(self._Nx);
        if (delta < 0.5):
            x = 2*self._sX/self._L-1;
            regwt = np.tanh((x+1)/delta)-np.tanh((x-1)/delta)-1;
            sNew = self._sX.copy();
            sNew[self._sX < self._L/2] =  regwt[self._sX < self._L/2]*self._sX[self._sX < self._L/2]+\
                (1-regwt[self._sX < self._L/2]**2)*delta*self._L/2;
            sNew[self._sX > self._L/2] = self._L-np.flip(sNew[self._sX < self._L/2]);
        self._leadordercs = np.log(4.0*sNew*(self._L-sNew)/(self._epsilon*self._L)**2);
           
    def averageTau(self,Xs):
        avgTau = np.zeros(3);
        for p in range(self._Ntau):
            avgTau+= 1/(self._L)*Xs[p,:]*self._wTau[p];
        return avgTau
    
    def averageXsXs(self,Xs):
        avgXsXs = np.zeros((3,3));
        for p in range(self._Ntau):
            avgXsXs+= 1/(self._L)*np.outer(Xs[p,:],Xs[p,:])*self._wTau[p];
        return avgXsXs
    
    def calcFibCurvature(self,X):
        """
        Calculate curvature for fiber X (N x 3 vector)
        """
        Xss = np.dot(self._DXGrid,np.dot(self._DXGrid,X));
        curvaturessq = np.sum(Xss*Xss,axis=1);
        avgcurvature = np.sqrt(1.0/self._L*sum(self._wX*curvaturessq));
        #normalizedcurvature = avgcurvature*self._L/(2*np.pi);
        # Obtain energy from avg curvature by kappa*L*avgcurvature**2
        return avgcurvature;
    
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
        rigid=False,trueRPYMobility=False,UseEnergyDisc=True,penaltyParam=0):
        super().__init__(L,epsilon,Eb,mu,N,NupsampleForDirect,nptsUniform,rigid,trueRPYMobility,UseEnergyDisc);
		# Chebyshev grid and weights
        self._sTau = cf.chebPts(self._Ntau,[0,self._L],chebGridType);
        self._wTau = cf.chebWts(self._Ntau,[0,self._L],chebGridType);
        self._XGridType = 1;
        if (UseEnergyDisc):
            self._XGridType = 2;
        self._sX = cf.chebPts(self._Nx,[0,self._L],self._XGridType);
        self._wX = cf.chebWts(self._Nx,[0,self._L],self._XGridType);
        self._wForDirect = cf.chebWts(self._nptsDirect, [0,self._L],chebGridType);
        self.initIs();
        self.calcXFromXsMap();
        self.initLocalcvals(deltaLocal);
        self.initD4BC(UseEnergyDisc,penaltyParam);
        self.initFPMatrix();
        MatfromNtoUniform = cf.ResamplingMatrix(self._nptsUniform,self._Nx,'u',self._XGridType);
        self._MatfromNtoUniform = np.zeros((3*self._nptsUniform,3*self._Nx));
        for iD in range(3):
            self._MatfromNtoUniform[iD::3,iD::3]=MatfromNtoUniform;
    
        
    def calcXFromXsMap(self):
        MidpointMat = cf.ResamplingMatrix(3,self._Nx,self._XGridType,self._XGridType);
        MidpointMat = MidpointMat[1,:];
        self._DXGrid =  cf.diffMat(1,[0,self._L],self._Nx,self._Nx,self._XGridType,self._XGridType);
        RNToNp1 = cf.ResamplingMatrix(self._Nx,self._Ntau,self._XGridType,chebGridType);
        RNp1ToN = cf.ResamplingMatrix(self._Ntau,self._Nx,chebGridType,self._XGridType);
        XonNp1MatOne = np.dot(np.dot((np.eye(self._Nx)-np.tile(MidpointMat,(self._Nx,1))),\
            np.linalg.pinv(self._DXGrid)),RNToNp1);
        XsFromXOne = np.dot(RNp1ToN,self._DXGrid);
        self._XonNp1MatNoMP = np.zeros((3*self._Nx,3*self._Nx-3));
        self._XsFromX = np.zeros((3*self._Nx-3,3*self._Nx));
        self._MidpointMat = MidpointMat;
        for iD in range(3):
            self._XonNp1MatNoMP[iD::3,iD::3]=XonNp1MatOne;
            self._XsFromX[iD::3,iD::3]=XsFromXOne;
        self._XonNp1Mat = np.concatenate((self._XonNp1MatNoMP, self._I),axis=1);
    
    def calcXFromXsAndMP(self,Xs,XMP):
        return np.dot(self._XonNp1Mat,np.concatenate((Xs,XMP)));     
    
    def gets(self):
        return self._sX;
    
    def getw(self):
        return self._wX; 
    
    def initD4BC(self,UseEnergyDisc,penaltyParam):
        """
        Compute the operator D_4^BC that is necessary when computing 
        the bending forces. The constant -E_bend is included in 
        this operator
        """
        
        sDouble = cf.chebPts(2*self._Nx,[0,self._L],self._XGridType);
        wDouble = cf.chebWts(2*self._Nx,[0,self._L],self._XGridType);
        R_Nx_To_2x = cf.ResamplingMatrix(2*self._Nx,self._Nx,self._XGridType,self._XGridType);
        DSq = cf.diffMat(2,[0,self._L],self._Nx,self._Nx,self._XGridType,self._XGridType);
        WtildeNx = np.dot(R_Nx_To_2x.T,np.dot(np.diag(wDouble),R_Nx_To_2x));
        if (UseEnergyDisc):
            OneD4BCForce = -self._Eb*np.dot(DSq.T,np.dot(WtildeNx,DSq));
            # Convert to force density by inverting by Wtilde
            OneD4BC = np.linalg.solve(WtildeNx,OneD4BCForce);
        else:
            DownsamplingMat = cf.ResamplingMatrix(self._Nx,self._Nx+numBCs,chebGridType,D4BCgridType);
            SecDMat_upgrid = cf.diffMat(2,[0,self._L],2,self._Nx+numBCs,D4BCgridType,D4BCgridType);
            ThirDMat_upgrid = cf.diffMat(3,[0,self._L],2,self._Nx+numBCs,D4BCgridType,D4BCgridType);
            TotalBCMatrix = np.concatenate((DownsamplingMat,SecDMat_upgrid,ThirDMat_upgrid));
            # This is the only place where you would modify the "free fiber" BCs
            RHS = np.concatenate((np.identity(self._Nx),np.zeros((4,self._Nx))));
            TildeConfigMatrix = np.linalg.solve(TotalBCMatrix,RHS);
            FourDMat_fromUptoDwn = cf.diffMat(4,[0,self._L],self._Nx,self._Nx+numBCs,chebGridType,D4BCgridType);
            OneD4BC = -self._Eb*np.dot(FourDMat_fromUptoDwn,TildeConfigMatrix);
            OneD4BCForce = np.dot(WtildeNx, OneD4BC);
            # Convert to force by multipying by Wtilde
            # OneD4BC = np.dot(WtildeNx,OneD4BC);
        if (penaltyParam > 0):
            OneD4BC = OneD4BC - penaltyParam*np.eye(self._Nx); 
            OneD4BCForce = OneD4BCForce - penaltyParam*WtildeNx;
        # Fill in the block stacked matrix
        OneD4BCForceHalf = sqrtm(-1.0*OneD4BCForce);
        self._D4BC = np.zeros((3*self._Nx,3*self._Nx));
        self._D4BCForce = np.zeros((3*self._Nx,3*self._Nx));
        self._D4BCForceHalf = np.zeros((3*self._Nx,3*self._Nx));
        self._stackWTilde_Nx =np.zeros((3*self._Nx,3*self._Nx));
        self._stackWTildeInverse_Nx =np.zeros((3*self._Nx,3*self._Nx));
        for iD in range(3):
            self._D4BC[iD::3,iD::3]=OneD4BC; 
            self._D4BCForce[iD::3,iD::3]=OneD4BCForce; 
            self._D4BCForceHalf[iD::3,iD::3]=OneD4BCForceHalf; 
            self._stackWTilde_Nx[iD::3,iD::3] =  WtildeNx;    
            self._stackWTildeInverse_Nx[iD::3,iD::3] =  np.linalg.inv(WtildeNx);   
    
    def calcBendMatX0(self,X0,penaltyParam):
        self._BendMatX0 = np.dot(self._D4BCForce,X0);
        if (abs(penaltyParam) < 1e-10):
            self._BendMatX0 = 0*self._BendMatX0;    
    
    def initFPMatrix(self):
        """
        Initialize the matrix for the finite part integral. 
        Uses the adjoint method of Anna Karin Tornberg. 
        This method is distinct from the one in the parent class because it uses 
        numerical integration of Chebyshev polynomials instead of exact integration
        of monomials
        """
        sscale=-1+2*self._sX/self._L;
        ChebCoeffsToVals = cf.CoeffstoValuesMatrix(self._Nx,self._Nx,self._XGridType);
        AllQs = np.zeros((self._Nx,self._Nx));
        AllDs = np.zeros((self._Nx,self._Nx));
        self._DoubletFPMatrix = np.zeros((self._Nx,self._Nx));
        self._RLess2aResamplingMat = np.zeros((self._Nx*self._Nsmall,self._Nx));
        self._RLess2aWts = np.zeros(self._Nx*self._Nsmall);
        a = 0;
        if (self._truRPYMob):
            a = self._a;
        for iPt in range(self._Nx):
            s = self._sX[iPt];
            eta = sscale[iPt];
            sLow = max(s-2*a, 0);
            sHi = min(s+2*a, self._L);
            etaLow = -1+2*sLow/self._L;
            etaHi = -1+2*sHi/self._L;
            # Compute integrals numerically to high accuracy
            q = np.zeros(self._Nx);
            qd = np.zeros(self._Nx);
            NoversampToCompute = 200; # enough to get 10 digits
            for kk in range(self._Nx):
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
        self._FPMatrix =  1.0/(8.0*np.pi*self._mu)*0.5*self._L*np.linalg.solve(ChebCoeffsToVals.T,AllQs.T);
        if (self._truRPYMob):
            self._DoubletFPMatrix =  1.0/(8.0*np.pi*self._mu)*0.5*self._L*np.linalg.solve(ChebCoeffsToVals.T,AllDs.T);
            # Initialize the resampling matrices for R < 2a
            # The only thing that can be precomputed are the matrices on [s-a, s] and [s,s+a] for resampling the
            # fiber positions
            # Each of these matrices is Nsmall/2 x N 
            # So the total matrix is Nsmall x N for each point. What gets passed into C++ is the row stacked version
            # of the grand matrix, which is (N*Nsmall) x N
            self._RLess2aResamplingMat = np.zeros((self._Nx*self._Nsmall,self._Nx));
            self._RLess2aWts = np.zeros(self._Nx*self._Nsmall);
            for iPt in range(self._Nx):
                for iD in range(2):   
                    dom = [self._sX[iPt], min(self._sX[iPt]+2*self._a,self._L)];
                    if (iD==0):
                        dom = [max(self._sX[iPt]-2*self._a,0),self._sX[iPt]];  
                    ssm = cf.chebPts(self._Nsmall//2,dom,'L');
                    wsm = cf.chebWts(self._Nsmall//2,dom,'L');
                    LegCoeffsToVals = np.cos(np.outer(np.arccos(2*ssm/self._L-1),np.arange(self._Nx)));
                    self._RLess2aResamplingMat[iPt*self._Nsmall+iD*self._Nsmall//2:iPt*self._Nsmall+(iD+1)*self._Nsmall//2,:] =\
                     np.dot(LegCoeffsToVals,np.linalg.inv(ChebCoeffsToVals));
                    self._RLess2aWts[iPt*self._Nsmall+iD*self._Nsmall//2:iPt*self._Nsmall+(iD+1)*self._Nsmall//2]=wsm;
