import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import time
from math import sqrt, exp
from scipy.linalg import lu_factor, lu_solve, sqrtm
from warnings import warn
from RPYVelocityEvaluator import RPYVelocityEvaluator

aRPYFac = exp(1.5)/4        # equivalent RPY blob radius a = aRPYFac*epsilon*L;
# Some definitions that are not specific to any particular discretization
numBCs = 4; # number of boundary conditions
SpecialQuadEigenvalueSafetyFactor = 1;

class FibCollocationDiscretization(object):

    """ 
    This class stores information about the collocation discretization
    of a single fiber.
    The abstract class is a template for ANY collocation discretization,
    then the child class is specific to Chebyshev. The child class contains
    the relevant methods that need to be modified if we were to switch to e.g. 
    Legendre.
    
    The primary purpose of this class is to precompute the matrices that get used in the 
    force and kinematic $K$ matrix calculations. 
    These matrix are passed to C++ through the class
    fiberCollection. There are NO actual computations done in this class (only precomputations).
    
    This class is where the self mobility is set. There are four options for this:
    
    1) Special quadrature on the RPY integral (set RPYSpecialQuad = True)
    2) Direct quadrature without upsampling on the RPY integral (set RPYDirectQuad = True)
    3) Oversampled quadrature on the RPY integral (set RPYOversample = True)
    4) Slender body theory with modified local drag coefficients at the endpoints (set all three
       options to false to obtain this)
    
    In the case of 1) and 4), the mobility can be written as a local drag part + a remainder which 
    describes the hydrodynamics from the rest of the fiber. The variable FPIsLocal (self._FinitePartLocal) 
    controls whether this remaining intra-fiber hydrodynamics (the finite part integral in SBT) is considered "local"
    by the code (whether it becomes part of the dense matrix $M_L$) or whether it is nonlocal, 
    which means it can only be applied, but not formed.
    """
    
    ## ===========================================
    ##           METHODS FOR INITIALIZATION
    ## ===========================================
    def __init__(self, L, epsilon,Eb=1,mu=1,N=16,deltaLocal=1,NupsampleForDirect=64,nptsUniform=16,\
        rigid=False,RPYSpecialQuad=False,RPYDirectQuad=False,RPYOversample=True,UseEnergyDisc=True,\
        FPIsLocal=True,penaltyParam=0):
        """
        Constructor. 
        
        Parameters
        -----------
        L: double
            The fiber length $L$
        epsilon: double
            The fiber aspect ratio. IMPORTANT: this is the actual aspect ratio, i.e., the 
            fiber radius $a$ divided by the length:=, $\\epsilon=a/L$.
            This is NOT the aspect ratio of the RPY tensor, $\\hat{\\epsilon}=\\hat{a}/L$.
            The RPY radius can be computed from the actual radius
            by multiplying by the double aRPYFac$=e^{3/2}/4 \\approx 1.12$. 
            This double is set as a constant in the 
            header of this file, and is the only place throughout the code where it is set.
        Eb: double
            The bending stiffness (called $\\kappa$ in most papers, thesis)
        mu: double
            The suspending fluid viscosity.
        N: int
            Number of unit-length tangent vectors making up the discretized fiber. The number of 
            degrees of freedom is $N_x=N+1$, since the fiber midpoint makes an additional DOF.
        delta: double
            When using SBT, this regularizes the local drag coefficients so they are nonsingular
            at the endpoints. The regularization is basically to take an ellipsoidal fiber on 
            lengthscales less than $\\delta L$ and a cylindrical fiber on lengthscales larger. 
            See Section 1.1.3 in Maxian's PhD thesis.
        NupsampleForDirect: int
            If we do direct quadrature/upsampling for the RPY mobility, this is the number 
            of points we upsample to 
        nptsUniform: int 
            Number of uniform points we use for resampling in external (typically cross linking)
            calculations
        rigid: bool
            If the fibers are rigid
        RPYSpecialQuad: bool
            Whether to use special quadrature for the mobility
        RPYDirectQuad: bool
            Whether to use direct quadrature for the mobility
        RPYOversample: bool
            Whether to use oversampled quadrature for the mobility
        UseEnergyDisc: bool, optional 
            Whether to use an energy discretization for the elastic force (defaults to true). If false, uses
            rectangular spectral collocation. The difference between these two will be described in 
            more detail below (method initD4BC). Basically, the energy discretization is better when the problem
            is nonsmooth since it guarantees zero force and torque on the fiber.
        FPIsLocal: bool, optional
            Whether intra-fiber (or "finite part") hydrodynamics is considered local or nonlocal.
            This is important when we form the mobility matrix $M_L$ in the preconditioner. The 
            default is true.
        penaltyParam: double, optional
            If the fiber is bound by a penalty force, this gives the penalty parameter. Defaults to 
            zero.
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
        self._RPYSpecialQuad = RPYSpecialQuad;
        self._RPYDirectQuad = RPYDirectQuad;
        self._FinitePartLocal=FPIsLocal;
        if (self._RPYDirectQuad):
            print('Overwriting the number of oversampling points with Nx, since you selected direct quad')
            self._nptsDirect = self._Nx;
        self._RPYOversample = RPYOversample;
        self._Nsmall = 0;
        self._BendMatX0 = np.zeros(3*self._Nx)
        if (self._RPYSpecialQuad):
            self._Nsmall = 4;
            if (self._epsilon > 1e-3):
                self._Nsmall = 8;

    def initIs(self):
        """
        Initialize the identity matrix which maps a $3 \\times 1$ vector to the
        whole grid
        """
        self._I = np.zeros((3*self._Nx,3));
        try:
            for j in range(self._Nx):
                self._I[3*j:3*j+3,:]=np.identity(3);
        except:
            raise NotImplementedError('You need to instantiate a specific discretization')

    def initLocalcvals(self,delta=0.1):
        """
        Initialize local leading order coefficients for the local drag matrix $M_L$. 
        As mentioned in the constructor, the mobility on a single fiber can, in the 
        case of SBT, be written as
        $$M = M_{LD} + M_{FP}$$
        The local drag matrix $M_{LD}$ can in turn be written as 
        $$M_{LD}^{SBT} = c(s) \\left(I+\\tau \\tau\\right)+\\left(I-3\\tau\\tau\\right)$$
        This method initializes $c(s)$ ONLY in the case of SBT, i.e., when RPYSpecialQuad,
        RPYDirectQuad, and RPYOversample have ALL been set to false. 
        The formula for the effective $c$ is to first set $\\eta=2s/L=1$, then 
        compute a weight function
        $$w(s;\\delta)=\\text{tanh}{\\left(\\frac{\\eta(s)+1}{\\delta}\\right)}- \\text{tanh}{\\left(\\frac{\\eta(s)-1}{\\delta}\\right)}-1,$$
        which is 1 near the fiber center and zero at the fiber ends. We then assign a regularized $s$
        $$\\bar{s}(s;\\delta)=w(s;\\delta)s+\\left(1-w(s;\\delta)^2\\right)\\frac{\\delta L}{2}$$
        on $0 \\leq s \\leq L/2$, with the corresponding reflection for $s > L/2$. The regularized 
        coefficient $\\delta$ is then given by 
        $$c(s;\\delta)=\\ln{\\left(\\frac{4\\bar{s}\\left(L-\\bar{s}\\right)}{\\left(\\epsilon L\\right)^2}\\right)}$$
        These coefficients are passed to the C++ code. When we use RPY, the coefficients 
        are computed directly in the C++ code. 
        
        Parameters
        ----------
        delta: double, optional
            The amount of regularization in the formulas above. The default is 0.1 if not provided.
        """
        if (self._RPYSpecialQuad or self._RPYDirectQuad or  self._RPYDirectQuad):
            self._sRegularized = self._sX;
            return;
        self._delta = delta;
        sNew = 0.5*self._L*np.ones(self._Nx);
        if (delta < 0.5):
            x = 2*self._sX/self._L-1;
            regwt = np.tanh((x+1)/delta)-np.tanh((x-1)/delta)-1;
            sNew = self._sX.copy();
            sNew[self._sX < self._L/2] =  regwt[self._sX < self._L/2]*self._sX[self._sX < self._L/2]+\
                (1-regwt[self._sX < self._L/2]**2)*delta*self._L/2;
            sNew[self._sX > self._L/2] = self._L-np.flip(sNew[self._sX < self._L/2]);
        self._sRegularized = sNew;       
           
    def averageTau(self,Xs):
        """
        The average tangent vector on a fiber, given by the formula
        $$\\bar{\\tau}  = \\frac{1}{L}\\int_0^L \\tau(s) ds$$
        
        Parameters
        -----------
        Xs: array
            Array of $\\tau$ at all the collocation points
            
        Returns
        -------
        array
            $\\bar{\\tau}$ as a $3 \\times 1$ array
        """
        avgTau = np.zeros(3);
        for p in range(self._Ntau):
            avgTau+= 1/(self._L)*Xs[p,:]*self._wTau[p];
        return avgTau
    
    def averageXsXs(self,Xs):
        """
        The average matrix $\\tau \\tau$ on a fiber, given by the formula
        $$\\bar{\\tau \\tau}  = \\frac{1}{L}\\int_0^L \\tau(s) \\tau(s) ds$$
        
        Parameters
        -----------
        Xs: array
            Array of $\\tau$ at all the collocation points
            
        Returns
        -------
        array
            $\\bar{\\tau \\tau}$ as a $3 \\times 3$ array
        """
        avgXsXs = np.zeros((3,3));
        for p in range(self._Ntau):
            avgXsXs+= 1/(self._L)*np.outer(Xs[p,:],Xs[p,:])*self._wTau[p];
        return avgXsXs
    
    def calcFibCurvature(self,X):
        """
        Calculate the absolute average curvature, given by the formula
        $$\\sqrt{\\frac{1}{L}\\int_0^L X_{ss}(s)\\cdot X_{ss}(s) ds}$$
        
        Parameters
        -----------
        X: array
            Array of positions $X$ at all the collocation points
            
        Returns
        -------
        double
            The mean $L^2$ fiber curvature according to the formula above.
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
        rigid=False,RPYSpecialQuad=False,RPYDirectQuad=False,RPYOversample=False,UseEnergyDisc=True, FPIsLocal=True,penaltyParam=0):
        """
        Initialize Chebyshev grids for $X$ and $\\tau$.
        
        In this discretization, which is described in Section 6.1 of Maxian's PhD thesis,
        the tangent vectors are always described on a type 1 Chebyshev grid of size $N$
        (this grid does not include the endpoints). Then the collocation points $X$
        are obtained on a grid of size $N_x=N+1$ by integrating the tangent vectors. 
        The type of grid for $X$ depends on how the elastic force is calculated - if we use rectangular spectral
        collocation, the collocation grid must be type 1 (no endpoints). See this paper:
        https://tobydriscoll.net/publication/driscoll-rectangular-spectral-collocation-2015/driscoll-rectangular-spectral-collocation-2015.pdf
        for explanation. 
        Otherwise, we use a grid of type 2 (endpoints included). 
        """
        super().__init__(L,epsilon,Eb,mu,N,deltaLocal,NupsampleForDirect,nptsUniform,rigid,RPYSpecialQuad,RPYDirectQuad,RPYOversample,UseEnergyDisc,FPIsLocal);
		# Chebyshev grid and weights
        self._sTau = cf.chebPts(self._Ntau,[0,self._L],chebGridType);
        self._wTau = cf.chebWts(self._Ntau,[0,self._L],chebGridType);
        self._XGridType = 1;
        if (UseEnergyDisc):
            self._XGridType = 2;
        self._sX = cf.chebPts(self._Nx,[0,self._L],self._XGridType);
        self._wX = cf.chebWts(self._Nx,[0,self._L],self._XGridType);
        self.initIs();
        self.calcXFromXsMap();
        self.initLocalcvals(deltaLocal);
        self.initD4BC(UseEnergyDisc,penaltyParam);
        self.initFPMatrix();
        self.initUpsamplingMatricesForDirectQuad();
        self.FindEigenvalueThreshold();
        MatfromNtoUniform = cf.ResamplingMatrix(self._nptsUniform,self._Nx,'u',self._XGridType);
        self._MatfromNtoUniform = np.zeros((3*self._nptsUniform,3*self._Nx));
        for iD in range(3):
            self._MatfromNtoUniform[iD::3,iD::3]=MatfromNtoUniform;
    
    def calcXFromXsMap(self):
        """
        Initialize the mapping _XonNp1Mat, which takes in the tangent vectors and
        fiber midpoint and returns the positions of the fiber collocation points. That is, 
        we can write the positions $X$ from the tangent vectors $\\tau$ and midpoint 
        $X_{mp}$ as 
        $$
        X =
        \\begin{pmatrix}
        D_{N+1}^\\dagger E_{N \\rightarrow N+1} & B 
        \\end{pmatrix} 
        \\begin{pmatrix}
        \\tau \\\\
        X_{mp} 
        \\end{pmatrix}
        :=\\mathcal{X}\\tau
        $$
        So the matrix $\\mathcal{X}$ is what we are computing here. It is made of two blocks, 
        the first block is the pseudo-inverse of the differentiation matrix on the grid of 
        size $N+1=N_x$, multiplied by the extension matrix which takes the tangent vectors from
        a grid of size $N$ to one of size $N_x=N+1$. The second block ensures that the midpoint
        of $X$ is $X_{mp}$.
        """
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
    
    def initUpsamplingMatricesForDirectQuad(self):
        """ 
        Initialize the up and downsampling matrices used when we do upsampled direct
        quadrature. Letting $N_{up}$ be the number of points we upsample to, 
        there are two matrices here. There is the matrix $E_{N_x \\rightarrow N_{up}}$,
        which takes the points and upsamples them. But there is the force upsampling
        matrix, which is the matrix which oversamples \\emph{forces}, and is given by
        $$E_{force} = W_{up} E_{N_x \\rightarrow N_{up}} \\widetilde{W}^{-1},$$
        where $$\\widetilde{W}^{-1} = E_{N_x \\rightarrow 2N_x}^T W_{2N_x} E_{N_x \\rightarrow 2N_x}$$
        is an inner product weights matrix, see (7.10) and(6.11) in Maxian's PhD dissertation. 
        
        The more complex upsampling of the forces can be thought of in a sequence of three steps:
        
        1) Convert to density by applying $\\widetilde{W}^{-1}$
        2) Oversample the force density (well defined in continuum) to an upsampled grid by 
           applying $E_{N_x \\rightarrow 2N_x}$
        3) Multiply by the weights $W_{up}$ on the oversampled grid to get force
        """
        self._wForDirect = cf.chebWts(self._nptsDirect, [0,self._L],self._XGridType);   
        self._EUpsample =  cf.ResamplingMatrix(self._nptsDirect, self._Nx,self._XGridType,self._XGridType);
        OneDResamp = np.dot(np.diag(self._wForDirect),np.dot(self._EUpsample,np.linalg.inv(self._WtildeNx)));
        self._OversamplingWtsMat = np.zeros((3*self._nptsDirect,3*self._Nx));
        for iD in range(3):
            self._OversamplingWtsMat[iD::3,iD::3]=OneDResamp; 
            
    def FindEigenvalueThreshold(self):
        """
        This method computes a threshold eigenvalue for the matrix $M_{SQ}$, describing
        the hydrodynamics on a single fiber computed via special quadrature on the RPY
        integral. The procedure is to oversample to $1/\\epsilon$ points, then form the
        mobility matrix 
        $$M_{ref} = \\widetilde{W}^{-1} E_{up} W_{up} M_{RPY, up} W_{up} E_{up} \\widetilde{W}^{-1}$$
        using $N_{up}=1/\\epsilon$ oversampled points on a straight fiber (the calculaton is 
        relatively insensitive to the fiber shape). We then return the smallest eigenvalue
        of this matrix, and use that as a threshold to modify any negative eigenvalues that 
        come about as a result of special quadrature.
        Note that if the mobility is RPYDirectQuad or RPYOversample, we set the eigenvalue 
        threshold to zero, since in those two cases we are guaranteed to have an SPD matrix 
        and never need eigenvalue truncation.
        """
        # We only want to set the eigenvalue threshold when we do special quad (it doesn't really matter)
        self._EigValThres = 0.001 # Could cause convergence issues
        if (self._RPYDirectQuad or self._RPYOversample):
            self._EigValThres = 0;
        return
        if ((not self._RPYDirectQuad) and (not self._RPYOversample)):
            NToUpsamp = int(1/self._epsilon);
            # Compute mobility for straight fiber on upsampled grid
            sUp = cf.chebPts(NToUpsamp,[0,self._L],self._XGridType)
            wUp = cf.chebWts(NToUpsamp,[0,self._L],self._XGridType)
            Ext =  cf.ResamplingMatrix(NToUpsamp, self._Nx,self._XGridType,self._XGridType);
            OneDResamp = np.dot(np.diag(wUp),np.dot(Ext,np.linalg.inv(self._WtildeNx)));
            OversamplingWtsMat = np.zeros((3*NToUpsamp,3*self._Nx));
            for iD in range(3):
                OversamplingWtsMat[iD::3,iD::3]=OneDResamp; 
            # Compute minimum eigenvalue  
            StraightFiber = np.concatenate((np.reshape(sUp,(NToUpsamp,1)),np.zeros((NToUpsamp,1)),np.zeros((NToUpsamp,1))),axis=1);
            MRPY = RPYVelocityEvaluator.RPYMatrix(NToUpsamp,StraightFiber,self._mu,self._a)
            MRPYOversamp = np.dot(OversamplingWtsMat.T,np.dot(MRPY,OversamplingWtsMat));
            self._EigValThres = SpecialQuadEigenvalueSafetyFactor*np.min(np.linalg.eigvalsh(MRPYOversamp));
        print('Eigenvalue threshold %1.8E' %self._EigValThres)
    
    def UniformUpsamplingMatrix(self,Nunipts,typ='u'):
        """
        Compute a set of uniform points and return the matrix that 
        oversamples to those uniform points
        
        Parameters
        -----------
        Nunipts: int
            Number of uniform points
        typ: char, optional
            Default is 'u', in which case $\\Delta s =L/(N_u-1)$ and 
            we get points like 0, $\\Delta s$, 2$\\Delta s$, ...
            Otherwise, $\\Delta s =L/N_u$ and we get $\\Delta s/2$, $3\\Delta s/2$, ....
        
        Returns
        --------
        (array, array)
            The arclength coordinates su of the uniform points and the 
            matrix $R$ that resamples the Chebyshev point coordinates to
            the uniform point coordinates
        """
        if (typ=='u'):
            ds = self._L/(Nunipts-1);
            su = np.arange(0,Nunipts)*ds;
        else:
            ds = self._L/Nunipts;
            su = np.arange(0.5,Nunipts)*ds;
        return su, cf.ResamplingMatrix(Nunipts, self._Nx,typ,self._XGridType); 
    
    def ResampleFromOtherGrid(self,XOther,Nother,typeOther):
         return np.dot(cf.ResamplingMatrix(self._Nx,Nother,self._XGridType,typeOther),XOther); 
    
    def SampleToOtherGrid(self,XSelf,Nother,typeOther):
         return np.dot(cf.ResamplingMatrix(Nother,self._Nx,typeOther,self._XGridType),XSelf); 
         
    def ForceFromForceDensity(self,ForceDen):
        return np.dot(self._WtildeNx,ForceDen);
    
    def DiffXMatrix(self):
        return self._DXGrid; 
    
    def calcXFromXsAndMP(self,Xs,XMP):
        """
        Computes Chebyshev point positions $X$ from tangent vectors
        $\\tau$ and fiber midpoint $X_{mp}$ by applying the matrix 
        $\\mathcal{X}$ (see method calcXFromXsMap)
        
        Parameters
        ----------
        Xs: array
            $3N$ array of tangent vectors
        Xmp: array
            $3$ array of the fiber midpoint
            
        Returns
        -------
        array
            $3N_x=3(N+1)$ array of the positions at the collocation points
        """
        return np.dot(self._XonNp1Mat,np.concatenate((Xs,XMP)));  
    
    def calcXsAndMPFromX(self,X):
        """
        This is the inverse of the previous method.
        Computes tangent vectors $\\tau$ and fiber midpoint $X_{mp}$
        from Chebyshev point positions $X$ by applying the matrix
        $\\mathcal{X}^{-1}$ (see method calcXFromXsMap)
        
        Parameters
        ----------
        X: array
            $3N_x=3(N+1)$ array of the positions at the collocation points
            
        Returns
        -------
        (array, array)
            $3N$ array of tangent vectors $\\tau$ followed by
            $3$ array of the fiber midpoint
        """
        XsXMP = np.linalg.solve(self._XonNp1Mat,X);    
        Xs = XsXMP[:3*self._Ntau];
        XMP = XsXMP[3*self._Ntau:];
        return Xs, XMP;
    
    def gets(self):
        return self._sX;
    
    def getw(self):
        return self._wX; 
    
    def initD4BC(self,UseEnergyDisc,penaltyParam):
        """
        The elastic force can be written as $F^\\kappa=FX$, and likewise the
        force density can be written as $f^\\kappa = \\widetilde{W}^{-1}FX:=F_{den}X$, where
        $$\\widetilde{W}^{-1} = E_{N_x \\rightarrow 2N_x}^T W_{2N_x} E_{N_x \\rightarrow 2N_x}$$
        is the inner product matrix that converts between force and force density. There
        are two ways to compute these matrices:
        
        1) The energy discretization, in which case we first discretize the energy
           $$\\mathcal{E}_{b} = \\frac{\\kappa}{2} \\int_0^L X_{ss}(s) \\cdot X_{ss}(s) ds$$
           via $\mathcal{E}_b = 1/2 X^T L X$, where
           $$L = \\kappa \\left(D^2\\right)^T \\widetilde{W} D^2$$
           and then set the force matrix $F=-L$. In this case, we obtain the force 
           matrix (self._D4BCForce), then compute the force density matrix (self._D4BC)
           by setting $F_{den}=\\widetilde{W}^{-1}F$. This is also the default way to handle
           elastic forces.
        2) Rectangular spectral collocation, in which case the formula for the force matrix
           is more complicated. In brief, we solve for a configuration on an $N_x+4$ point grid
           that satisfies the BCs, then compute elastic force on that grid, and downsample to obtain
           a matrix
           $$F_{den}= -\\kappa R D^4_{N_x+4} E,$$
           where $E$ is the extension matrix that accounts for the BCs, $D^4_{N_x+4}$ is the
           differentiation matrix on the upsampled grid, and $R$ restricts the result to the 
           original grid.
           This is summarized in (6.17-19) in Maxian's PhD thesis. What results is the force
           density matrix, from which we obtain the force matrix by setting $F=\\widetilde{W}F_{den}$.
        
        In the case when there is a penalty force keeping the fiber position, the force 
        density matrix is modified as $F_{den}-P I$, and the force matrix is $F-P\\widetilde{W}$, 
        where $P$ is the penalty parameter.
        
        Parameters
        -----------
        UseEnergyDisc: bool
            If true, we use the energy discretization (option 1 above). 
            Otherwise, we use rectangular spectral collocation (option 2).
        penaltyParam: double
            The strength of the penalty force $P$.
        """
        sDouble = cf.chebPts(2*self._Nx,[0,self._L],self._XGridType);
        wDouble = cf.chebWts(2*self._Nx,[0,self._L],self._XGridType);
        R_Nx_To_2x = cf.ResamplingMatrix(2*self._Nx,self._Nx,self._XGridType,self._XGridType);
        DSq = cf.diffMat(2,[0,self._L],self._Nx,self._Nx,self._XGridType,self._XGridType);
        self._WtildeNx = np.dot(R_Nx_To_2x.T,np.dot(np.diag(wDouble),R_Nx_To_2x));
        if (UseEnergyDisc):
            OneD4BCForce = -self._Eb*np.dot(DSq.T,np.dot(self._WtildeNx,DSq));
            # Convert to force density by inverting by Wtilde
            OneD4BC = np.linalg.solve(self._WtildeNx,OneD4BCForce);
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
            OneD4BCForce = np.dot(self._WtildeNx, OneD4BC);
            # Convert to force by multipying by Wtilde
            # OneD4BC = np.dot(self._WtildeNx,OneD4BC);
        if (penaltyParam > 0):
            OneD4BC = OneD4BC - penaltyParam*np.eye(self._Nx); 
            OneD4BCForce = OneD4BCForce - penaltyParam*self._WtildeNx;
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
            self._D4BCForceHalf[iD::3,iD::3]=np.real(OneD4BCForceHalf); 
            self._stackWTilde_Nx[iD::3,iD::3] =  self._WtildeNx;    
            self._stackWTildeInverse_Nx[iD::3,iD::3] =  np.linalg.inv(self._WtildeNx);   
    
    def calcBendMatX0(self,X0,penaltyParam):
        """
        For penalty-forced fibers, we need to subtract $FX_0$ from the
        force so that there is no elastic force in the reference configuration
        $X_0$. This method computes $FX_0$.
        """
        self._BendMatX0 = np.dot(self._D4BCForce,X0);
        if (abs(penaltyParam) < 1e-10):
            self._BendMatX0 = 0*self._BendMatX0;  
    
    def StackedValsToCoeffsMatrix(self):
        """
        Returns
        --------
        array
            The matrix mapping the Chebyshev values
            to their coefficients on the grid of size $N_x$. This matrix
            is stacked so that it operates on vectors of size $3N_x$.
        """
        CToV = cf.CoeffstoValuesMatrix(self._Nx,self._Nx,self._XGridType);
        VToC = np.linalg.inv(CToV);
        return VToC;
    
    def initFPMatrix(self):
        """
        Precompute an auxillary matrix for special quadrature. In brief, the special
        quadrature schemes all wind up computing $$\\left(V^{-1}g\\right)^{-T} q,$$ where $g$ is a smooth
        function which expresses the behavior with the nonsmooth part removed, $q$ is
        a set of integrals of the nonsmooth function (usually sign(s-s')) against Chebyshev 
        polynomials, and $V$ is the matrix which computes coefficients from values. 
        The integral we want can be written as
        $$g^T V^{-T} q.$$ 
        Thus, in this method, we compute the matrix $V^{-T} q$ and store it for later use.
        When we do RPY special quadrature, we also precompute the same matrix for 
        special quadrature for the doublet here, and also the matrices needed to upsample
        the positions and forces to a Gauss-Legendre grid of size $2\\hat{a}$. The number of 
        points we resample these integrals to is set in the constructor.
        
        More information can be found in Section 6.2.1 of Maxian's PhD thesis.
        """
        sscale=-1+2*self._sX/self._L;
        ChebCoeffsToVals = cf.CoeffstoValuesMatrix(self._Nx,self._Nx,self._XGridType);
        AllQs = np.zeros((self._Nx,self._Nx));
        AllDs = np.zeros((self._Nx,self._Nx));
        self._DoubletFPMatrix = np.zeros((self._Nx,self._Nx));
        self._RLess2aResamplingMat = np.zeros((self._Nx*self._Nsmall,self._Nx));
        self._RLess2aWts = np.zeros(self._Nx*self._Nsmall);
        a = 0;
        if (self._RPYSpecialQuad):
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
        if (self._RPYSpecialQuad):
            self._DoubletFPMatrix =  1.0/(8.0*np.pi*self._mu)*2.0/self._L*np.linalg.solve(ChebCoeffsToVals.T,AllDs.T);
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
