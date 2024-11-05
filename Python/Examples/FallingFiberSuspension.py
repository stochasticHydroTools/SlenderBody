from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter, RPYVelocityEvaluator
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from StericForceEvaluator import StericForceEvaluator, SegmentBasedStericForceEvaluator
from FileIO import prepareOutFile
import numpy as np
import sys

"""
Falling fiber suspension, which appears in Section 6.2.4 of Maxian & Donev (2024). 
"""

# Inputs 
nFib=400          # number of fibers
N=16;           # number of tangent vectors per fiber
Lf=1            # length of each fiber
nonLocal=True     # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=4          # length of the periodic domain
mu=1            # fluid viscosity
eps=4e-3        # slenderness ratio
kbT = 1e-2;
Eb=1e-2;       # fiber bending stiffness
tscale = Lf**4*mu/Eb;
omega=0         # frequency of oscillations
gam0=0          # base rate of strain
gDen=5;
tf = 0.1*tscale/gDen;
dt = 5e-6*tscale/gDen;
giters = 1;   # Number of GMRES iterations. The number of iterations on the residual hydrodynamic system is giters-1. 
              # So if this number is set to 1 it will do hydrodynamic interactions explicitly by time lagging.
seed=1;
nThr=12;
FluctuatingFibs=True;

Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
RPYQuad=True
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,\
    RPYOversample=(not RPYQuad),NupsampleForDirect=100);
FatCor=True; # whether to do a correction for the fat mobility
eps_Star = 1e-2*4/np.exp(1.5);
fibDiscFat = ChebyshevDiscretization(Lf, eps_Star,Eb,mu,N,\
    NupsampleForDirect=100,RPYOversample=(not RPYQuad),RPYSpecialQuad=RPYQuad);
if (not FatCor):
    fibDiscFat=None;

# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,fibDiscFat=fibDiscFat,nThreads=nThr);
else:
    allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,0,fibDiscFat=fibDiscFat);
    
# Initialize the fiber list (straight fibers)
nStericPts = int(1/eps); # Used for contact checking / pre-computations only
NsegForSterics = 10;
if (NsegForSterics > 0):
    StericEval = SegmentBasedStericForceEvaluator(nFib,fibDisc._Nx,nStericPts,fibDisc,allFibers._ptsCheb, Dom, eps*Lf,kbT,NsegForSterics,nThreads=nThr);
else: 
    StericEval = StericForceEvaluator(nFib,fibDisc._Nx,nStericPts,fibDisc,allFibers._ptsCheb, Dom, eps*Lf,kbT,nThr);
    
np.random.seed(seed);
fibList = [None]*nFib;
allFibers.RSAFibers(fibList,Dom,StericEval,nDiameters=2);

# Initialize Ewald for non-local velocities
#Ewald = RPYVelocityEvaluator(fibDisc._a,mu,fibDisc._nptsDirect*nFib);
Ewald = None;
if (nonLocal):
    totnumDir = fibDisc._nptsDirect*nFib;
    xi = 3*totnumDir**(1/3)/Ld; # Ewald param
    RPYRadius = fibDisc._a;
    if (FatCor):
        RPYRadius = fibDiscFat._a; 
    Ewald = GPUEwaldSplitter(RPYRadius,mu,xi,Dom,fibDisc._nptsDirect*nFib);
    HydroStr='Per';

# Initialize the temporal integrator
if (FluctuatingFibs):
    TIntegrator = MidpointDriftIntegrator(allFibers);
else:
    TIntegrator = BackwardEuler(allFibers);
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(giters);

# Prepare the output file and write initial locations
FileString='D05400FallingDETNLHydro_G'+str(gDen)+'_N'+str(N)+'SQFat100_'+str(seed)+'.txt'
allFibers.writeFiberLocations(FileString,'w');
saveEvery = 10;

stopcount = int(tf/dt+1e-10);
ItsNeed = np.zeros(stopcount);
numSaves = stopcount//saveEvery+1;
nContacts = np.zeros(numSaves);
_, _, FibContacts=StericEval.CheckContacts(allFibers._ptsCheb,Dom, excludeSelf=True);
nCont, _ = FibContacts.shape;
nContacts[0] = nCont;

# Time loop
for iT in range(stopcount): 
    wr=0;
    if ((iT % saveEvery)==saveEvery-1):
        print('Time %f' %(float(iT+1)*dt));
        wr=1;
    maxX, ItsNeed[iT],_= TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravden=-gDen,StericEval=StericEval,write=wr,outfile=FileString);
    if (wr):
        _, _, FibContacts=StericEval.CheckContacts(allFibers._ptsCheb,Dom, excludeSelf=True);
        nContacts[iT], _ = FibContacts.shape;
        print('Number of contacts %d' %nContacts[iT])
        print('Number of iterations %d' %ItsNeed[iT])
    
np.savetxt('ItsNeeded_'+FileString,ItsNeed);
np.savetxt('nContacts_'+FileString,nContacts);
