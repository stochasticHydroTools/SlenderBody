from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter, RPYVelocityEvaluator
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler, MidpointDriftIntegrator
from CrossLinkedNetwork import CrossLinkedNetwork
from DiscretizedFiber import DiscretizedFiber
from FileIO import prepareOutFile
import numpy as np
import sys

"""
Two falling fibers example; see Section 6.2 of Maxian & Donev (2024)
"""

def makeTwoFalling(Lf,N,fibDisc):
    """
    Initialize the two fibers for the simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    d=0.5; # initial separation
    Xs13 = np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T;
    fibList = [None]*2;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([0,0,d]));
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([0,0,0]));
    return fibList;

# Inputs 
nFib=2          # number of fibers
N=int(sys.argv[1]);           # number of tangent vectors per fiber
Lf=1            # length of each fiber
nonLocal=True     # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=10          # length of the periodic domain
mu=1            # fluid viscosity
eps=1e-3        # slenderness ratio
kbT = 1e-2;
Eb=1e-2;       # fiber bending stiffness
tscale = Lf**4*mu/Eb;
omega=0         # frequency of oscillations
gam0=0          # base rate of strain
gDen=5;
if (gDen==0):
    tf=0.1/5*tscale;
    if (nonLocal):
        dt=5e-6*tscale/5
    else:
        dt=1e-5*tscale/5 # 1e-5 for local, 5e-6 for nonlocal (incl deterministic)
else:
    tf=0.1*tscale/gDen;       # final time
    if (nonLocal):
        dt=5e-6*tscale/gDen
    else:
        dt=1e-5*tscale/gDen # 1e-5 for local, 5e-6 for nonlocal (incl deterministic)   
giters = 1;   # Number of GMRES iterations. The number of iterations on the residual hydrodynamic system is giters-1. 
              # So if this number is set to 1 it will do hydrodynamic interactions explicitly by time lagging.
HydroTyp=int(sys.argv[2]); 
seed=int(sys.argv[3]); 

FluctuatingFibs=True;

Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
RPYQuad=True
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,\
    RPYOversample=(not RPYQuad),NupsampleForDirect=100);
FatCor=True;
eps_Star = 1e-2*4/np.exp(1.5);
fibDiscFat = ChebyshevDiscretization(Lf, eps_Star,Eb,mu,N,\
    NupsampleForDirect=100,RPYOversample=(not RPYQuad),RPYSpecialQuad=RPYQuad);
if (not FatCor):
    fibDiscFat=None;
    
# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,fibDiscFat=fibDiscFat,nThreads=2);
else:
    allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,0,fibDiscFat=fibDiscFat);
fibList = makeTwoFalling(Lf,N,fibDisc);
allFibers.initFibList(fibList,Dom);
allFibers.fillPointArrays();

np.random.seed(seed);
# Initialize Ewald for non-local velocities
#Ewald = RPYVelocityEvaluator(fibDisc._a,mu,fibDisc._nptsDirect*nFib);
totnumDir = fibDisc._nptsDirect*nFib;
xi = 3*totnumDir**(1/3)/Ld; # Ewald param
RPYRadius = fibDisc._a;
if (FatCor):
    RPYRadius = fibDiscFat._a; 
if (HydroTyp==1):
    Ewald = GPUEwaldSplitter(RPYRadius,mu,xi,Dom,fibDisc._nptsDirect*nFib);
    HydroStr='Per';
else:
    Ewald = RPYVelocityEvaluator(RPYRadius,mu,fibDisc._nptsDirect*nFib);
    HydroStr='Free';

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
FileString='D05TwoFallingBRNLocHydro_G'+str(gDen)+'_N'+str(N)+'OS2000_Per.txt'
allFibers.writeFiberLocations(FileString,'w');
saveEvery = 1;

# Time loop
stopcount = int(tf/dt+1e-10);
for iT in range(stopcount): 
    wr=0;
    if ((iT % saveEvery)==saveEvery-1):
        print('Time %f' %(float(iT+1)*dt));
        wr=1;
    maxX,itsneeded,_= TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravden=-gDen,write=wr,outfile=FileString);
    

