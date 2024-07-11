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
Three sheared fibers example. See Section 7.4.2 in Maxian's PhD thesis 
for more information on the set-up. 
"""

def makeThreeSheared(Lf,N,fibDisc):
    """
    Initialize the three fibers for the shear simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    Xs13 = np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T;
    fibList = [None]*3;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([-0.75,0,0]));
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([0.75,0,0]));
    return fibList;

# Inputs 
nFib=2          # number of fibers
N=16;           # number of tangent vectors per fiber
Lf=1            # length of each fiber
nonLocal=1      # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=4          # length of the periodic domain
mu=1            # fluid viscosity
eps=4e-3        # slenderness ratio
Eb=1e-3        # fiber bending stiffness
dt=0.05;       # timestep
omega=0         # frequency of oscillations
gam0=0          # base rate of strain
tf=20        # final time
giters = 1;   # Number of GMRES iterations. The number of iterations on the residual hydrodynamic system is giters-1. 
              # So if this number is set to 1 it will do hydrodynamic interactions explicitly by time lagging. 
FluctuatingFibs=True;

Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=False,\
    RPYDirectQuad=False,RPYOversample=True,NupsampleForDirect=200,FPIsLocal=True);

# Initialize the master list of fibers
if (FluctuatingFibs):
    kbT = 1e-3;
    allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT);
else:
    allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,0);
fibList = makeThreeSheared(Lf,N,fibDisc);
allFibers.initFibList(fibList,Dom);
allFibers.fillPointArrays();

# Initialize Ewald for non-local velocities
#Ewald = RPYVelocityEvaluator(fibDisc._a,mu,fibDisc._nptsDirect*nFib);
totnumDir = fibDisc._nptsDirect*nFib;
xi = 3*totnumDir**(1/3)/Ld; # Ewald param
Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,fibDisc._nptsDirect*nFib);


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
FileString='ThreeShearedLocations.txt'
allFibers.writeFiberLocations(FileString,'w');

# Time loop
stopcount = int(tf/dt+1e-10);
for iT in range(stopcount): 
    print('Time %f' %(float(iT)*dt));
    maxX,_,_,_ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravden=-0.1,write=True,outfile=FileString);
    print(maxX)

