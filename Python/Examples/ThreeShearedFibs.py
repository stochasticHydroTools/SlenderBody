from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter, RPYVelocityEvaluator
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler
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
    Xs2 = np.concatenate(([np.zeros(N)],[np.ones(N)],[np.zeros(N)]),axis=0).T;
    fibList = [None]*3;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([0,-0.6,-0.04]));
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(Xs2,3*N),np.zeros(3));
    fibList[2] = DiscretizedFiber(fibDisc,np.reshape(Xs13,3*N),np.array([0,0.6,0.06]));
    return fibList;

# Inputs 
nFib=3          # number of fibers
N=16;           # number of tangent vectors per fiber
Lf=2            # length of each fiber
nonLocal=1      # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=2.4          # length of the periodic domain
mu=1            # fluid viscosity
eps=1e-3        # slenderness ratio
Eb=1e-2        # fiber bending stiffness
dt=0.01;       # timestep
omega=0         # frequency of oscillations
gam0=1          # base rate of strain
tf=2.4        # final time
giters = 1;   # Number of GMRES iterations. The number of iterations on the residual hydrodynamic system is giters-1. 
              # So if this number is set to 1 it will do hydrodynamic interactions explicitly by time lagging. 

Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=True,\
    RPYDirectQuad=False,RPYOversample=False,NupsampleForDirect=40,FPIsLocal=True);

# Initialize the master list of fibers
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
    maxX,_,_,_ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,write=True,outfile=FileString);
    print(maxX)


