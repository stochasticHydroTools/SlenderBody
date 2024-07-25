from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter
import numpy as np
from math import exp
import sys, os

"""
This file runs trajectories of free fibers which go from straight to fluctuating in their equilibrium state. 
See Sections 4 and 5 in 
"Semiflexible bending fluctuations in inextensible slender filaments in Stokes flow: towards a spectral discretization"
by O. Maxian, B. Sprinkle, and A. Donev. 

This can also be found in Section 8.4 of Maxian's PhD thesis.

The goal of this is to measure the equilibrium distribution of fiber lengths and make sure that it matches that
obtained from MCMC. 
"""

# Inputs 
nFib = 200         # number of fibers
N = 12          # number of points per fiber
Lf = 1            # length of each fiber
nonLocal = True
nThr = 16;   # Number of OpenMP threads
# Mobility options (can do SBT if all are set to false, otherwise some variant of RPY as described below)
RPYQuad = False;        # Special quadrature
NupsampleForDirect = 500; # Number of pts for oversampled quad
Ld=2;        # length of the periodic domain
mu=1            # fluid viscosity
logeps = 3;
eps=10**(-logeps)*4/exp(1.5);       # slenderness ratio (aRPY/L=1e-3). The factor 4/exp(1.5) converts to actual fiber slenderness. 
lpstar = 1;    
kbT = 4.1e-3;
Eb=lpstar*kbT*Lf;         # fiber bending stiffness
tfund = 0.003*4*np.pi*mu*Lf**4/(np.log(10**logeps)*Eb) 
tf = 10*tfund;
dtfund = 0.001;
dt = tfund*dtfund;
print(dt)
penaltyParam = 0;
seed = int(sys.argv[1]);
saveEvery = max(np.floor(0.01/dtfund),1);#int(tf/(nSaves*dt)+1e-6);

MobStr = 'OS'+str(NupsampleForDirect);
if (RPYQuad):
    MobStr = 'SQStar'

if not os.path.exists('SemiflexFlucts'):
    os.makedirs('SemiflexFlucts')

saveStr='Eps'+str(logeps)+MobStr+'_N'+str(N)+'_Ld'+str(Ld)+'_Lp'+str(lpstar)+'_dtf'+str(dtfund)+'_'+str(seed)+'.txt'
FileString = 'SemiflexFlucts/Locs'+saveStr;
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Define discretization and fiber collection
fibDisc = ChebyshevDiscretization(Lf, eps,Eb,mu,N,NupsampleForDirect=NupsampleForDirect,RPYOversample=(not RPYQuad),RPYSpecialQuad=RPYQuad);
Ewald = None;
if (RPYQuad and nonLocal):
    eps_Star = 1e-2*4/np.exp(1.5);
    fibDiscFat = ChebyshevDiscretization(Lf, eps_Star,Eb,mu,N,NupsampleForDirect=NupsampleForDirect,RPYOversample=(not RPYQuad),RPYSpecialQuad=RPYQuad);
    allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,0,0,Dom,kbT,nThreads=nThr,fibDiscFat=fibDiscFat);
    if (nonLocal):
        totnumDir = fibDiscFat._nptsDirect*nFib;
        xi = 3*totnumDir**(1/3)/Ld; # Ewald param
        Ewald = GPUEwaldSplitter(fibDiscFat._a,mu,xi,Dom,fibDiscFat._nptsDirect*nFib);
else:
    allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,0,0,Dom,kbT,nThreads=nThr);
    if (nonLocal):
        totnumDir = fibDisc._nptsDirect*nFib;
        xi = 3*totnumDir**(1/3)/Ld; # Ewald param
        Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,fibDisc._nptsDirect*nFib);
    
np.random.seed(seed);
fibList = [None]*nFib;
allFibers.initFibList(fibList,Dom);

# Initialize the temporal integrator
TIntegrator = MidpointDriftIntegrator(allFibers);

# Prepare the output file and write initial locations
np.random.seed(seed);
allFibers.writeFiberLocations(FileString,'w');
        
# Time loop
stopcount = int(tf/dt+1e-10);
itsNeeded = np.zeros(stopcount,dtype=np.int64);
for iT in range(stopcount): 
    wr = 0;
    if ((iT % saveEvery) == (saveEvery-1)):
        wr=1;
        print('Fraction done %f' %((iT+1)/stopcount))
    maxX, its, _, _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,write=wr,outfile=FileString);
    print(its)
    itsNeeded[iT]=its;
np.savetxt('SemiflexFlucts/ItsNeeded'+saveStr,itsNeeded)
        
