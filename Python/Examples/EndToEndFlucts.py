from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter
import numpy as np
from math import exp
import sys

"""
This file runs trajectories of free fibers which go from straight to fluctuating in their equilibrium state. 
See Sections 4 and 5 in 
"Semiflexible bending fluctuations in inextensible slender filaments in Stokes flow: towards a spectral discretization"
by O. Maxian, B. Sprinkle, and A. Donev. 

As written right now, this function takes two command line arguments: 
the -log of epsilonRPY (so 2 for epsRPY = 1e-2, 3 for epsRPY=1e-3),
and the random seed. So calling
python EndToEndFlucts.py 2 3
would simulate fibers with aRPY/L = 1e-2 and seed = 3. 
"""

def makeStraightFibs(nFib,Lf,N,fibDisc,Ld=0):
    """
    Initialize a list of straight fibers that we will simulate 
    """
    # Falling fibers
    Xs = (np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T);
    Xs = np.reshape(Xs,3*N);
    XMP = np.array([0,Ld/nFib,0]);
    fibList = [None]*nFib
    for iFib in range(nFib):
        fibList[iFib] = DiscretizedFiber(fibDisc,Xs,XMP);
    return fibList;

# Inputs 
nFib=100         # number of fibers
N=12          # number of points per fiber
Lf=2            # length of each fiber
nonLocal=4   # doing nonlocal solves? 0 = local drag, > 1 = nonlocal hydro on each fiber.
nThr = 8;   # Number of OpenMP threads
# Mobility options (can do SBT if all are set to false, otherwise some variant of RPY as described below)
MobStr='NLOS64';
RPYQuad = True;        # Special quadrature
RPYDirect = False;       # Direct Clenshaw-Curtis quadrature
RPYOversample = False;  # Oversampled quad
NupsampleForDirect = 64; # Number of pts for oversampled quad
Ld=float(sys.argv[4]);        # length of the periodic domain
mu=1            # fluid viscosity
logeps = float(sys.argv[1]);
eps=10**(-logeps)*4/exp(1.5);       # slenderness ratio (aRPY/L=1e-3). The factor 4/exp(1.5) converts to actual fiber slenderness. 
lpstar = 10;    
kbT = 4.1e-3;
Eb=lpstar*kbT*Lf;         # fiber bending stiffness
tfund = 0.003*4*np.pi*mu*Lf**4/(np.log(10**logeps)*Eb) 
tf = 10*tfund;
dtfund = float(sys.argv[3]);
dt = tfund*dtfund;
penaltyParam = 0;
seed = int(sys.argv[2]);
#nSaves = 100; # target number
saveEvery = max(np.floor(0.01/dtfund),1);#int(tf/(nSaves*dt)+1e-6);

saveStr='NLTol2Eps'+str(logeps)+MobStr+'_N'+str(N)+'_Ld'+str(Ld)+'_Lp'+str(lpstar)+'_dtf'+str(dtfund)+'_'+str(seed)+'.txt'
FileString = 'SemiflexFlucts/Locs'+saveStr;
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,deltaLocal=1,\
    RPYDirectQuad=RPYDirect,RPYOversample=RPYOversample,NupsampleForDirect=NupsampleForDirect);
    
# Initialize the master list of fibers
allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,0,0,Dom,kbT,nThreads=nThr);
#fibList = makeStraightFibs(nFib,Lf,N,fibDisc,Ld);
np.random.seed(seed);
fibList = [None]*nFib;
allFibers.initFibList(fibList,Dom);

Ewald=None;
if (nonLocal==1):
    totnumDir = fibDisc._nptsDirect*nFib;
    xi = 3*totnumDir**(1/3)/Ld; # Ewald param
    Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,totnumDir);

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
    itsNeeded[iT]=its;
np.savetxt('SemiflexFlucts/ItsNeeded'+saveStr,itsNeeded)
np.savetxt('SemiflexFlucts/LanczosNeeded'+saveStr,np.array(TIntegrator._nLanczos))
        
