from fiberCollectionNew import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretizationNew import ChebyshevDiscretization
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler
from DiscretizedFiberNew import DiscretizedFiber
import numpy as np
from math import exp
import sys

"""
This file runs trajectories of free fibers which go from straight to fluctuating in their equilibrium state. 
See Sections 4 and 5 in 
"Semiflexible bending fluctuations in inextensible slender filaments in Stokes flow: towards a spectral discretization"
by O. Maxian, B. Sprinkle, and A. Donev. 

As written right now, this function takes three command line arguments: 
the -log of epsilonRPY (so 2 for epsRPY = 1e-2, 3 for epsRPY=1e-3), the time step size dt,
and the random seed. So calling
python EndToEndFlucts.py 2 0.0001 3
would simulate fibers with aRPY/L = 1e-2, dt = 0.0001, and seed = 3. 
"""

def makeStraightFibs(nFib,Lf,N,fibDisc):
    """
    Initialize a list of straight fibers that we will simulate 
    """
    # Falling fibers
    Xs = (np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T);
    Xs = np.reshape(Xs,3*N);
    XMP = np.zeros(3);
    fibList = [None]*nFib
    for iFib in range(nFib):
        fibList[iFib] = DiscretizedFiber(fibDisc,Xs,XMP);
    return fibList;

# Inputs 
nFib=10          # number of fibers
N=12          # number of points per fiber
Lf=2            # length of each fiber
nonLocal=4   # doing nonlocal solves? 0 = local drag, > 1 = nonlocal hydro on each fiber.
nThr = 8;   # Number of OpenMP threads
# Mobility options (can do SBT if all are set to false, otherwise some variant of RPY as described below)
MobStr='CC';
RPYQuad = False;        # Special quadrature
RPYDirect = True;       # Direct Clenshaw-Curtis quadrature
RPYOversample = False;  # Oversampled quad
NupsampleForDirect = 20; # Number of pts for oversampled quad
FluctuatingFibs = True;
RigidDiffusion = False;
rigidDetFibs = False;
Ld=10        # length of the periodic domain (not relevant here)
mu=1            # fluid viscosity
logeps = float(sys.argv[1]);
eps=10**(-logeps)*4/exp(1.5);       # slenderness ratio (aRPY/L=1e-3). The factor 4/exp(1.5) converts to actual fiber slenderness. 
lpstar = 1;
dt = float(sys.argv[2]);
eigvalThres = 0.0;
if (eps < 5e-3): # Only used for special quad
    if (N==12):
        eigvalThres = 3.2/Lf; 
    elif (N==24):
        eigvalThres = 5.0/Lf; 
    elif (N==36):
        eigvalThres = 6.7/Lf; 
else:   # Only used for special quad
    if (N==12):
        eigvalThres = 1.6/Lf; 
    elif (N==24):
        eigvalThres = 1.0/Lf; 
    elif (N==36):
        eigvalThres = 0.34/Lf; 
    
kbT = 4.1e-3;
Eb=lpstar*kbT*Lf;         # fiber bending stiffness
tf = 0.01*mu*Lf**4/(np.log(10**logeps)*Eb) # extent longer if you want to simulate eq fluctuations
penaltyParam = 0;
seed = int(sys.argv[3]);
nSaves = 100; # target number
saveEvery = int(tf/(nSaves*dt));

FileString = 'FreeFib.txt'#'SemiflexFlucts/Eps'+str(logeps)+'EndToEnd'+MobStr+'_N'+str(N)+'_Lp'+str(lpstar)\
    #+'_dt'+str(dt)+'_'+str(seed)+'.txt'
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,deltaLocal=1,\
    RPYDirectQuad=RPYDirect,RPYOversample=RPYOversample,NupsampleForDirect=NupsampleForDirect);
    
# Initialize the master list of fibers
allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,0,0,Dom,kbT,eigvalThres,nThreads=nThr);
fibList = makeStraightFibs(nFib,Lf,N,fibDisc);
allFibers.initFibList(fibList,Dom);

# Initialize the temporal integrator
TIntegrator = BackwardEuler(allFibers,FPimp=(nonLocal > 0));
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(1);

# Prepare the output file and write initial locations
np.random.seed(seed);
allFibers.writeFiberLocations(FileString,'w');
        
# Time loop
stopcount = int(tf/dt+1e-10);
print(stopcount)
for iT in range(stopcount): 
    wr = 0;
    if ((iT % saveEvery) == (saveEvery-1)):
        wr=1;
        print((iT+1)/stopcount)
    TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,write=wr,outfile=FileString);
        