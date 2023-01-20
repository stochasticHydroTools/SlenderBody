from fiberCollectionNew import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretizationNew import ChebyshevDiscretization
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler
from DiscretizedFiberNew import DiscretizedFiber
from FileIO import prepareOutFile
import numpy as np
import chebfcns as cf
from math import exp
import sys

"""
This file performs the three sheared fiber test in Section 5.1.2
"""

def makeCurvedFib(Lf,N,fibDisc):
    """
    Initialize the three fibers for the shear simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    q=1;
    s=fibDisc._sTau;
    # Falling fibers
    Xs = (np.concatenate(([np.cos(q*s**3 * (s-Lf)**3)],[np.sin(q*s**3*(s - Lf)**3)],[np.ones(N)]),axis=0).T)/np.sqrt(2);
    Xs = np.reshape(Xs,3*N);
    XMP = np.zeros(3);
    fibList = [None];
    fibList[0] = DiscretizedFiber(fibDisc,Xs,XMP);
    return fibList;

# Inputs 
nFib=1          # number of fibers
N=24          # number of points per fiber
Lf=2            # length of each fiber
nonLocal=4     # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=2.4          # length of the periodic domain
xi=3            # Ewald parameter
mu=1            # fluid viscosity
eps=1e-3*4/exp(1.5);       # slenderness ratio (aRPY/L=1e-3)
lpstar = float(sys.argv[1]);
tf = 0.5;
dt = 5e-4;
omega=0         # frequency of oscillations
gam0=0          # base rate of strain
kbT = 4.1e-3;
Eb=lpstar*kbT*Lf;         # fiber bending stiffness
penaltyParam = 1.6e4*kbT/Lf**3;
saveEvery = 1;
seed = int(sys.argv[2]);
eigvalThres =5.0/Lf 
RPYQuad = True;
RPYDirect = False;
RPYOversample = False;
NupsampleForDirect = 20;
if (N==12):
    eigvalThres = 3.2/Lf;
FileString = 'TESTCurved.txt';#'SemiflexFlucts/Penalty_N'+str(N)+'_Lp'+str(lpstar)+'_dt'+str(dtnum)+'_'+str(seed)+'.txt'

# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,deltaLocal=0,\
    RPYDirectQuad=RPYDirect,RPYOversample=RPYOversample,NupsampleForDirect=NupsampleForDirect,penaltyParam=penaltyParam);
fibList = makeCurvedFib(Lf,N,fibDisc);
fibDisc.calcBendMatX0(fibList[0]._X,penaltyParam);

# Initialize the master list of fibers
allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigvalThres);
allFibers.initFibList(fibList,Dom);


# Initialize the temporal integrator
TIntegrator = BackwardEuler(allFibers,FPimp=1);
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(1);

# Prepare the output file and write initial locations
np.random.seed(seed);
allFibers.writeFiberLocations(FileString,'w');
    
# Time loop
stopcount = int(tf/dt+1e-10);
for iT in range(stopcount): 
    TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,write=True,outfile=FileString);
