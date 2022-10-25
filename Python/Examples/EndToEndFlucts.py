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

def makeStraightFib(Lf,N,fibDisc):
    """
    Initialize the three fibers for the shear simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    q=1;
    s=fibDisc._sTau;
    # Falling fibers
    Xs = (np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T);
    Xs = np.reshape(Xs,3*N);
    XMP = np.zeros(3);
    fibList = [None];
    fibList[0] = DiscretizedFiber(fibDisc,Xs,XMP);
    #fibList[1] = DiscretizedFiber(fibDisc,Xs,XMP);
    return fibList;

# Inputs 
dtnum = int(sys.argv[2]);
nFib=1         # number of fibers
N=int(sys.argv[1]);            # number of points per fiber
Lf=2            # length of each fiber
nonLocal=4;   # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=2.4          # length of the periodic domain
xi=3            # Ewald parameter
mu=1            # fluid viscosity
eps=1e-3*4/exp(1.5);       # slenderness ratio (aRPY/L=1e-3)
Eb=1         # fiber bending stiffness
dt=10.0**(-dtnum);         # timestep
omega=0         # frequency of oscillations
gam0=0          # base rate of strain
tf=20      # final time
kbT = float(sys.argv[3]);
penaltyParam = 0*kbT;
saveEvery = max(1,int(1e-3/dt+1e-10));
seed = int(sys.argv[4]);
lp = Eb/kbT;
lpstar = int(lp/Lf+1e-10);
eigvalThres =5.0/Lf 
if (N==12):
    eigvalThres = 3.2/Lf; 
FileString = 'SemiflexFlucts/Eps3EndToEnd_N'+str(N)+'_Lp'+str(lpstar)\
    +'_dt'+str(dtnum)+'_'+str(seed)+'.txt'
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,trueRPYMobility=True,\
    UseEnergyDisc=True,penaltyParam=penaltyParam);

# Initialize the master list of fibers
allFibers = SemiflexiblefiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigvalThres);
fibList = makeStraightFib(Lf,N,fibDisc);
allFibers.initFibList(fibList,Dom);
allFibers._fiberDisc.calcBendMatX0(fibList[0]._X,penaltyParam);

# Initialize the temporal integrator
TIntegrator = BackwardEuler(allFibers,FPimp=(nonLocal > 0));
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(1);

# Prepare the output file and write initial locations
np.random.seed(seed);
# Barymat for the quarter points
#allFibers.writeFiberLocations(FileString,'w');
if (True):
    QuarterMat = cf.ResamplingMatrix(5,allFibers._fiberDisc._Nx,'u',allFibers._fiberDisc._XGridType);
    XLocs = allFibers._ptsCheb.copy();
    f=open(FileString,'w')
    np.savetxt(f, np.dot(QuarterMat,XLocs));#, fmt='%1.13f')
    f.close()
        
# Time loop
stopcount = int(tf/dt+1e-10);
for iT in range(stopcount): 
    wr = 0;
    if ((iT % saveEvery) == (saveEvery-1)):
        wr=1;
    TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,write=0,outfile=FileString);
    if (True and wr):
        #print('Time %1.2E' %(float(iT+1)*dt));
        XLocs = allFibers._ptsCheb.copy();
        f=open(FileString,'a')
        np.savetxt(f, np.dot(QuarterMat,XLocs));#, fmt='%1.13f')
        f.close()
        
